"""Prepare extraction inputs and compare intact, refusal, and complement branches."""
import argparse
import json
import lzma
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "circuit"))
MODEL = Path.home() / "models" / "Meta-Llama-3-8B"
TEMPLATE = Path(__file__).with_name("llama3_chat.jinja")
BEAVER = Path.home() / "datasets" / "PKU-Alignment" / "BeaverTails"
LAT = Path.home() / "datasets" / "LLM-LAT" / "harmful-dataset" / "train-8bfba31.parquet"
EVAL_ROWS = (401, 2658, 6389, 7769, 8840, 12915, 14617, 24405, 24721, 29479)
TOP_K = 10376


def write_beavertails(work):
    source, = BEAVER.glob("*/round0/330k/test.jsonl.xz")
    rows = []
    with lzma.open(source, "rt", encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i in EVAL_ROWS:
                row = json.loads(line)
                assert not row["is_safe"]
                assert [name for name, value in row["category"].items() if value] == [
                    "non_violent_unethical_behavior"
                ]
                rows.append({"row_index": i, "prompt": row["prompt"]})
    assert len(rows) == len(EVAL_ROWS)
    (work / "beavertails_subset.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def prepare_lat(work):
    """Map LLM-LAT chosen (refusal) to accept and rejected (compliance) to refuse."""
    import pyarrow.parquet as pq

    columns = pq.ParquetFile(LAT).read(columns=["prompt", "chosen", "rejected"]).to_pylist()
    assert len(columns) == 4948
    pairs = [
        {"prompt": row["prompt"], "accept": row["chosen"], "refuse": row["rejected"]}
        for row in columns
    ]
    (work / "lat_pairs.json").write_text(json.dumps({"results": pairs}, indent=2))


def top_k_support(state, k):
    """Break ties by saved module order and then row index."""
    import torch

    logits = torch.cat([v.flatten().float() for v in state.values()])
    assert torch.isfinite(logits).all() and 0 < k <= logits.numel()
    order = logits.argsort(descending=True, stable=True)
    selected = torch.zeros_like(logits, dtype=torch.bool)
    selected[order[:k]] = True
    support = {
        name: chunk.reshape_as(state[name])
        for name, chunk in zip(state, selected.split([v.numel() for v in state.values()]))
    }
    return support, {
        "total_rows": logits.numel(),
        "selected_rows": k,
        "minimum_selected_logit": float(logits[order[k - 1]]),
        "ties_at_cutoff": int((logits == logits[order[k - 1]]).sum()),
    }


def evaluate(work, checkpoint, template=None, gate="topk"):
    import torch
    from transformers import AutoModelForCausalLM
    from extract import _format_prompt, generation_stop_ids, load_tokenizer
    from matrix_masks import assign_mask_logits, install_masks, partition

    mask_file = work / checkpoint / "final_safety_masks.pt"
    state = torch.load(mask_file, map_location="cpu", weights_only=True)
    if gate == "topk":
        support, stats = top_k_support(state, TOP_K)
    else:
        # The extraction gate itself: hard sigmoid(q) > 0.5, equivalently q > 0.
        support = {name: value > 0 for name, value in state.items()}
        stats = {"readout": "training gate q > 0", "selected_rows": sum(int(v.sum()) for v in support.values())}
    tokenizer = load_tokenizer(MODEL, template)
    tokenizer.padding_side = "left"
    stop_ids = generation_stop_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
    ).eval()
    rows = json.loads((work / "beavertails_subset.json").read_text())

    for branch, complement in (("base", False), ("c1", False), ("complement", True)):
        if branch == "c1":
            install_masks(model)
            # Signed logits implement the exact binary support.
            assert assign_mask_logits(model, {
                name: value.float() * 2 - 1 for name, value in support.items()
            })["mlp"] == len(support)
        with torch.inference_mode(), partition(model, complement=complement):
            for start in range(0, len(rows), 4):
                batch = rows[start:start + 4]
                inputs = tokenizer(
                    [_format_prompt(tokenizer, item["prompt"]) for item in batch],
                    padding=True, return_tensors="pt", add_special_tokens=False,
                ).to(model.device)
                ids = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
                )[:, inputs.input_ids.shape[1]:]
                for item, tokens in zip(batch, ids):
                    values = tokens.tolist()
                    stop = next((i for i, token in enumerate(values) if token in stop_ids), len(values))
                    item[branch] = {
                        "text": tokenizer.decode(values[:stop], skip_special_tokens=True),
                        "new_tokens": stop,
                        "eos_seen": stop < len(values),
                        "stop_token": tokenizer.convert_ids_to_tokens(values[stop]) if stop < len(values) else None,
                    }
    result = {
        "mask_checkpoint": str(mask_file),
        "training_data": "lat_pairs.json (LLM-LAT harmful-dataset, 4948 records; first 100/50 split)",
        "eval_data": "BeaverTails 8401fe6 / round0 / 330k / test (manually screened 10 rows)",
        "template": str(template.resolve().relative_to(ROOT)) if template else "plain 'User: …\\nAssistant:'",
        "decoding": {"do_sample": False, "max_new_tokens": 128, "batch_size": 4, "stop_ids": stop_ids},
        "gate": gate,
        "support": stats,
        "results": rows,
    }
    suffix = "" if gate == "topk" else "-traingate"
    (work / f"branch_generations-{checkpoint}{suffix}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("lat", "evaluate"))
    parser.add_argument("--work", type=Path, default=ROOT / "outputs" / "repro-lat")
    parser.add_argument("--checkpoint", default="llama3-8b-plain")
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--gate", choices=("topk", "train"), default="topk")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.step == "lat":
        prepare_lat(args.work)
        write_beavertails(args.work)
    else:
        evaluate(args.work, args.checkpoint, args.template, args.gate)


if __name__ == "__main__":
    main()
