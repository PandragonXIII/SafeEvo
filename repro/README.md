# Reproduction notes: Llama-3-8B refusal circuit (Section 3.2 / Table 1)

This directory records an independent attempt to reproduce the refusal-circuit
extraction for Llama-3-8B, using the released code, the public LLM-LAT harmful
dataset, and the configuration in the top-level `README.md`. It is provided so the
authors can check where our results diverge.

The reproduction does not redistribute LLM-LAT. `run_lat_repro.sh` reads the
dataset from a local path and writes the derived pairs locally; only aggregate
statistics and the evaluation generations are included in `results/`.

## Environment

- 1 x NVIDIA A40 48 GB, Python 3.9, `torch 2.8.0+cu128`, `transformers 4.57.6`,
  `accelerate 1.10.1`, `peft 0.17.1`.
- Model: `meta-llama/Meta-Llama-3-8B` (base), bf16, 4 shards.
- Dataset: `LLM-LAT/harmful-dataset` at `8bfba31` (4,948 records). Fields are
  `prompt`, `chosen`, `rejected`; `chosen` is the safe refusal (400/400 sampled
  records contain an explicit refusal phrase in the first 300 characters) and is
  mapped to `accept`, while `rejected` is mapped to `refuse`.

## Configuration

Exactly the command in the top-level README: seed 1, the first 100 records for
training and the next 50 for validation, `max_length 256`, batch size 4, learning
rate 0.01, `init_value 0.2`, loss weights 1 / 1 / 0.05, a 100-epoch learning-rate
horizon, and stop after epoch 2 (50 optimizer updates). Deployment reads the
globally highest 10,376 raw MLP mask logits.

The base tokenizer has no `chat_template`. The README's built-in fallback is
`User: …\nAssistant:`, so two variants were run:

| Variant | Format |
|---|---|
| `llama3-8b-plain` | tokenizer fallback `User: …\nAssistant:` |
| `llama3-8b-chat` | explicit Llama 3 template in `llama3_chat.jinja` |

## Results

Training (best epoch is 2 in both runs; loss excludes the sparsity term):

| Variant | Best val loss | Active rows under the training gate |
|---|---:|---:|
| plain | 9.816214 | 827,287 / 1,048,576 (78.90%) |
| chat | 10.168178 | 882,443 / 1,048,576 (84.16%) |

Deployment support (global top-10,376 of 1,048,576 candidate rows, 0.99%):

| Variant | Minimum selected logit | Ties at the cutoff |
|---|---:|---:|
| plain | 0.500000 | 3,412 |
| chat | 0.48828125 | 875 |

Generation on 10 BeaverTails test records (`8401fe6`, `round0/330k/test`),
screened to `non_violent_unethical_behavior` and to exclude self-harm, sexual,
violent, and weapon-related requests. Greedy decoding, 128 new tokens, batch 4:

| Variant / branch | Reached 128 tokens | Refusal-like prefix | Degenerate repetition |
|---|---:|---:|---:|
| plain / base | 10/10 | 6/10 | 0/10 |
| plain / C1 (top-k) | 10/10 | 0/10 | 10/10 |
| plain / complement (top-k) | 10/10 | 0/10 | 0/10 (still loops) |
| chat / base | 10/10 | 0/10 | 2/10 |
| chat / C1 (top-k) | 10/10 | 0/10 | 10/10 |
| chat / complement (top-k) | 9/10 | 0/10 | 4/10 |
| plain / C1 (training gate, 80.4% active) | 10/10 | 9/10 | 0/10 |
| plain / complement (training gate) | 10/10 | 0/10 | 10/10 |

"Refusal-like" and "degenerate" are simple string heuristics, not the paper's
classifier; they are only meant to orient the reader. Full generations are in
`results/`.

Observations:

1. Under the paper's top-10,376 readout, C1 produces repetitive fragments
   (`seauseause…`, `202202…`) for every prompt in both template variants. The
   complement branch produces readable text but continues the conversation or
   loops.
2. With the training gate (`sigmoid(q) > 0.5`, i.e. `q > 0`), C1 emits coherent
   refusals. That gate keeps 80.4% of rows active, so this does not show that the
   sparse support alone is functional; the complement branch is degenerate under
   the same gate.
3. Mask logits take very few distinct values (3,313 distinct values across
   1,048,576 rows in the plain run), and the top-k cutoff falls exactly on a large
   tie group. The deployed support is therefore not uniquely determined by the
   readout rule.

## Reproduction commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# model at ~/models/Meta-Llama-3-8B, dataset at
# ~/datasets/LLM-LAT/harmful-dataset/train-8bfba31.parquet
bash repro/run_lat_repro.sh
```

The script picks an idle GPU, trains both variants, and writes masks, logs,
and generations to `outputs/repro-lat/`. `python repro/paper_check.py evaluate
--gate train` reproduces the training-gate diagnostic.

## Deviations and open questions

- The paper states a model-native chat template, but the base tokenizer ships
  none. We do not know which template the authors used; results for both are
  reported.
- Evaluation uses 10 screened out-of-distribution prompts, not the full
  BeaverTails set behind Table 1, and no LlamaGuard/refusal classifier, so the
  rates above are not comparable to Table 1.
- `max_length 256` truncates 42 of the first 150 pairs under a rough character
  estimate; the paper uses the same limit.
- Questions: which template and decoding settings produced Table 1; whether the
  same refusal criterion was applied to BeaverTails and HarmBench; how the
  top-k size was chosen; and why the training gate (`q > 0`) and the SCA reader
  (`q > 0.5`) differ from the diagnostic top-k readout.
