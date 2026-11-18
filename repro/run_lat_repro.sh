#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
host=$(hostname -s)
case "$host" in
    gpu1|gpu3|gpu4) ;;
    *) echo '请先 SSH 登录 gpu1、gpu3 或 gpu4。' >&2; exit 1 ;;
esac

model="$HOME/models/Meta-Llama-3-8B"
for file in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json \
    model-000{01,02,03,04}-of-00004.safetensors; do
    if [[ ! -s "$model/$file" ]]; then
        echo "模型未下载完整：$model/$file" >&2
        exit 1
    fi
done
for spec in 01:4976698672 02:4999802720 03:4915916176 04:1168138808; do
    file="$model/model-000${spec%:*}-of-00004.safetensors"
    if [[ $(stat -c %s "$file") != "${spec#*:}" ]]; then
        echo "模型分片大小不符：$file" >&2
        exit 1
    fi
done

gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | \
    awk -F, '$2 < 256 && $3 > 40000 && !found {print $1; found=1}')
if [[ -z "$gpu" ]]; then
    echo "$host 当前没有满足条件的空闲 GPU，未启动。" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "使用 $host 物理 GPU $gpu"

out=outputs/repro-lat
.venv/bin/python repro/paper_check.py lat --work "$out"

# Variant 1: repository default (no chat template, plain User:/Assistant:).
.venv/bin/python circuit/extract.py \
    --base_model_path "$model" --data_path "$out/lat_pairs.json" \
    --output_base "$out" --checkpoint_name llama3-8b-plain \
    --seed 1 --num_epochs 100 --stop_after_epoch 2 --batch_size 4 --lr 0.01 \
    --max_length 256 --train_samples 100 --val_samples 50 \
    --accept_weight 1 --refuse_weight 1 --sparsity_weight_mlp 0.05 \
    --init_value 0.2 --skip_post_generation 2>&1 | tee "$out/train-plain.log"
.venv/bin/python repro/paper_check.py evaluate --work "$out" --checkpoint llama3-8b-plain

# Variant 2: explicit Llama 3 template, as the paper describes model-native formatting.
.venv/bin/python circuit/extract.py \
    --base_model_path "$model" --data_path "$out/lat_pairs.json" \
    --chat_template_path repro/llama3_chat.jinja \
    --output_base "$out" --checkpoint_name llama3-8b-chat \
    --seed 1 --num_epochs 100 --stop_after_epoch 2 --batch_size 4 --lr 0.01 \
    --max_length 256 --train_samples 100 --val_samples 50 \
    --accept_weight 1 --refuse_weight 1 --sparsity_weight_mlp 0.05 \
    --init_value 0.2 --skip_post_generation 2>&1 | tee "$out/train-chat.log"
.venv/bin/python repro/paper_check.py evaluate --work "$out" --checkpoint llama3-8b-chat \
    --template repro/llama3_chat.jinja
