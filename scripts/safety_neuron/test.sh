#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_DATASETS_CACHE=Alignment/.cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---- Which GPUs to use ----
GPUS=(2 3 4 6)

# ---- Paths & settings ----
data_root=Alignment
model_root=models

BASE_MODEL=(Meta-Llama-3-8B-Instruct)
FULL_FT_MODEL=(Meta-Llama-3-8B-Instruct-TARharden)

DATASET=(data/eval/codex_humaneval/HumanEval.jsonl)
DATASET_NAME=(HumanEval)

# Use smaller per-process load to avoid host RAM OOM
BATCH_SIZE=4
NUM_SAMPLE=-1        # use all lines in shard
MAX_NEW_TOKENS=96   # a bit smaller than 256 to save memory
DTYPE=bf16

# ---- Sanity print for GPUs ----
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES (global) =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Physical GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

# ---- Run 4 processes in parallel: one shard per GPU ----
for MODEL in ${BASE_MODEL[@]:0:1}; do
for FT in ${FULL_FT_MODEL[@]:0:1}; do
for ((j=0; j<${#DATASET[*]}; ++j)); do

  SRC="${data_root}/${DATASET[j]}"
  # split into 4 parts: ${SRC}.00 .. ${SRC}.03
  split -n l/4 -d --additional-suffix="" "${SRC}" "${SRC}."

  pids=()
  for idx in 0 1 2 3; do
    GPU=${GPUS[$idx]}
    SHARD="${SRC}.$(printf "%02d" ${idx})"
    OUT="${data_root}/hooked_llama/neuron_activation/${MODEL}_vs_${FT}_on_${DATASET_NAME[j]}_tar_completion.shard${idx}.pt"

    echo "[LAUNCH] shard ${idx} on GPU ${GPU} -> ${OUT}"
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -m src.change_scores \
      --dataset "${SHARD}" \
      --output_file "${OUT}" \
      --model_name_or_path ${model_root}/${MODEL} \
      --first_model_path ${model_root}/${MODEL} \
      --second_model_path ${model_root}/${FT} \
      --first_tokenizer_path ${model_root}/${MODEL} \
      --second_tokenizer_path ${model_root}/${MODEL} \
      --eval_batch_size ${BATCH_SIZE} \
      --num_samples ${NUM_SAMPLE} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --torch_dtype ${DTYPE} \
      >change_scores_shard${idx}.log 2>&1 &

    pids+=($!)
  done

  # wait for all 4 shards
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "[OK] All shards done for ${DATASET_NAME[j]}"

done
done
done
