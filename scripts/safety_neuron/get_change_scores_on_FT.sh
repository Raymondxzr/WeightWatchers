#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,4
export HF_DATASETS_CACHE=Alignment/.cache

# ---- Paths & settings ----
data_root=Alignment
model_root=models

# Base and full-finetuned (TAR-harden) checkpoints (adjust names if yours differ)
BASE_MODEL=(Meta-Llama-3-8B-Instruct)
FULL_FT_MODEL=(Meta-Llama-3-8B-Instruct-TARharden)

# DATASET=(data/eval/codex_humaneval/HumanEval.jsonl)
DATASET=(data/eval/harmbench/HarmBench_standard.jsonl)
DATASET_NAME=(harmbench)

BATCH_SIZE=4
NUM_SAMPLE=200
# /data/raymondxia/SafetyNeuron/Alignment/data/eval/harmbench/HarmBench_standard.jsonl
# ---- Sanity print for GPUs ----
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

# ---- Run change scores: generation by SECOND model (TAR), compare FIRST (base) vs SECOND (TAR) ----
# Note: With CUDA_VISIBLE_DEVICES=2,3,4,6, logical GPUs 0..3 map to physical 2,3,4,6.
for MODEL in ${BASE_MODEL[@]:0:1}; do
for FT in ${FULL_FT_MODEL[@]:0:1}; do
for ((j=0; j<${#DATASET[*]}; ++j)); do

  python -m src.change_scores \
    --dataset ${data_root}/${DATASET[j]} \
    --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_vs_${FT}_on_${DATASET_NAME[j]}_tar_completion.pt \
    --model_name_or_path ${model_root}/${MODEL} \
    --first_model_path ${model_root}/${MODEL} \
    --second_model_path ${model_root}/${FT} \
    --first_tokenizer_path ${model_root}/${MODEL} \
    --second_tokenizer_path ${model_root}/${MODEL} \
    --eval_batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLE} \
    --device_map balanced \
    --max_memory "0:70GiB,1:70GiB,2:70GiB" \
    --torch_dtype bf16

  # If you want base-completion instead (tokens from base), swap first/second:
  # python -m src.change_scores \
  #   --dataset ${data_root}/${DATASET[j]} \
  #   --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_vs_${FT}_on_${DATASET_NAME[j]}_base_completion.pt \
  #   --model_name_or_path ${model_root}/${MODEL} \
  #   --first_model_path ${model_root}/${FT} \
  #   --second_model_path ${model_root}/${MODEL} \
  #   --first_tokenizer_path ${model_root}/${MODEL} \
  #   --second_tokenizer_path ${model_root}/${MODEL} \
  #   --eval_batch_size ${BATCH_SIZE} \
  #   --num_samples ${NUM_SAMPLE} \
  #   --device_map balanced \
  #   --max_memory "0:74GiB,1:74GiB,2:74GiB,3:74GiB" \
  #   --torch_dtype bf16

  # To compute on prompt tokens instead of completion tokens:
  # add: --token_type prompt
  # Or for only the last prompt token:
  # add: --token_type prompt_last

done
done
done
