#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3
export HF_DATASETS_CACHE=Alignment/.cache

data_root=Alignment
model_root=models

BASE_MODEL=(Meta-Llama-3-8B-Instruct)

# ---- Raw datasets (your inputs) ----
RAW_DATASET=(
    # data/eval/harmbench/HarmBench_standard.jsonl
    # data/eval/wmdp/wmdp_cyber.jsonl
    data/eval/wmdp/wmdp_bio.jsonl
)

DATASET_NAME=(
    # harmbench
    # wmdp_cyber
    wmdp_bio
)

BATCH_SIZE=4
NUM_SAMPLE=100
TOKEN_TYPE=prompt

###########################################
# Print GPU visibility
###########################################
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY


###########################################
#                MAIN LOOP
###########################################
for MODEL in ${BASE_MODEL[@]:0:1}
do
for ((j=0; j<${#RAW_DATASET[*]}; ++j))
do
    RAW_DS=${RAW_DATASET[j]}
    DS_NAME=${DATASET_NAME[j]}

    echo ""
    echo "======================================================"
    echo "              RUNNING DATASET: $DS_NAME"
    echo "              RAW INPUT FILE: $RAW_DS"
    echo "======================================================"

    ###########################################
    # Step 1 — Construct paths
    ###########################################
    CACHE_DIR="${data_root}/data/eval/${DS_NAME}"
    mkdir -p "$CACHE_DIR"

    FILTER_PATH="${CACHE_DIR}/${DS_NAME}_filtered.jsonl"
    CACHE_PATH="${CACHE_DIR}/${DS_NAME}_completions.jsonl"

    echo "[INFO] Filtered dataset path:   $FILTER_PATH"
    echo "[INFO] Completions cache path:  $CACHE_PATH"
    echo ""

    ###########################################
    # Step 2 — Preprocess + generate completion
    ###########################################
    PRE_LOG="${CACHE_DIR}/preprocess.log"

    if [ ! -f "$FILTER_PATH" ]; then
        echo "------------------------------------------------------"
        echo "[STEP 1] No filtered dataset found."
        echo "[ACTION] Running preprocess_and_generate.py..."
        echo "         Log → $PRE_LOG"
        echo "------------------------------------------------------"

        nohup python scripts/safety_neuron/preprocess_and_generate.py \
            --model ${model_root}/${MODEL} \
            --dataset ${data_root}/${RAW_DS} \
            --out_filtered ${FILTER_PATH} \
            --out_completions ${CACHE_PATH} \
            --num_sample ${NUM_SAMPLE} \
            --batch_size ${BATCH_SIZE} \
            > "$PRE_LOG" 2>&1

        echo "[DONE] Preprocess + completions finished for $DS_NAME"
        echo "[LOG] See: $PRE_LOG"

    else
        echo "------------------------------------------------------"
        echo "[CACHE] Filtered dataset already exists → $FILTER_PATH"
        echo "[CACHE] Completions cache → $CACHE_PATH"
        echo "[CACHE] Skipping preprocessing."
        echo "------------------------------------------------------"
    fi

    ###########################################
    # Step 3 — Activation / change-score scoring
    ###########################################
    OUT_DIR="${data_root}/hooked_llama/neuron_activation/${DS_NAME}/${MODEL}"
    mkdir -p "$OUT_DIR"

    SCORE_OUTPUT="${OUT_DIR}/completion.pt"
    SCORE_LOG="${OUT_DIR}/activation.log"

    if [ -f "$SCORE_OUTPUT" ]; then
        echo "------------------------------------------------------"
        echo "[SKIP] Activation score file already exists:"
        echo "       $SCORE_OUTPUT"
        echo "[SKIP] Skipping activation scoring step."
        echo "------------------------------------------------------"
    else
        echo ""
        echo "------------------------------------------------------"
        echo "[STEP 2] Running activation scoring..."
        echo "[INPUT]  Filtered dataset: $FILTER_PATH"
        echo "[OUTPUT] Scores → $SCORE_OUTPUT"
        echo "[LOG]    Log file → $SCORE_LOG"
        echo "------------------------------------------------------"

        nohup python -m src.change_scores \
            --dataset ${FILTER_PATH} \
            --output_file ${SCORE_OUTPUT} \
            --token_type ${TOKEN_TYPE} \
            \
            --first_model_path ${model_root}/Meta-Llama-3-8B-Instruct \
            --second_model_path ${model_root}/Meta-Llama-3-8B-Instruct-TARharden \
            --first_tokenizer_path ${model_root}/Meta-Llama-3-8B-Instruct \
            --second_tokenizer_path ${model_root}/Meta-Llama-3-8B-Instruct \
            \
            --eval_batch_size ${BATCH_SIZE} \
            --num_samples ${NUM_SAMPLE} \
            --completion_cache_path ${CACHE_PATH} \
            > "$SCORE_LOG" 2>&1

        echo "[DONE] Activation scoring completed for $DS_NAME"
        echo "[LOG]  See: $SCORE_LOG"
    fi


    ###########################################
    # Step 4 — Postprocess scores (top-K + plots)
    ###########################################
    if [ -f "$SCORE_OUTPUT" ]; then
        echo "------------------------------------------------------"
        echo "[STEP 3] Analyzing change scores for $DS_NAME / $MODEL"
        echo "         Score file: $SCORE_OUTPUT"
        echo "------------------------------------------------------"

        python analyze.py \
            --score_file "$SCORE_OUTPUT" \
            --topk 5000
    else
        echo "------------------------------------------------------"
        echo "[WARN] Score file not found, skipping analysis:"
        echo "       $SCORE_OUTPUT"
        echo "------------------------------------------------------"
    fi

done
done
