#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3
export HF_DATASETS_CACHE=Alignment/.cache
data_root=Alignment
model_root=models
BASE_MODEL=(Meta-Llama-3-8B-Instruct)
DATASET=(data/eval/codex_humaneval/HumanEval.jsonl)
DATASET_NAME=(HumanEval)
BATCH_SIZE=20
NUM_SAMPLE=200

python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY


for MODEL in ${BASE_MODEL[@]:0:1}
do
for ((j=0; j<${#DATASET[*]}; ++j))
do
    for seed in 1
    do
        for model in hh_harmless
        do
            PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_${model}_dpo_ia3_ff)
            peft_path=()
            for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
            do
                peft_path[i]=${data_root}/output/${MODEL}_${PEFT_NAME[i]}
            done

            python -m src.change_scores \
                --dataset ${data_root}/${DATASET[j]} \
                --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_sft_completion.pt \
                --model_name_or_path ${model_root}/${MODEL} \
                --tokenizer_name_or_path ${model_root}/${MODEL} \
                --first_peft_path ${peft_path[@]} \
                --second_peft_path ${peft_path[0]} \
                --eval_batch_size ${BATCH_SIZE} \
                --num_samples ${NUM_SAMPLE} 

            # python -m src.change_scores \
            #     --dataset ${data_root}/${DATASET[j]} \
            #     --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_dpo_completion.pt \
            #     --model_name_or_path ${model_root}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root}/${MODEL} \
            #     --first_peft_path ${peft_path[0]} \
            #     --second_peft_path ${peft_path[@]} \
            #     --eval_batch_size ${BATCH_SIZE} \
            #     --num_samples ${NUM_SAMPLE} 

            # python -m src.change_scores \
            #     --dataset ${data_root}/${DATASET[j]} \
            #     --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_prompt.pt \
            #     --model_name_or_path ${model_root}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root}/${MODEL} \
            #     --first_peft_path ${peft_path[@]} \
            #     --second_peft_path ${peft_path[0]} \
            #     --eval_batch_size ${BATCH_SIZE} \
            #     --num_samples ${NUM_SAMPLE} \
            #     --token_type prompt


            # python -m src.change_scores \
            #     --dataset ${data_root}/${DATASET[j]} \
            #     --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_prompt_last.pt \
            #     --model_name_or_path ${model_root}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root}/${MODEL} \
            #     --first_peft_path ${peft_path[@]} \
            #     --second_peft_path ${peft_path[0]} \
            #     --eval_batch_size ${BATCH_SIZE} \
            #     --num_samples -1 \
            #     --token_type prompt_last  
    
        done
    done
done
done
