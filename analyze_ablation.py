#!/usr/bin/env python3
import os
import json

import torch
import datasets
from tqdm.auto import tqdm

from src.utils import seed_torch
from src.eval.utils import load_hooked_lm_and_tokenizer, generate_completions
from src.eval.templates import create_prompt_with_tulu_chat_format
from src.neuron_ablation import register_neuron_intervention


# =======================
# CONFIG / PATHS / FLAGS
# =======================

OUT_DIR = "Alignment/ablation_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD = "dynamic_patch"  # or "zero", but this script is focused on dynamic_patch

# IMPORTANT: paper-style direction
#   - MODEL_PATH      = less-safe / base model   (target, gets patched)
#   - DONOR_MODEL_PATH = more-safe / aligned model (donor, provides activations)
MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-TARharden"              # target (base)
TOKENIZER_PATH = "models/Meta-Llama-3-8B-Instruct"
# DONOR_MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-TARharden"  # donor (aligned)
DONOR_MODEL_PATH = "models/Meta-Llama-3-8B-Instruct"  # donor (aligned)
# Single HarmBench change-scores file (base vs TAR-harden)
MERGED_CS_PATH = (
    "Alignment/hooked_llama/neuron_activation/"
    "Meta-Llama-3-8B-Instruct_vs_Meta-Llama-3-8B-Instruct-"
    "TARharden_on_harmbench_tar_completion.pt"
)

# HarmBench dataset (already converted to your HumanEval-style JSONL)
DATASET_PATH = "Alignment/data/eval/harmbench/HarmBench_standard.jsonl"

# Fractions of top-change neurons to patch
# FRACTIONS = [0.01, 0.005, 0.001, 0.0005]
FRACTIONS = [0]
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4
NUM_SAMPLES = 4   # -1 for all


# =======================================
# Helper: dynamic patch generation loop
# =======================================

def generate_completions_dynamic_patch(
    donor_model,
    target_model,
    tokenizer,
    prompts,
    patcher,
    batch_size=4,
    max_new_tokens=128,
    disable_tqdm=False,
    do_sample=False,
):
    """
    Dynamic activation patching with KV-cache, SafetyNeuron-style.

    donor_model: aligned (safe) model, provides cached activations
    target_model: less-safe (base) model, gets patched at selected neurons
    """
    donor_model.eval()
    target_model.eval()
    device = next(target_model.parameters()).device
    all_outputs = []

    indices = range(0, len(prompts), batch_size)
    iterator = indices if disable_tqdm else tqdm(indices, desc="Dynamic patching")

    for start in iterator:
        batch_prompts = prompts[start:start + batch_size]
        if not batch_prompts:
            break

        # Tokenize batch
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(device)          # [B, L]
        attention_mask = enc["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1)       # per-example prefix length

        batch_bs = input_ids.size(0)
        eos_token_id = tokenizer.eos_token_id

        # ---- 1) Initial full-prompt forward to set up caches ----
        patcher.reset()
        with torch.no_grad():
            donor_out = donor_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        donor_past = donor_out.past_key_values

        with torch.no_grad():
            target_out = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        target_past = target_out.past_key_values
        logits = target_out.logits[:, -1, :]              # [B, V]

        generated_ids = input_ids
        finished = torch.zeros(batch_bs, dtype=torch.bool, device=device)

        # ---- 2) Decode one token at a time with dynamic patching ----
        for _ in range(max_new_tokens):
            # 2a) choose next token from patched logits
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            if eos_token_id is not None:
                eos_fill = torch.full_like(next_tokens, eos_token_id)
                next_tokens = torch.where(finished, eos_fill, next_tokens)

            next_tokens_unsqueezed = next_tokens.unsqueeze(-1)   # [B, 1]
            generated_ids = torch.cat([generated_ids, next_tokens_unsqueezed], dim=1)

            # extend attention mask
            attention_mask = torch.cat(
                [attention_mask,
                 torch.ones_like(next_tokens_unsqueezed, device=device)],
                dim=1,
            )

            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)
                if finished.all():
                    break

            # 2b) donor: one-step forward with cache, refresh patcher.cache
            patcher.reset()
            with torch.no_grad():
                donor_out = donor_model(
                    input_ids=next_tokens_unsqueezed,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=donor_past,
                )
            donor_past = donor_out.past_key_values

            # 2c) target: one-step forward with cache, patched by donor activations
            with torch.no_grad():
                target_out = target_model(
                    input_ids=next_tokens_unsqueezed,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=target_past,
                )
            target_past = target_out.past_key_values
            logits = target_out.logits[:, -1, :]

        # ---- 3) Decode only the completion (after each prompt) ----
        batch_out = []
        for ids, plen in zip(generated_ids, prompt_lengths):
            plen = int(plen.item())
            completion_ids = ids[plen:]

            if eos_token_id is not None:
                eos_positions = (completion_ids == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    completion_ids = completion_ids[:eos_positions[0].item()]

            text = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            batch_out.append(text)

        all_outputs.extend(batch_out)

    return all_outputs


# =========
#   MAIN
# =========

def main():
    seed_torch(42)

    # ---- 1. Load change scores / neuron ranks ----
    change_scores, neuron_ranks, *_ = torch.load(MERGED_CS_PATH, map_location="cpu")
    total_neurons = neuron_ranks.shape[0]

    # ---- 2. Load dataset & build prompts ----
    ds = datasets.load_dataset("json", data_files=DATASET_PATH)["train"]
    if NUM_SAMPLES > 0:
        ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
    prompts = [
        create_prompt_with_tulu_chat_format(
            [{"role": "user", "content": p.strip()}],
            add_bos=False,
        )
        for p in ds["prompt"]
    ]
    print(f"Loaded {len(prompts)} HarmBench prompts.")

    # ---- 3. Load models (target: base, donor: aligned) ----
    # Target = base (less-safe) model
    target_model, tok = load_hooked_lm_and_tokenizer(
        model_name_or_path=MODEL_PATH,
        tokenizer_name_or_path=TOKENIZER_PATH,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
    )
    target_model.set_tokenizer(tok)

    # Donor = aligned (TAR-harden) model
    donor_model, _ = load_hooked_lm_and_tokenizer(
        model_name_or_path=DONOR_MODEL_PATH,
        tokenizer_name_or_path=TOKENIZER_PATH,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
    )
    donor_model.set_tokenizer(tok)

    # ---- 4. Baseline completions from target (base) model ----
    print("Generating baseline completions (target/base model)...")
    baseline = generate_completions(
        target_model,
        tok,
        prompts,
        batch_size=BATCH_SIZE,
        disable_tqdm=False,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    # ---- 5. Sweep fractions of neurons for dynamic patching ----
    for FRACTION in FRACTIONS:
        k = int(total_neurons * FRACTION)
        top_pairs = [tuple(map(int, xy)) for xy in neuron_ranks[:k].tolist()]
        print(f"\n=== Running fraction {FRACTION:.4f} ({k} neurons) ===")

        # Reset hooks so we don't stack perma-hooks across fractions
        if hasattr(target_model, "reset_hooks"):
            target_model.reset_hooks(including_permanent=True)
        if hasattr(donor_model, "reset_hooks"):
            donor_model.reset_hooks(including_permanent=True)

        # Register dynamic neuron patching (donor -> target)
        patcher = register_neuron_intervention(
            method="dynamic_patch",
            target_model=target_model,
            neuron_tuples=top_pairs,
            donor_model=donor_model,
        )

        # Generate patched completions
        ablated = generate_completions_dynamic_patch(
            donor_model=donor_model,
            target_model=target_model,
            tokenizer=tok,
            prompts=prompts,
            patcher=patcher,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            disable_tqdm=False,
            do_sample=False,
        )

        # ---- 6. Save outputs for this fraction ----
        out_path = os.path.join(
            OUT_DIR,
            f"base_patched_with_TARharden_fraction_{FRACTION:.4f}_on_HarmBench.jsonl",
        )
        with open(out_path, "w") as f:
            for p_raw, out0, out1 in zip(ds["prompt"], baseline, ablated):
                rec = {
                    "fraction": FRACTION,
                    "prompt": p_raw,
                    "baseline_completion": out0,
                    "patched_completion": out1,
                    "method": "dynamic_patch",
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Saved outputs -> {out_path}")

        # ---- 7. Print a few example diffs ----
        print("\nExample diffs for fraction", FRACTION)
        for i in range(min(2, len(prompts))):
            print("=" * 80)
            print("PROMPT:", ds["prompt"][i])
            print("--- BASELINE (target/base) ---")
            print(baseline[i])
            print("--- PATCHED (with aligned donor) ---")
            print(ablated[i])

    # Final cleanup
    if hasattr(target_model, "reset_hooks"):
        target_model.reset_hooks(including_permanent=True)
    if hasattr(donor_model, "reset_hooks"):
        donor_model.reset_hooks(including_permanent=True)


if __name__ == "__main__":
    main()
