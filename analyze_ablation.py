#!/usr/bin/env python3
"""
Single-GPU neuron ablation evaluator with dynamic patching.

- Assumes change-score results are stored at:
    Alignment/hooked_llama/neuron_activation/<slug>/<BASE_NAME>/completion.pt

- Writes dynamic patch outputs to:
    Alignment/hooked_llama/neuron_activation/<slug>/<BASE_NAME>/Dynamic_patching_{FRACTION:.4f}.jsonl
"""

import os
import json
from datetime import datetime

import torch
import datasets
from tqdm.auto import tqdm

from src.utils import seed_torch
from src.eval.utils import load_hooked_lm_and_tokenizer, generate_completions
from src.eval.templates import create_prompt_with_tulu_chat_format
from src.neuron_ablation import register_neuron_intervention

# Make allocator a bit more robust to fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ==========================================================
# CONFIG — EDIT THESE AS NEEDED
# ==========================================================
DATA_ROOT = "Alignment"

OUT_DIR = os.path.join(DATA_ROOT, "ablation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Target (less-safe / TAR-hardened) and donor (aligned / base) models
MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-TARharden"
DONOR_MODEL_PATH = "models/Meta-Llama-3-8B-Instruct"
TOKENIZER_PATH = "models/Meta-Llama-3-8B-Instruct"

# Name of the base model folder inside neuron_activation
BASE_NAME = "Meta-Llama-3-8B-Instruct"

# Datasets to run over (slug used in paths; pretty is just for logging)
DATASETS = [
    {"slug": "harmbench", "pretty": "HarmBench"},
    {"slug": "wmdp", "pretty": "WMDP"},
]

# Fractions of top-change neurons to patch
FRACTIONS = [0.001]     # e.g., [0.0, 0.0005, 0.001]
NUM_SAMPLES = -1        # -1 = use all examples in filtered file
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4

SEED = 42
# ==========================================================


# ==========================================================
# Logging helper
# ==========================================================
def log(msg: str) -> None:
    """Print a message with ISO timestamp."""
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


# ==========================================================
# Dynamic Patch Generation
# ==========================================================
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
    Single process; uses whatever GPUs are visible.
    """
    donor_model.eval()
    target_model.eval()
    device = next(target_model.parameters()).device
    all_outputs = []

    for start in tqdm(
        range(0, len(prompts), batch_size),
        disable=disable_tqdm,
        desc="Dynamic patching",
    ):
        batch_prompts = prompts[start:start + batch_size]
        if not batch_prompts:
            break

        tok_out = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = tok_out["input_ids"].to(device)
        attention_mask = tok_out["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1)
        B = input_ids.size(0)
        eos_token_id = tokenizer.eos_token_id

        # 1) Initial full-prompt forward (set up caches)
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
        logits = target_out.logits[:, -1, :]

        generated_ids = input_ids
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # 2) Token-by-token decode with dynamic patching
        for _ in range(max_new_tokens):
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            generated_ids = torch.cat(
                [generated_ids, next_token.unsqueeze(-1)], dim=1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, device=device)], dim=1
            )

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all():
                    break

            # donor model step (refresh cache)
            patcher.reset()
            with torch.no_grad():
                donor_out = donor_model(
                    input_ids=next_token.unsqueeze(-1),
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=donor_past,
                )
            donor_past = donor_out.past_key_values

            # target model step (patched)
            with torch.no_grad():
                target_out = target_model(
                    input_ids=next_token.unsqueeze(-1),
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=target_past,
                )
            target_past = target_out.past_key_values
            logits = target_out.logits[:, -1, :]

        # 3) Decode completions (strip prompt + EOS)
        batch_outputs = []
        for ids, plen in zip(generated_ids, prompt_lengths):
            cids = ids[int(plen.item()):]
            if eos_token_id is not None:
                eos_positions = (cids == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    cids = cids[:eos_positions[0].item()]
            text = tokenizer.decode(cids, skip_special_tokens=True)
            batch_outputs.append(text)

        all_outputs.extend(batch_outputs)

    return all_outputs


# ==========================================================
# === MAIN (single entry point)
# ==========================================================
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    log(f"CUDA visible devices: {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    log(f"Using primary device: {device}")

    seed_torch(SEED)
    log(f"Seeded all RNGs with SEED={SEED}")

    # ---- Load models once, reuse across datasets ----
    log(f"Loading target (TAR-hardened) model from: {MODEL_PATH}")
    target_model, tok = load_hooked_lm_and_tokenizer(
        model_name_or_path=MODEL_PATH,
        tokenizer_name_or_path=TOKENIZER_PATH,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=True,  # explicitly half-precision to save VRAM
    )
    target_model.set_tokenizer(tok)
    log("Target model loaded.")

    log(f"Loading donor (base) model in 8-bit from: {DONOR_MODEL_PATH}")
    donor_model, _ = load_hooked_lm_and_tokenizer(
        model_name_or_path=DONOR_MODEL_PATH,
        tokenizer_name_or_path=TOKENIZER_PATH,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=True,      # big VRAM saving; safe for donor
        convert_to_half=False,
    )
    donor_model.set_tokenizer(tok)
    log("Donor model loaded.")

    # ==================================================
    # Loop over datasets
    # ==================================================
    for cfg in DATASETS:
        slug = cfg["slug"]        # e.g., 'harmbench' / 'wmdp'
        pretty = cfg["pretty"]    # e.g., 'HarmBench' / 'WMDP'

        log("=" * 80)
        log(f"Running dataset: {pretty} (slug='{slug}')")
        log("=" * 80)

        # ---- Paths for change-scores and dataset ----
        # change scores (your completion.pt)
        merged_cs_path = os.path.join(
            DATA_ROOT,
            "hooked_llama",
            "neuron_activation",
            slug,
            BASE_NAME,
            "completion.pt",
        )

        # filtered dataset
        dataset_path = os.path.join(
            DATA_ROOT,
            "data",
            "eval",
            slug,
            f"{slug}_filtered.jsonl",
        )

        # output directory for dynamic patching jsonl
        out_dir_ds = os.path.join(
            DATA_ROOT,
            "hooked_llama",
            "neuron_activation",
            slug,
            BASE_NAME,
        )
        os.makedirs(out_dir_ds, exist_ok=True)

        log(f"Change-score file: {merged_cs_path}")
        log(f"Filtered dataset: {dataset_path}")
        log(f"Output directory: {out_dir_ds}")

        if not os.path.exists(merged_cs_path):
            raise FileNotFoundError(f"Missing change-score file: {merged_cs_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Missing filtered dataset: {dataset_path}")

        # ---- 1. Load change scores / neuron ranks ----
        log("Loading change scores & neuron ranks...")
        change_scores, neuron_ranks, *_ = torch.load(merged_cs_path, map_location="cpu")
        total_neurons = neuron_ranks.shape[0]
        log(f"Total neurons ranked: {total_neurons}")

        # ---- 2. Load dataset & build prompts ----
        log("Loading filtered dataset with datasets.load_dataset...")
        ds = datasets.load_dataset("json", data_files=dataset_path)["train"]
        if NUM_SAMPLES > 0:
            ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
        log(f"Loaded {len(ds)} prompts from {dataset_path} (NUM_SAMPLES={NUM_SAMPLES})")

        raw_prompts = ds["prompt"]
        prompts = [
            create_prompt_with_tulu_chat_format(
                [{"role": "user", "content": p.strip()}],
                add_bos=False,
            )
            for p in raw_prompts
        ]

        # ---- 3. Baseline completions from target (no patch) ----
        log("Generating baseline completions with target (TAR-hardened) model...")
        baseline = generate_completions(
            target_model,
            tok,
            prompts,
            batch_size=BATCH_SIZE,
            disable_tqdm=False,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        log("Baseline generation complete.")

        total_examples = len(prompts)

        # ---- 4. Sweep fractions of neurons for dynamic patching ----
        for FRACTION in FRACTIONS:
            # Resume logic: check if output file already has some lines
            out_path = os.path.join(
                out_dir_ds,
                f"Dynamic_patching_{FRACTION:.4f}.jsonl",
            )

            done_n = 0
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    done_n = sum(1 for line in f if line.strip())

                if done_n >= total_examples:
                    log(
                        f"[{slug}] FRACTION={FRACTION:.4f}: "
                        f"already complete ({done_n}/{total_examples}), skipping."
                    )
                    continue
                else:
                    log(
                        f"[{slug}] FRACTION={FRACTION:.4f}: "
                        f"resuming from example {done_n}/{total_examples}."
                    )
            else:
                log(
                    f"[{slug}] FRACTION={FRACTION:.4f}: "
                    f"starting fresh for {total_examples} examples."
                )

            # Determine how many neurons to patch
            k = int(total_neurons * FRACTION)
            top_pairs = [tuple(map(int, xy)) for xy in neuron_ranks[:k].tolist()]
            log(
                f"[{slug}] FRACTION={FRACTION:.4f}: "
                f"patching top {k} neurons out of {total_neurons}."
            )

            # Reset hooks to avoid stacking
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

            # Compute remaining slice
            remaining_count = total_examples - done_n
            prompts_tail = prompts[done_n:]
            raw_prompts_tail = raw_prompts[done_n:]
            baseline_tail = baseline[done_n:]

            log(
                f"[{slug}] FRACTION={FRACTION:.4f}: "
                f"generating patched completions for {remaining_count} remaining examples..."
            )

            ablated_tail = generate_completions_dynamic_patch(
                donor_model=donor_model,
                target_model=target_model,
                tokenizer=tok,
                prompts=prompts_tail,
                patcher=patcher,
                batch_size=BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
                disable_tqdm=False,
                do_sample=False,
            )

            # ---- 5. Save outputs (append if resuming) ----
            mode = "a" if os.path.exists(out_path) and done_n > 0 else "w"
            written = 0
            with open(out_path, mode) as f:
                for p_raw, out0, out1 in zip(
                    raw_prompts_tail, baseline_tail, ablated_tail
                ):
                    rec = {
                        "dataset": slug,
                        "fraction": FRACTION,
                        "prompt": p_raw,
                        "baseline_completion": out0,
                        "patched_completion": out1,
                        "method": "dynamic_patch",
                    }
                    f.write(json.dumps(rec) + "\n")
                    written += 1

            log(
                f"[{slug}] FRACTION={FRACTION:.4f}: "
                f"wrote {written} records to {out_path} "
                f"(total now ≈ {done_n + written}/{total_examples})."
            )

    # Final cleanup
    if hasattr(target_model, "reset_hooks"):
        target_model.reset_hooks(including_permanent=True)
    if hasattr(donor_model, "reset_hooks"):
        donor_model.reset_hooks(including_permanent=True)
    log("All datasets completed. Hooks reset and script finished.")


if __name__ == "__main__":
    main()
