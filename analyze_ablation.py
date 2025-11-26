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
from src.ppl import layer_patch_hook

# Make allocator a bit more robust to fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ==========================================================
# CONFIG — EDIT THESE AS NEEDED
# ==========================================================
DATA_ROOT = "Alignment"

# Name of the base model folder inside neuron_activation
BASE_NAME = "Meta-Llama-3-8B-Instruct"

# Datasets to run over (slug used in paths; pretty is just for logging)
DATASETS = [
    # {"slug": "harmbench", "pretty": "HarmBench"},
    {"slug": "wmdp_cyber", "pretty": "WMDP_CYBER"},
    {"slug": "wmdp_bio", "pretty": "WMDP_BIO"},
]

# Fractions of top-change neurons to patch
FRACTIONS = [0, 0.001, 0.005]     # e.g., [0.0, 0.0005, 0.001]
NUM_SAMPLES = 8        # -1 = use all examples in filtered file
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4

SEED = 42
# Base models
MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-TARharden"
DONOR_MODEL_PATH = "models/Meta-Llama-3-8B-Instruct"
TOKENIZER_PATH = "models/Meta-Llama-3-8B-Instruct"


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
    index,
    batch_size=4,
    max_new_tokens=128,
    disable_tqdm=False,
    do_sample=False,
):
    """Generate completions from `target_model` while dynamically patching
    selected neurons using `donor_model` as the guided model.

    This is a thin wrapper around `generate_completions` that passes the
    `guided_model`, `index`, and `hook_fn` arguments through to
    `HookedModelBase.generate`, which implements the SafetyNeuron-style
    dynamic activation patching internally.
    """
    # `generate_completions` handles batching, decoding, and error handling.
    completions = generate_completions(
        target_model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        disable_tqdm=disable_tqdm,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        guided_model=donor_model,
        index=index,
        hook_fn=layer_patch_hook,
    )
    return completions


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
    log("Donor model loaded (8-bit).")

    # Put primary model on CUDA:0 explicitly (device_map may have done this already)
    # target_model.to(device)
    # donor_model.to(device)

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

            # Determine how many neurons to patch (as a tensor of [K, 2])
            k = int(total_neurons * FRACTION)
            index = neuron_ranks[:k]
            log(f"[{slug}] FRACTION={FRACTION:.4f}: patching top {k} neurons out of {total_neurons}.")

            # Compute remaining slice
            remaining_count = total_examples - done_n
            prompts_tail = prompts[done_n:]
            raw_prompts_tail = raw_prompts[done_n:]
            baseline_tail = baseline[done_n:]

            log(
                f"[{slug}] FRACTION={FRACTION:.4f}: "
                f"generating patched completions for {remaining_count} remaining examples..."
            )

            # ---- 4b. Dynamic patched completions using guided generation ----
            ablated_tail = generate_completions_dynamic_patch(
                donor_model=donor_model,
                target_model=target_model,
                tokenizer=tok,
                prompts=prompts_tail,
                index=index,
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
