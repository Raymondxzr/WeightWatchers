#!/usr/bin/env python3
"""
Convert WMDP dataset (cyber/bio/chem) into SafetyNeuron eval format.

Output format:
{
    "task_id": "wmdp_cyber/<index>",
    "prompt": "<question text + choices>",
    "entry_point": "",
    "canonical_solution": "",
    "test": ""
}

Run:
    python Alignment/data/eval/convert_wmdp_to_humaneval.py
"""

import os
import json
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================

# Use short names here:
DOMAIN = "bio"   # "cyber", "bio", or "chem"
SPLIT = "test"

# Map to actual HF builder configs
HF_CONFIG_MAP = {
    "cyber": "wmdp-cyber",
    "bio": "wmdp-bio",
    "chem": "wmdp-chem",
}

if DOMAIN not in HF_CONFIG_MAP:
    raise ValueError(f"Unknown DOMAIN={DOMAIN!r}, must be one of {list(HF_CONFIG_MAP)}")

HF_CONFIG = HF_CONFIG_MAP[DOMAIN]

OUT_DIR = "Alignment/data/eval/"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, f"wmdp_{DOMAIN}.jsonl")


# ============================================================
# LOAD WMDP SPLIT
# ============================================================

print(f"Loading WMDP config='{HF_CONFIG}', split='{SPLIT}' ...")

dataset = load_dataset(
    "cais/wmdp",
    HF_CONFIG,   # <-- this was the bug: must use 'wmdp-cyber', etc.
    split=SPLIT,
)

print("Columns in dataset:", dataset.column_names)
# Typically: ['answer', 'question', 'choices']


# ============================================================
# CONVERSION
# ============================================================

def build_prompt(example):
    """
    Create a unified prompt string from:
        question
        choices (multiple choice options)
    """
    q = (example.get("question") or "").strip()
    choices = example.get("choices", [])

    if choices and isinstance(choices, list):
        # Format as A., B., C., ...
        choices_text = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        return f"{q}\n\nChoices:\n{choices_text}"

    return q


print(f"Writing converted JSONL to {OUT_PATH} ...")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, ex in enumerate(dataset):
        obj = {
            "task_id": f"wmdp_{DOMAIN}/{i}",
            "prompt": build_prompt(ex),
            "entry_point": "",
            "canonical_solution": "",
            "test": "",
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done!")
print(f"Saved → {OUT_PATH}")
