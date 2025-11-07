#!/usr/bin/env python3
"""
Convert HarmBench behaviors CSV to a HumanEval-style JSONL.

Input columns:
    Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID

Output schema:
    {
        "task_id": "HarmBench_standard/<BehaviorID or idx>",
        "prompt": "<ContextString>\n\n<Behavior>",
        "entry_point": "",
        "canonical_solution": "",
        "test": ""
    }
"""

import csv
import json
from pathlib import Path

# ==========================
# HARD-CODED CONFIG
# ==========================

INPUT_CSV = Path(
    "/data/raymondxia/SafetyNeuron/Alignment/data/eval/harmbench/harmbench_behaviors_text_val.csv"
)

OUTPUT_JSONL = Path(
    "/data/raymondxia/SafetyNeuron/Alignment/data/eval/harmbench/HarmBench_standard.jsonl"
)

TASK_PREFIX = "HarmBench_standard"

BEHAVIOR_COL = "Behavior"
CONTEXT_COL = "ContextString"
ID_COL = "BehaviorID"


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open("r", encoding="utf-8") as f_in, \
         OUTPUT_JSONL.open("w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)

        for idx, row in enumerate(reader):
            # Basic column sanity check
            for col in (BEHAVIOR_COL, CONTEXT_COL):
                if col not in row:
                    raise KeyError(
                        f"Expected column '{col}' in CSV, "
                        f"but got columns: {list(row.keys())}"
                    )

            behavior = (row[BEHAVIOR_COL] or "").strip()
            context = (row[CONTEXT_COL] or "").strip()

            if context:
                prompt = context + "\n\n" + behavior
            else:
                prompt = behavior

            beh_id = (row.get(ID_COL) or "").strip()
            if beh_id:
                task_id = f"{TASK_PREFIX}/{beh_id}"
            else:
                task_id = f"{TASK_PREFIX}/{idx}"

            rec = {
                "task_id": task_id,
                "prompt": prompt,
                "entry_point": "",
                "canonical_solution": "",
                "test": "",
            }
            f_out.write(json.dumps(rec) + "\n")

    print(f"Wrote {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
