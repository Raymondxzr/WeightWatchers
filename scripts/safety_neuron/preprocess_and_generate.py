#!/usr/bin/env python3
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out_filtered", type=str, required=True)
    p.add_argument("--out_completions", type=str, required=True)

    p.add_argument("--num_sample", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--max_new_tokens", type=int, default=256)

    return p.parse_args()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    args = parse_args()

    MODEL = args.model
    DATASET = args.dataset
    OUT_FILTERED = args.out_filtered
    OUT_COMPLETIONS = args.out_completions

    NUM_SAMPLE = args.num_sample
    MAX_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size

    GEN_KWARGS = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False
    )

    os.makedirs(os.path.dirname(OUT_FILTERED), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_COMPLETIONS), exist_ok=True)

    # -----------------------------------------
    # Tokenizer & model
    # -----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    with open(DATASET, "r") as f:
        raw = [json.loads(x) for x in f]

    raw = raw[:NUM_SAMPLE]
    print(f"Loaded {len(raw)} raw datapoints.")

    # -----------------------------------------
    # Filter too-long datapoints
    # -----------------------------------------
    filtered = []
    for item in raw:
        text = item["prompt"]
        length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if length <= MAX_TOKENS:
            filtered.append(item)

    print(f"Filtered dataset: {len(filtered)} / {len(raw)} remain.")

    with open(OUT_FILTERED, "w") as f:
        for ex in filtered:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved filtered dataset → {OUT_FILTERED}")

    # -----------------------------------------
    # Generate completions (batched)
    # -----------------------------------------
    print("Generating completions in batches...")

    num_batches = (len(filtered) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(OUT_COMPLETIONS, "w") as f:
        for batch in tqdm(chunks(filtered, BATCH_SIZE), total=num_batches):
            prompts = [x["prompt"] for x in batch]
            task_ids = [x["task_id"] for x in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    **GEN_KWARGS,
                    pad_token_id=tokenizer.pad_token_id,
                )

            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            for task_id, prompt, text in zip(task_ids, prompts, texts):
                completion = text[len(prompt):] if text.startswith(prompt) else text
                f.write(json.dumps({
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion.strip()
                }) + "\n")

    print(f"Saved completions → {OUT_COMPLETIONS}")


if __name__ == "__main__":
    main()
