# src/change_scores.py

import os
import argparse
from typing import Dict, Optional

import torch
import datasets

from src.utils import seed_torch
from src.activation_processor import ActivationContrasting
from src.eval.templates import create_prompt_with_tulu_chat_format


# ---------- Helpers ----------

def _parse_max_memory(s: Optional[str]) -> Optional[Dict[int, str]]:
    if not s:
        return None
    out: Dict[int, str] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(":")
        out[int(k.strip())] = v.strip()
    return out


def _parse_torch_dtype(s: str):
    if not isinstance(s, str):
        return "auto"
    key = s.lower()
    mapping = {
        "auto": "auto",
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    return mapping.get(key, "auto")


# ---------- Main ----------

def main(args):
    seed_torch(42)

    # Load dataset
    eval_data = datasets.load_dataset("json", data_files=args.dataset)["train"]["prompt"]
    if args.num_samples > 0:
        eval_data = eval_data[:args.num_samples]

    # Build formatted prompts
    prompts = []
    for example in eval_data:
        prompt = example.strip()
        messages = [{"role": "user", "content": prompt}]
        prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(prompt + args.generation_startswith)

    names_filter = lambda name: name.endswith("hook_post")

    # Loader kwargs
    load_kwargs = {
        "device_map": args.device_map,
        "torch_dtype": _parse_torch_dtype(args.torch_dtype),
        "load_in_8bit": args.load_in_8bit,
        "convert_to_half": args.convert_to_half,
        "use_fast_tokenizer": True,
        "padding_side": "left",
    }
    max_mem = _parse_max_memory(args.max_memory)
    if max_mem:
        load_kwargs["max_memory"] = max_mem

    # Instantiate contrasting object
    ac = ActivationContrasting(
        base_model_name_or_path=args.model_name_or_path,
        first_peft_path=args.first_peft_path,
        second_peft_path=args.second_peft_path,
        first_model_name_or_path=args.first_model_path,
        second_model_name_or_path=args.second_model_path,
        first_tokenizer_name_or_path=args.first_tokenizer_path or args.tokenizer_name_or_path,
        second_tokenizer_name_or_path=args.second_tokenizer_path or args.tokenizer_name_or_path,
        batchsize=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        **load_kwargs,
    )

    # NEW: inject completion cache path into processor
    if args.completion_cache_path:
        print(f"[info] Using cached completions at: {args.completion_cache_path}")
        ac._completion_cache_path = args.completion_cache_path

    # Compute scores
    change_scores, neuron_ranks, first_mean, first_std, second_mean, second_std = ac.compute_change_scores(
        prompts, names_filter, args.token_type
    )

    # Save outputs
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        (
            change_scores.cpu(),
            neuron_ranks.cpu(),
            first_mean.cpu(),
            first_std.cpu(),
            second_mean.cpu(),
            second_std.cpu(),
        ),
        args.output_file,
    )
    print(f"[change_scores] Saved results to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute change scores via generation-time activation contrasting."
    )

    # Generation / eval
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--token_type",
        type=str,
        default="completion",
        choices=["prompt", "prompt_last", "completion"],
    )
    parser.add_argument("--generation_startswith", type=str, default="")

    # Data I/O
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output_file", type=str, default="../data/default.pt")

    # NEW: cache path for completions  
    parser.add_argument(
        "--completion_cache_path",
        type=str,
        default=None,
        help="If provided, load cached completions instead of regenerating.",
    )

    # Model + tokenizer
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)

    parser.add_argument("--first_model_path", type=str, default=None)
    parser.add_argument("--second_model_path", type=str, default=None)
    parser.add_argument("--first_tokenizer_path", type=str, default=None)
    parser.add_argument("--second_tokenizer_path", type=str, default=None)

    # PEFT
    parser.add_argument("--first_peft_path", nargs="+", default=None)
    parser.add_argument("--second_peft_path", nargs="+", default=None)

    # Memory / device config
    parser.add_argument("--device_map", type=str, default="balanced_low_0")
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--convert_to_half", action="store_true")

    args = parser.parse_args()
    main(args)
