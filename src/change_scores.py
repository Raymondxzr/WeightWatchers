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
    """
    Parse a string like "0:74GiB,1:74GiB,2:74GiB,3:74GiB" into {0: "74GiB", 1: "74GiB", ...}
    """
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
    """
    Map common strings to torch dtypes. Falls back to 'auto'.
    """
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

    # Load dataset (expects JSON/JSONL with a "prompt" field)
    eval_data = datasets.load_dataset("json", data_files=args.dataset)["train"]["prompt"]
    if args.num_samples > 0:
        eval_data = eval_data[:args.num_samples]

    # Build chat-formatted prompts
    prompts = []
    for example in eval_data:
        prompt = example.strip()
        messages = [{"role": "user", "content": prompt}]
        prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(prompt + args.generation_startswith)

    # Names filter for activations (keeps MLP post activations)
    names_filter = lambda name: name.endswith("hook_post")

    # Loader kwargs propagated to src.utils.load_hooked_lm_and_tokenizer
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
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens in generation.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument(
        "--token_type",
        type=str,
        default="completion",
        choices=["prompt", "prompt_last", "completion"],
        help="Token positions to contrast.",
    )
    parser.add_argument(
        "--generation_startswith",
        type=str,
        default="",
        help="Optional prefix to force generations to start with.",
    )

    # Data I/O
    parser.add_argument("--dataset", type=str, default="", help="Path to JSON/JSONL dataset.")
    parser.add_argument("--output_file", type=str, default="../data/default.pt")

    # Default (legacy) base model/tokenizer for both sides
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Default base for both models if --first_model_path/--second_model_path are unset.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Default tokenizer dir if specific first/second tokenizer paths are unset.",
    )

    # NEW: direct model/tokenizer paths per side (optional)
    parser.add_argument("--first_model_path", type=str, default=None, help="HF path for FIRST model (comparator).")
    parser.add_argument("--second_model_path", type=str, default=None, help="HF path for SECOND model (generator).")
    parser.add_argument(
        "--first_tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path for FIRST model; defaults to --tokenizer_name_or_path or model dir.",
    )
    parser.add_argument(
        "--second_tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path for SECOND model; defaults to --tokenizer_name_or_path or model dir.",
    )

    # PEFT (optional; can pass multiple adapters)
    parser.add_argument(
        "--first_peft_path",
        nargs="+",
        default=None,
        help="PEFT adapters for FIRST model (e.g., IA3/LORA paths).",
    )
    parser.add_argument(
        "--second_peft_path",
        nargs="+",
        default=None,
        help="PEFT adapters for SECOND model (e.g., IA3/LORA paths).",
    )

    # Placement / memory knobs
    parser.add_argument(
        "--device_map",
        type=str,
        default="balanced_low_0",
        help='Model placement strategy, e.g. "balanced", "balanced_low_0", or "auto".',
    )
    parser.add_argument(
        "--max_memory",
        type=str,
        default=None,
        help='Logical GPU memory limits, e.g. "0:74GiB,1:74GiB,2:74GiB,3:74GiB".',
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help='Torch dtype: "auto", "bfloat16"/"bf16", "float16"/"fp16", or "float32"/"fp32".',
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization (if supported by your Hooked model).",
    )
    parser.add_argument(
        "--convert_to_half",
        action="store_true",
        help="Convert model to fp16 after loading (ignored if torch_dtype is not float32).",
    )

    args = parser.parse_args()
    main(args)
