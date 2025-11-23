import os
import json
from typing import Callable
from collections import defaultdict

from tqdm import tqdm
import torch

from src.utils import topk_index
from src.eval.utils import (
    load_hooked_lm_and_tokenizer,
    load_hf_score_lm_and_tokenizer,
    generate_completions_and_masks,
    generate_completions_and_scores,
)
from src.models.HookedModelBase import HookedPreTrainedModel


# ============================================================
# Base Activation Processor
# ============================================================
class BaseActivationProcessor:
    def __init__(self, batchsize=20, max_new_tokens=128, max_seq_len=512) -> None:
        """
        max_seq_len: hard cap on sequence length for activations.
        We keep ONLY the last max_seq_len tokens to avoid OOM.
        """
        self.batchsize = batchsize
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len

    def _clip_tokens(self, tok):
        """
        Clip tokens to last `max_seq_len` positions (if set).
        This directly reduces activation memory in run_with_cache.
        """
        if self.max_seq_len is None:
            return tok
        seq_len = tok.input_ids.shape[1]
        if seq_len <= self.max_seq_len:
            return tok

        start = seq_len - self.max_seq_len
        tok.input_ids = tok.input_ids[:, start:]
        if getattr(tok, "attention_mask", None) is not None:
            tok.attention_mask = tok.attention_mask[:, start:]
        return tok

    def read_activation_from_cache(self, cache, select_mask):
        """Read activation from cache by given select_mask."""
        cache.to("cpu")
        select_mask = select_mask.cpu()

        stack_cache = torch.stack([cache[key] for key in cache.keys()], dim=-2)
        del cache
        torch.cuda.empty_cache()

        stack_cache = stack_cache.float()
        size = stack_cache.size()

        if select_mask.shape[1] != size[1]:
            pad = torch.zeros(select_mask.shape[0], size[1] - select_mask.shape[1])
            select_mask = torch.cat([pad, select_mask], dim=1)

        cache_select = torch.masked_select(
            stack_cache, select_mask[..., None, None].bool()
        )
        cache_select = cache_select.reshape(-1, size[-2], size[-1])
        return cache_select

    # =====================================================
    #       ★★ PATCHED COMPLETION CACHE LOGIC ★★
    # =====================================================
    def process_prompts(self, model: HookedPreTrainedModel, prompts: list[str], token_type: str):
        """
        Tokenize or load cached completions and build masks.
        token_type: 'prompt', 'prompt_last', or 'completion'.

        NEW: dynamically load cached completions if model._completion_cache_path exists
        """
        assert token_type in ["prompt", "prompt_last", "completion"]

        # -------------------------------------------------------
        # CASE 1: prompt / prompt_last tokens
        # -------------------------------------------------------
        if "prompt" in token_type:
            batch_input_ids, batch_attention_masks, batch_select_masks = [], [], []
            for i in tqdm(range(0, len(prompts), self.batchsize), desc="Processing prompts"):
                batch_prompts = prompts[i:i+self.batchsize]
                tok = model.to_tokens(batch_prompts, device=model.device)
                tok = self._clip_tokens(tok)

                batch_input_ids.append(tok.input_ids)
                batch_attention_masks.append(tok.attention_mask)

                if token_type == "prompt_last":
                    m = torch.zeros_like(tok.attention_mask)
                    m[:, -1] = 1
                    batch_select_masks.append(m)
                else:
                    batch_select_masks.append(tok.attention_mask)

            return batch_input_ids, batch_attention_masks, batch_select_masks

        # -------------------------------------------------------
        # CASE 2: COMPLETION TOKENS — using cached completions
        # -------------------------------------------------------
        cache_path = getattr(model, "_completion_cache_path", None)

        if cache_path and os.path.exists(cache_path):
            print(f"[Cached completions] Loading from {cache_path}")

            completions = []
            with open(cache_path) as f:
                for line in f:
                    obj = json.loads(line)
                    completions.append(obj["completion"].strip())

            total_cached = len(completions)
            num_prompts = len(prompts)
            print(f"[Cached completions] cached={total_cached}, prompts={num_prompts}")

            # Handle discrepancy between cached completions and the requested prompts
            if total_cached < num_prompts:
                raise ValueError(
                    f"Cached completions ({total_cached}) < prompts ({num_prompts}). "
                    "This usually means the cache and filtered dataset are misaligned."
                )
            elif total_cached > num_prompts:
                print(
                    f"[Cached completions] More cached completions than prompts; "
                    f"truncating to first {num_prompts} entries."
                )
                completions = completions[:num_prompts]

            batch_input_ids, batch_attention_masks, batch_select_masks = [], [], []

            for i in tqdm(
                range(0, len(completions), self.batchsize),
                desc="Tokenizing cached completions",
            ):
                batch = completions[i:i+self.batchsize]
                tok = model.to_tokens(batch, device=model.device)
                tok = self._clip_tokens(tok)

                batch_input_ids.append(tok.input_ids)
                batch_attention_masks.append(tok.attention_mask)
                batch_select_masks.append(tok.attention_mask)

            return batch_input_ids, batch_attention_masks, batch_select_masks

        # -------------------------------------------------------
        # FALLBACK: regenerate completions
        # -------------------------------------------------------
        print("[Warning] No cached completions found — generating completions on the fly!")
        return generate_completions_and_masks(
            model,
            model.tokenizer,
            prompts,
            batch_size=self.batchsize,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

    def _get_activation(
        self,
        model: HookedPreTrainedModel,
        batch_input_ids: torch.Tensor,
        batch_attention_masks: torch.Tensor,
        batch_select_masks: torch.Tensor,
        names_filter: Callable = lambda name: name.endswith("hook_post"),
    ):
        """
        Get activations with memory-safe micro-batching.
        """
        device = model.device
        activation = []

        micro_bs = 1

        for input_ids, attention_mask, select_mask in tqdm(
            zip(batch_input_ids, batch_attention_masks, batch_select_masks),
            desc="Getting Activations",
            total=len(batch_input_ids),
        ):
            B = input_ids.size(0)

            for start in range(0, B, micro_bs):
                end = min(start + micro_bs, B)
                ids_mb = input_ids[start:end].to(device)
                attn_mb = attention_mask[start:end].to(device)
                sel_mb = select_mask[start:end].to(device)

                _, cache = model.run_with_cache(
                    input_ids=ids_mb,
                    attention_mask=attn_mb,
                    names_filter=names_filter,
                )

                batch_activation = self.read_activation_from_cache(cache, sel_mb)
                activation.append(batch_activation)

        return torch.concat(activation, dim=0)

    def get_activation(self, model, prompts: list[str], names_filter, token_type: str = "completion"):
        batch_input_ids, batch_attention_masks, batch_select_masks = self.process_prompts(
            model, prompts, token_type
        )
        return self._get_activation(
            model, batch_input_ids, batch_attention_masks, batch_select_masks, names_filter
        )


# ============================================================
# Activation Contrasting
# ============================================================
class ActivationContrasting(BaseActivationProcessor):
    def __init__(
        self,
        base_model_name_or_path: str,
        first_peft_path: list[str],
        second_peft_path: list[str],
        first_model_name_or_path: str = None,
        second_model_name_or_path: str = None,
        first_tokenizer_name_or_path: str = None,
        second_tokenizer_name_or_path: str = None,
        batchsize=20,
        max_new_tokens=128,
        max_seq_len=512,
        **load_parameter,
    ) -> None:
        super().__init__(batchsize, max_new_tokens, max_seq_len=max_seq_len)
        self.base_model_name_or_path = base_model_name_or_path
        self.first_peft_path = first_peft_path
        self.second_peft_path = second_peft_path

        self.first_model_name_or_path = first_model_name_or_path
        self.second_model_name_or_path = second_model_name_or_path
        self.first_tokenizer_name_or_path = first_tokenizer_name_or_path
        self.second_tokenizer_name_or_path = second_tokenizer_name_or_path

        self.load_parameter = load_parameter

    def compute_change_scores(
        self,
        prompts: list[str],
        names_filter: Callable = lambda name: name.endswith("hook_post"),
        token_type: str = "completion",
    ):
        # SECOND (generator model)
        second_base = self.second_model_name_or_path or self.base_model_name_or_path
        second_tok = self.second_tokenizer_name_or_path or second_base

        hooked_model_2, tokenizer_2 = load_hooked_lm_and_tokenizer(
            model_name_or_path=second_base,
            tokenizer_name_or_path=second_tok,
            peft_name_or_path=self.second_peft_path,
            **self.load_parameter,
        )
        hooked_model_2.set_tokenizer(tokenizer_2)

        if hasattr(self, "_completion_cache_path"):
            hooked_model_2._completion_cache_path = self._completion_cache_path

        batch_input_ids, batch_attention_masks, batch_select_masks = self.process_prompts(
            hooked_model_2, prompts, token_type
        )

        second_activation = self._get_activation(
            hooked_model_2,
            batch_input_ids,
            batch_attention_masks,
            batch_select_masks,
            names_filter=names_filter,
        )

        del hooked_model_2
        torch.cuda.empty_cache()

        # FIRST (comparator model)
        first_base = self.first_model_name_or_path or self.base_model_name_or_path
        first_tok = self.first_tokenizer_name_or_path or first_base

        hooked_model_1, tokenizer_1 = load_hooked_lm_and_tokenizer(
            model_name_or_path=first_base,
            tokenizer_name_or_path=first_tok,
            peft_name_or_path=self.first_peft_path,
            **self.load_parameter,
        )
        hooked_model_1.set_tokenizer(tokenizer_1)

        if hasattr(self, "_completion_cache_path"):
            hooked_model_1._completion_cache_path = self._completion_cache_path

        first_activation = self._get_activation(
            hooked_model_1,
            batch_input_ids,
            batch_attention_masks,
            batch_select_masks,
            names_filter=names_filter,
        )

        del hooked_model_1, tokenizer_1
        torch.cuda.empty_cache()

        print(f"Get neuron activation on {second_activation.shape[0]} tokens")

        change_scores = self.metric(first_activation, second_activation)
        first_mean = first_activation.mean(0)
        second_mean = second_activation.mean(0)
        first_std = first_activation.std(0)
        second_std = second_activation.std(0)

        neuron_ranks = torch.cat(
            [torch.tensor((i, j)).unsqueeze(0) for i, j in topk_index(change_scores, -1)],
            dim=0,
        )

        return change_scores, neuron_ranks, first_mean, first_std, second_mean, second_std

    def metric(self, first_activation: torch.Tensor, second_activation: torch.Tensor):
        """RMS distance across tokens: shape [layer, neuron]."""
        return (first_activation - second_activation).square().mean(0).sqrt()


# ============================================================
# Neuron Activation Dataset Builder
# ============================================================
class NeuronActivation(BaseActivationProcessor):
    def __init__(
        self,
        base_model_name_or_path: str,
        score_model_name_or_path: str,
        peft_path: list[str],
        neuron_ranks: list[torch.Tensor] = None,
        batchsize=20,
        max_new_tokens=128,
        max_seq_len=None,
        **load_parameter,
    ) -> None:
        super().__init__(batchsize, max_new_tokens, max_seq_len=max_seq_len)
        self.base_model_name_or_path = base_model_name_or_path
        self.score_model_name_or_path = score_model_name_or_path
        self.peft_path = peft_path
        self.load_parameter = load_parameter

        self.ranks = [defaultdict(list) for _ in neuron_ranks]
        for i, topk_index_ in enumerate(neuron_ranks):
            for layer, idx in topk_index_:
                self.ranks[i][layer.item()].append(idx)

    def get_labels(self, prompts, model):
        cost_model, cost_tokenizer = load_hf_score_lm_and_tokenizer(
            model_name_or_path=self.score_model_name_or_path,
            tokenizer_name_or_path=self.score_model_name_or_path,
            **self.load_parameter,
        )
        completed_prompts, cost_scores, *_ = generate_completions_and_scores(
            model,
            model.tokenizer,
            prompts,
            cost_model=cost_model,
            cost_tokenizer=cost_tokenizer,
            batch_size=self.batchsize,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        target = (torch.tensor(cost_scores) > 0).long()
        return cost_scores, target

    def create_dataset(self, prompts):
        hooked_model, tokenizer = load_hooked_lm_and_tokenizer(
            model_name_or_path=self.base_model_name_or_path,
            tokenizer_name_or_path=self.base_model_name_or_path,
            peft_name_or_path=self.peft_path,
            **self.load_parameter,
        )
        hooked_model.set_tokenizer(tokenizer)

        if hasattr(self, "_completion_cache_path"):
            hooked_model._completion_cache_path = self._completion_cache_path

        names_filter = lambda name: name.endswith("hook_post")

        activation = self.get_activation(
            hooked_model, prompts, names_filter, token_type="prompt_last"
        )

        activations = []
        for rank in self.ranks:
            layer_acts = []
            for layer, neurons in rank.items():
                layer_acts.append(activation[:, layer, neurons])
            activation_per_rank = torch.concat(layer_acts, -1)
            activations.append(activation_per_rank)

        cost_scores, targets = self.get_labels(prompts, hooked_model)
        return activations, cost_scores, targets
