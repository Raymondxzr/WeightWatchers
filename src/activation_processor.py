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


class BaseActivationProcessor:
    def __init__(self, batchsize=20, max_new_tokens=128) -> None:
        self.batchsize = batchsize
        self.max_new_tokens = max_new_tokens

    def read_activation_from_cache(self, cache, select_mask):
        """Read activation from cache by given select_mask."""
        cache.to('cpu')
        select_mask = select_mask.cpu()
        stack_cache = torch.stack([cache[key] for key in cache.keys()], dim=-2)  # ordered by layer (py>=3.7)
        del cache
        torch.cuda.empty_cache()
        stack_cache = stack_cache.float()
        size = stack_cache.size()
        if select_mask.shape[1] != size[1]:
            select_mask = torch.cat(
                [torch.zeros(select_mask.shape[0], size[1] - select_mask.shape[1]), select_mask],
                dim=1
            )
        cache_select = torch.masked_select(stack_cache, select_mask[..., None, None].bool())
        cache_select = cache_select.reshape(-1, size[-2], size[-1])
        return cache_select

    def process_prompts(self, model: HookedPreTrainedModel, prompts: list[str], token_type: str):
        """Tokenize/generate and build select masks for activations."""
        assert token_type in ['prompt', 'prompt_last', 'completion'], 'Unsupported token position'
        batch_input_ids, batch_attention_masks, batch_select_masks = [], [], []
        if 'prompt' in token_type:
            for i in tqdm(range(0, len(prompts), self.batchsize), desc="Processing prompts"):
                batch_prompts = prompts[i: i + self.batchsize]
                tokenized_prompts = model.to_tokens(batch_prompts, device=model.device)
                batch_input_ids.append(tokenized_prompts.input_ids)
                batch_attention_masks.append(tokenized_prompts.attention_mask)
                if 'last' in token_type:
                    select_mask = torch.zeros_like(tokenized_prompts.attention_mask)
                    select_mask[:, -1] = 1
                    batch_select_masks.append(select_mask)
                else:
                    batch_select_masks.append(tokenized_prompts.attention_mask)
        else:
            # completion tokens come from GENERATION of the SECOND model
            batch_input_ids, batch_attention_masks, batch_select_masks = generate_completions_and_masks(
                model,
                model.tokenizer,
                prompts,
                batch_size=self.batchsize,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
        return batch_input_ids, batch_attention_masks, batch_select_masks

    def _get_activation(
        self,
        model: HookedPreTrainedModel,
        batch_input_ids: torch.Tensor,
        batch_attention_masks: torch.Tensor,
        batch_select_masks: torch.Tensor,
        names_filter: Callable = lambda name: name.endswith('hook_post')
    ):
        """Get activations from processed inputs (for contrasting)."""
        device = model.device
        activation = []
        for input_ids, attention_mask, select_mask in tqdm(
            zip(batch_input_ids, batch_attention_masks, batch_select_masks),
            desc="Getting Activations",
            total=len(batch_input_ids)
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            select_mask = select_mask.to(device)
            _, cache = model.run_with_cache(input_ids=input_ids, attention_mask=attention_mask, names_filter=names_filter)
            batch_activation = self.read_activation_from_cache(cache, select_mask)
            activation.append(batch_activation)
        return torch.concat(activation, dim=0)

    def get_activation(self, model, prompts: list[str], names_filter, token_type: str = 'completion'):
        """Public entry: build inputs/masks then read activations."""
        batch_input_ids, batch_attention_masks, batch_select_masks = self.process_prompts(model, prompts, token_type)
        return self._get_activation(model, batch_input_ids, batch_attention_masks, batch_select_masks, names_filter)


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
        **load_parameter
    ) -> None:
        """
        If first/second_model_name_or_path are None, fall back to base_model_name_or_path
        with optional PEFTs (legacy behavior).
        """
        super().__init__(batchsize, max_new_tokens)
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
        names_filter: Callable = lambda name: name.endswith('hook_post'),
        token_type: str = 'completion'
    ):
        """
        Returns:
            change_scores, neuron_ranks, first_mean, first_std, second_mean, second_std
        """
        # SECOND model (generator)
        second_base = self.second_model_name_or_path or self.base_model_name_or_path
        second_tok = self.second_tokenizer_name_or_path or second_base

        hooked_model_2, tokenizer_2 = load_hooked_lm_and_tokenizer(
            model_name_or_path=second_base,
            tokenizer_name_or_path=second_tok,
            peft_name_or_path=self.second_peft_path,
            **self.load_parameter
        )
        hooked_model_2.set_tokenizer(tokenizer_2)

        # Build token stream using SECOND model (if token_type == 'completion')
        batch_input_ids, batch_attention_masks, batch_select_masks = self.process_prompts(
            hooked_model_2, prompts, token_type
        )
        second_activation = self._get_activation(
            hooked_model_2,
            batch_input_ids,
            batch_attention_masks,
            batch_select_masks,
            names_filter=names_filter
        )
        # hold tokenizer_2 for sanity checks
        del hooked_model_2
        torch.cuda.empty_cache()

        # FIRST model (comparator)
        first_base = self.first_model_name_or_path or self.base_model_name_or_path
        first_tok = self.first_tokenizer_name_or_path or first_base

        hooked_model_1, tokenizer_1 = load_hooked_lm_and_tokenizer(
            model_name_or_path=first_base,
            tokenizer_name_or_path=first_tok,
            peft_name_or_path=self.first_peft_path,
            **self.load_parameter
        )
        hooked_model_1.set_tokenizer(tokenizer_1)

        # Optional sanity: same hidden size / layers
        try:
            # attributes follow Hooked* implementations (Llama/Mistral/Gemma)
            n_layers_1 = len(hooked_model_1.model.layers)
            hidden_1 = hooked_model_1.model.config.hidden_size

            # Re-use tokenizer_2 with FIRST model isn't necessary, we only feed ids,
            # but enforce same tokenizer size to avoid mismatched IDs.
            # (If these mismatch, the activations won't be comparable.)
            assert tokenizer_1.vocab_size == tokenizer_2.vocab_size, \
                "Tokenizers differ (vocab_size mismatch). Use compatible checkpoints!"
        except Exception:
            pass  # keep it soft to avoid breaking custom forks

        first_activation = self._get_activation(
            hooked_model_1,
            batch_input_ids,
            batch_attention_masks,
            batch_select_masks,
            names_filter=names_filter
        )
        del hooked_model_1, tokenizer_1, tokenizer_2
        torch.cuda.empty_cache()

        print(f'Get neuron activation on {second_activation.shape[0]} tokens')
        change_scores = self.metric(first_activation, second_activation)
        first_mean = first_activation.mean(0)
        second_mean = second_activation.mean(0)
        first_std = first_activation.std(0)
        second_std = second_activation.std(0)
        neuron_ranks = torch.cat([torch.tensor((i, j)).unsqueeze(0) for i, j in topk_index(change_scores, -1)], dim=0)
        return change_scores, neuron_ranks, first_mean, first_std, second_mean, second_std

    def metric(self, first_activation: torch.Tensor, second_activation: torch.Tensor):
        """RMS distance across tokens: shape [layer, neuron]."""
        return (first_activation - second_activation).square().mean(0).sqrt()


class NeuronActivation(BaseActivationProcessor):
    def __init__(
        self,
        base_model_name_or_path: str,
        score_model_name_or_path: str,
        peft_path: list[str],
        neuron_ranks: list[torch.Tensor] = None,
        batchsize=20,
        max_new_tokens=128,
        **load_parameter
    ) -> None:
        super().__init__(batchsize, max_new_tokens)
        self.base_model_name_or_path = base_model_name_or_path
        self.score_model_name_or_path = score_model_name_or_path
        self.peft_path = peft_path
        self.load_parameter = load_parameter
        self.ranks = [defaultdict(list) for _ in neuron_ranks]
        for i, topk_index in enumerate(neuron_ranks):
            for layer, idx in topk_index:
                self.ranks[i][layer.item()].append(idx)

    def get_labels(self, prompts, model):
        cost_model, cost_tokenizer = load_hf_score_lm_and_tokenizer(
            model_name_or_path=self.score_model_name_or_path,
            tokenizer_name_or_path=self.score_model_name_or_path,
            **self.load_parameter
        )
        completed_prompts, cost_scores, *_ = generate_completions_and_scores(
            model,
            model.tokenizer,
            prompts,
            cost_model=cost_model,
            cost_tokenizer=cost_tokenizer,
            batch_size=self.batchsize,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        target = (torch.tensor(cost_scores) > 0).long()
        return cost_scores, target

    def create_dataset(self, prompts):
        hooked_model, tokenizer = load_hooked_lm_and_tokenizer(
            model_name_or_path=self.base_model_name_or_path,
            tokenizer_name_or_path=self.base_model_name_or_path,
            peft_name_or_path=self.peft_path,
            **self.load_parameter
        )
        hooked_model.set_tokenizer(tokenizer)
        names_filter = lambda name: name.endswith('hook_post')
        activation = self.get_activation(hooked_model, prompts, names_filter, token_type='prompt_last')
        activations = []
        for rank in self.ranks:
            layer_activations = []
            for layer, neurons in rank.items():
                layer_activations.append(activation[:, layer, neurons])
            activation_per_rank = torch.concat(layer_activations, -1)
            activations.append(activation_per_rank)
        cost_scores, targets = self.get_labels(prompts, hooked_model)
        return activations, cost_scores, targets
