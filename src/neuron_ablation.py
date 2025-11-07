# src/neuron_ablation.py
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Iterable, Tuple

import torch
from src.utils import get_act_name


# =========================
# Zeroing (existing method)
# =========================

def _zero_neurons_hook(value: torch.Tensor, hook, neurons: torch.Tensor):
    neurons = neurons.to(value.device)
    value[..., neurons] = 0
    return value


def register_zero_neurons(model, neuron_tuples: Iterable[Tuple[int, int]]):
    layers = defaultdict(list)
    for layer, idx in neuron_tuples:
        layers[int(layer)].append(int(idx))

    for layer, neuron_idx in layers.items():
        neurons_tensor = torch.tensor(neuron_idx)  # device set in hook
        hook_name = get_act_name("post", layer)
        model.add_perma_hook(
            name=hook_name,
            hook=partial(_zero_neurons_hook, neurons=neurons_tensor),
        )

    return model


# ==================================
# Dynamic activation patching method
# ==================================

@dataclass
class DynamicNeuronPatcher:
    """
    Shared patcher for donor (safe) and target (less-safe) models.

    layer_to_neurons: {layer_idx: 1D LongTensor of neuron indices}
    cache:           {layer_idx: [B, T, K] donor activations for that layer}
    """
    layer_to_neurons: Dict[int, torch.Tensor]
    cache: Dict[int, torch.Tensor] = field(default_factory=dict)

    def reset(self):
        """Clear cached donor activations (call once per decode step)."""
        self.cache.clear()

    # ------- hooks for donor model: cache activations -------

    def cache_hook(self, layer: int, value: torch.Tensor, hook):
        """
        Called on donor (aligned) model.

        value: [batch, seq, hidden]. We store only the selected neurons.
        """
        neurons = self.layer_to_neurons[layer].to(value.device)
        # detach + clone to avoid autograd and future in-place issues
        self.cache[layer] = value[..., neurons].detach().clone()
        return value

    # ------- hooks for target model: patch activations -------

    def patch_hook(self, layer: int, value: torch.Tensor, hook):
        """
        Called on target (less-safe) model.

        Replace selected neuron activations with cached donor activations.
        """
        if layer not in self.cache:
            return value  # nothing cached for this layer yet

        neurons = self.layer_to_neurons[layer].to(value.device)
        cached = self.cache[layer]

        # Match dtype/device
        cached = cached.to(device=value.device, dtype=value.dtype)

        # Defensive in case batch/seq shapes differ slightly
        B = min(value.shape[0], cached.shape[0])
        T = min(value.shape[1], cached.shape[1])
        if B == 0 or T == 0:
            return value

        value[:B, :T, neurons] = cached[:B, :T]
        return value


def register_dynamic_neuron_patching(
    target_model,
    donor_model,
    neuron_tuples: Iterable[Tuple[int, int]],
) -> DynamicNeuronPatcher:
    """
    Register paired hooks to implement dynamic activation patching
    as in SafetyNeuron.

    target_model: model being *patched* (e.g., Base/SFT, less safe).
    donor_model:  model providing aligned activations (e.g., TAR/DPO).
    neuron_tuples: iterable of (layer, neuron_idx) to patch.
    """
    layers = defaultdict(list)
    for layer, idx in neuron_tuples:
        layers[int(layer)].append(int(idx))

    layer_to_neurons = {
        layer: torch.tensor(idxs, dtype=torch.long)
        for layer, idxs in layers.items()
    }

    patcher = DynamicNeuronPatcher(layer_to_neurons=layer_to_neurons)

    for layer in layers.keys():
        hook_name = get_act_name("post", layer)

        # donor: cache activations only
        donor_model.add_perma_hook(
            name=hook_name,
            hook=partial(patcher.cache_hook, layer),
        )

        # target: patch activations from cache
        target_model.add_perma_hook(
            name=hook_name,
            hook=partial(patcher.patch_hook, layer),
        )

    return patcher


# ============================================
# Unified entrypoint
# ============================================

def register_neuron_intervention(
    method: str,
    target_model,
    neuron_tuples: Iterable[Tuple[int, int]],
    donor_model=None,
):
    method = method.lower()
    if method == "zero":
        return register_zero_neurons(target_model, neuron_tuples)

    if method in ("dynamic_patch", "dynamic", "patch"):
        if donor_model is None:
            raise ValueError("donor_model must be provided when method='dynamic_patch'.")
        return register_dynamic_neuron_patching(target_model, donor_model, neuron_tuples)

    raise ValueError(f"Unknown neuron intervention method: {method!r}")
