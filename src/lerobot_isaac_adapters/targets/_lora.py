"""LoRA / PEFT integration for SmolVLA fine-tunes.

Two integration surfaces:

1. **Subprocess path (default):** translate CLI flags into the corresponding
   ``lerobot-train`` overrides under the ``policy.*`` namespace, appended to the
   subprocess command. This works for normal ``lerobot-train`` if and only if
   lerobot 0.5+ accepts our overrides on ``SmolVLAPolicy`` (it does not yet
   accept LoRA flags natively as of 0.5.x — see §3.2). Therefore we use:

2. **Monkey-patch path:** when ``--use_lora`` is set, the adapter dispatches
   through ``cli_train_cached``-style in-process wrapper which constructs the
   policy via ``make_policy()`` and immediately wraps ``policy.model`` with
   ``peft.get_peft_model(...)`` before the lerobot trainer captures the
   parameter list. This is the path actually executed.

Soft-import contract: peft and SmolVLAPolicy are NOT imported at module
level so the argparse layer and dry-runs still work without peft installed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

# Named presets — see plan §2.3.
TARGET_MODULE_PRESETS: dict[str, list[str]] = {
    "attn_qv":     ["q_proj", "v_proj"],
    "attn_qkvo":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "expert_only": ["q_proj", "v_proj"],   # combined with submodule filter below
}

# Regex used for expert_only preset to restrict LoRA to lm_expert submodule.
_EXPERT_ONLY_REGEX = r"lm_expert.*\.(q_proj|v_proj)$"


@dataclass(frozen=True)
class LoraSpec:
    """Frozen LoRA configuration parsed from CLI flags."""

    rank: int
    alpha: int
    dropout: float
    target_modules: List[str]
    expert_only: bool  # True iff preset == "expert_only"

    @classmethod
    def from_args(
        cls,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules_spec: str,
    ) -> "LoraSpec":
        """Build a LoraSpec from raw CLI strings.

        Parameters
        ----------
        rank:
            LoRA rank ``r``. Common range 4–32.
        alpha:
            LoRA scaling factor. Effective scale = ``alpha / r``.
        dropout:
            Dropout on the LoRA path. 0.0 disables.
        target_modules_spec:
            Either a preset name from ``TARGET_MODULE_PRESETS``, OR a
            comma-separated list of layer-name suffixes
            (e.g. ``"q_proj,v_proj,gate_proj"``).
        """
        spec_stripped = target_modules_spec.strip()

        if spec_stripped in TARGET_MODULE_PRESETS:
            modules = TARGET_MODULE_PRESETS[spec_stripped]
            expert_only = spec_stripped == "expert_only"
        else:
            # Comma-separated raw layer suffixes — split and strip whitespace.
            modules = [s.strip() for s in spec_stripped.split(",") if s.strip()]
            expert_only = False

        return cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=modules,
            expert_only=expert_only,
        )


def build_peft_config(spec: LoraSpec):
    """Construct a ``peft.LoraConfig``. Soft-imports peft on first call.

    Raises
    ------
    ImportError
        With an install hint if ``peft >= 0.10`` is not installed.
    """
    try:
        from peft import LoraConfig  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install it with: pip install 'peft>=0.10'"
        ) from exc

    if spec.expert_only:
        # Restrict to lm_expert submodule via regex target_modules.
        target_modules = _EXPERT_ONLY_REGEX
    else:
        target_modules = spec.target_modules

    return LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        target_modules=target_modules,
        lora_dropout=spec.dropout,
        bias="none",
        use_rslora=False,
        use_dora=False,
        init_lora_weights=True,
    )


def wrap_smolvla_policy(policy, spec: LoraSpec):
    """Wrap an instantiated SmolVLAPolicy with PEFT LoRA.

    Mutation::

        policy.model = peft.get_peft_model(policy.model, lora_config)

    Returns the same policy object (for chaining). After this call,
    ``policy.parameters()`` returns ONLY the LoRA A/B matrices as trainable
    (plus any submodule the user un-froze elsewhere).

    Parameters
    ----------
    policy:
        An instantiated ``SmolVLAPolicy`` (or any policy whose ``.model``
        attribute is the ``nn.Module`` to wrap).
    spec:
        Parsed ``LoraSpec`` from ``LoraSpec.from_args()``.

    Returns
    -------
    policy
        The same object, mutated in-place.
    """
    try:
        from peft import get_peft_model  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install it with: pip install 'peft>=0.10'"
        ) from exc

    lora_config = build_peft_config(spec)

    base_model = getattr(policy, "model", None)
    if base_model is None:
        warnings.warn(
            "[_lora] policy has no .model attribute — skipping LoRA wrap. "
            "Check that this is a SmolVLAPolicy.",
            stacklevel=2,
        )
        return policy

    policy.model = get_peft_model(base_model, lora_config)

    # Log trainable parameter count to stdout for dry-run / debug visibility.
    try:
        policy.model.print_trainable_parameters()
    except Exception:  # noqa: BLE001
        pass

    return policy
