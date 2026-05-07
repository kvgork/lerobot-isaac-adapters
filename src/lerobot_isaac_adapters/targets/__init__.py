"""
targets
=======

Per-backend training dispatchers for lerobot-isaac-adapters.

Each module exposes a single ``run(args: argparse.Namespace) -> None`` function.
All modules are **stubs** — they raise ``NotImplementedError`` until wired in a
later phase.

Modules
-------
- ``policy_lerobot``  — smolvla / act / diffusion via ``lerobot-train``
- ``wm_dreamerv3``    — DreamerV3 (sheeprl or nm-wu/dreamer-v3-pytorch)
- ``wm_leworldmodel`` — HF LeWorldModel
"""
