"""
targets
=======

Per-backend training dispatchers for lerobot-isaac-adapters.

Each module exposes a single ``run(args: argparse.Namespace) -> int`` function
that returns a process exit code.  All three modules are wired with real
subprocess dispatch and metric parsing; ``--dry_run`` short-circuits before
spawning the heavy backend.

Modules
-------
- ``policy_lerobot``  — smolvla / act / diffusion via ``lerobot-train``
                        (parses ``eval/pc_success=``, re-emits as ``pc_success=``).
- ``wm_dreamerv3``    — DreamerV3 via ``sheeprl exp=dreamer_v3``.
                        Auto-converts Parquet → HDF5 (64×64) using the
                        ``lerobot_world_model_bridge`` skill.
                        Parses ``recon_loss=``.
- ``wm_leworldmodel`` — HF LeWorldModel via
                        ``python -m lerobot.scripts.train_world_model``.
                        Auto-converts Parquet → HDF5 (96×96, window=16).
                        Parses ``pred_loss=``.
"""
