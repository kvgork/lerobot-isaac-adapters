"""Sheeprl plugin for lerobot-isaac.

Provides a custom HDF5-replay gymnasium env that lets sheeprl's `dreamer_v3`
(and any other sheeprl algorithm) consume the world-model HDF5 produced by
`lerobot_world_model_bridge` directly — no need to write your own env from
scratch.

Usage
-----
After installing the adapters package, point sheeprl at the bundled config
dir and select ``env=custom_hdf5``::

    python -m sheeprl \\
        --config-dir=$(python -c 'import lerobot_isaac_adapters.sheeprl_plugin as p, os; print(os.path.join(os.path.dirname(p.__file__), "configs"))') \\
        exp=dreamer_v3 \\
        env=custom_hdf5 \\
        env.dataset_path=outputs/.../so101_dreamerv3_full.hdf5

The `wm_dreamerv3.py` adapter target sets this up automatically.
"""

from .hdf5_env import HDF5ReplayEnv, get_hdf5_env

__all__ = ["HDF5ReplayEnv", "get_hdf5_env"]
