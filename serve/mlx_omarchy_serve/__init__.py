"""Serve-side package for mlx-omarchy: catalog, memory budget, CLI.

Shipped by install.sh into $MLX_OMARCHY_HOME (~/.local/share/mlx-omarchy) as a
plain python package; the launcher runs it with PYTHONPATH set. Stdlib only;
model loaders are imported lazily in the launch path.
"""

__version__ = 1
