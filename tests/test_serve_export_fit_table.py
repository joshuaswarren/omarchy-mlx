"""The site fit-table export must agree with budget.py, never re-derive it."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget  # noqa: E402
from mlx_omarchy_serve.catalog import bundled_path  # noqa: E402
from mlx_omarchy_serve.export_fit_table import MEMORY_SIZES_GIB, build_fit_table  # noqa: E402


def test_export_cells_match_budget_calls():
    table = build_fit_table()
    assert table["memory_sizes_gib"] == MEMORY_SIZES_GIB
    cat_models = {m["id"]: m for m in json.loads(
        bundled_path().read_text(encoding="utf-8"))["models"]}
    for row in table["models"]:
        entry = cat_models[row["id"]]
        default_ctx = budget.resolve_context(entry["context"], None)
        for gib in MEMORY_SIZES_GIB:
            cell = row["fits"][str(gib)]
            available = gib * budget.GiB - budget.SAFETY_RESERVE_BYTES
            fits_at = budget.estimate_required(entry["memory"], cell["max_context"]).total
            assert fits_at <= available or cell["max_context"] == 0
            one_over = budget.estimate_required(entry["memory"], cell["max_context"] + 1).total
            assert one_over > available or cell["max_context"] == entry["context"]["max_tokens"]
            expected_fits = (
                cell["max_context"] >= default_ctx
                and (entry["capability"]["min_mem_gib"] is None
                     or entry["capability"]["min_mem_gib"] <= gib)
            )
            assert cell["fits"] is expected_fits, (row["id"], gib)


def test_export_is_json_serializable_and_sanitized():
    table = build_fit_table()
    text = json.dumps(table)
    for private in ("jwm1", "jw16", "jw14m2", "macstudio", "192.168.", "100."):
        assert private not in text
