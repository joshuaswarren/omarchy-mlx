import sys

sys.path.insert(0, ".")
from pathlib import Path

import gdn_numcmp

D = Path("./pfres")
for pn in ("P1", "P2"):
    for x, y in (("A", "B"), ("B", "C"), ("C", "C2"), ("A", "C"), ("A", "C2")):
        r = gdn_numcmp.compare(D / f"num-{pn}-{x}", D / f"num-{pn}-{y}")
        print(
            pn, x, y,
            "ids_equal", r["ids_equal"],
            "full_steps", r["full_steps_compared"],
            "max|d| full", r["max_abs_dlogprob_full"],
            "elements differ", r["n_elements_differ_full"],
            "top1", r["top1_equal_steps"],
            "top5", r["top5_set_equal_steps"],
            "max|d| top20", r["max_abs_dlogprob_top20_all_steps"],
            "KL", r["max_kl_x_to_y"],
        )
