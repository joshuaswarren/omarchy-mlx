"""Hardware claims in docs/compatibility.md need dated receipt links.

Issue #148: the 2026-09-20 status block said the M2 Max ANE never ran
inference on Linux while the README and its receipts said it had. A
negative hardware claim ("NOT live-inference-qualified", "deferred")
must never stand in the Hardware section without a dated receipt link
on the same bullet.
"""

import re
from pathlib import Path

BANNED = ("not live-inference-qualified", "deferred")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _hardware_section() -> str:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "compatibility.md").read_text()
    m = re.search(r"^## Hardware\n(.*?)(?=^## )", text, re.M | re.S)
    assert m, "no Hardware section in docs/compatibility.md"
    return m.group(1)


def _bullets(section: str) -> list[str]:
    out: list[str] = []
    for block in re.split(r"\n\s*\n", section):
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("- ") or not out:
                out.append(line)
            else:
                out[-1] += " " + line
    return out


def _has_dated_receipt_link(bullet: str) -> bool:
    return any(
        DATE.search(label) or DATE.search(url)
        for label, url in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", bullet)
    )


def test_negative_claims_carry_a_dated_receipt_link() -> None:
    bad = []
    for bullet in _bullets(_hardware_section()):
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z*\[])", bullet):
            low = sentence.lower()
            if any(p in low for p in BANNED) and not _has_dated_receipt_link(sentence):
                bad.append(sentence[:160])
    assert not bad, (
        "Hardware bullets state a negative with no dated receipt link "
        "on the same bullet:\n" + "\n".join(bad)
    )


def test_m2_max_ane_result_is_stated_with_receipt_and_version() -> None:
    section = _hardware_section()
    assert "receipts/2026-10-10-m2-max-ane.md" in section
    assert "omarchy-ane v0.4.8" in section
    assert "bit-exact" in section


def test_m3_m4_rows_match_readme() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    section = _hardware_section()
    for phrase in (
        "aurora mesa-m3 graphics; compute not certified",
        "no Linux GPU driver",
        "h15",
        "h16",
    ):
        assert phrase in readme, f"README lost the phrase: {phrase}"
        assert phrase in section, f"compatibility.md Hardware lost: {phrase}"
