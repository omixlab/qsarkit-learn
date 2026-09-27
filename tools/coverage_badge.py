#!/usr/bin/env python3
"""Render a coverage badge from ``coverage.xml``.

Self-contained on purpose. A badge is three numbers and some rounded
rectangles, and generating it here avoids depending on an external service
that would need a token, could rate-limit, and would put the project's test
numbers on someone else's infrastructure.

Usage::

    python -m pytest --cov=qsarkit --cov-branch --cov-report=xml
    python tools/coverage_badge.py                      # writes the SVG
    python tools/coverage_badge.py --check              # exit 1 if stale

The output lives in ``.github/badges/`` rather than ``docs/`` because
``make docs`` rebuilds ``docs/`` from scratch and would delete it.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_XML = ROOT / "coverage.xml"
BADGE = ROOT / ".github" / "badges" / "coverage.svg"

#: Thresholds and colours, following the convention shields.io uses.
BANDS: Tuple[Tuple[float, str], ...] = (
    (95.0, "#4c1"),      # brightgreen
    (90.0, "#97ca00"),   # green
    (75.0, "#a4a61d"),   # yellowgreen
    (60.0, "#dfb317"),   # yellow
    (40.0, "#fe7d37"),   # orange
    (0.0, "#e05d44"),    # red
)


def read_coverage(path: Path = COVERAGE_XML) -> float:
    """The overall percentage from a Cobertura report.

    Branch coverage is included when the report carries it, weighted the way
    ``coverage report`` weights it: covered lines plus covered branches over
    the totals, rather than the mean of two percentages.
    """
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing. Generate it with:\n"
            "  python -m pytest --cov=qsarkit --cov-branch --cov-report=xml"
        )
    root = ET.parse(path).getroot()

    lines_valid = float(root.get("lines-valid") or 0)
    lines_covered = float(root.get("lines-covered") or 0)
    branches_valid = float(root.get("branches-valid") or 0)
    branches_covered = float(root.get("branches-covered") or 0)

    total = lines_valid + branches_valid
    if total == 0:
        # An empty report would otherwise render a confident 0%.
        raise SystemExit(f"{path} reports no measurable lines or branches.")
    return 100.0 * (lines_covered + branches_covered) / total


def colour_for(percentage: float) -> str:
    for threshold, colour in BANDS:
        if percentage >= threshold:
            return colour
    return BANDS[-1][1]  # pragma: no cover - the last band starts at 0


def render(percentage: float) -> str:
    """The badge SVG, in the flat style."""
    label = "coverage"
    value = f"{percentage:.0f}%"
    # 6.2 px per character is close enough to the metrics of the font stack
    # below; the badge is not pixel-matched to shields.io, only consistent.
    label_width = int(len(label) * 6.2) + 10
    value_width = int(len(value) * 6.8) + 10
    total = label_width + value_width
    colour = colour_for(percentage)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" \
xmlns:xlink="http://www.w3.org/1999/xlink" width="{total}" height="20" \
role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{colour}"/>
    <rect width="{total}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" \
font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width / 2:.0f}" y="14">{label}</text>
    <text x="{label_width + value_width / 2:.0f}" y="15" fill="#010101" \
fill-opacity=".3">{value}</text>
    <text x="{label_width + value_width / 2:.0f}" y="14">{value}</text>
  </g>
</svg>
"""


def _display(path: Path) -> str:
    """A repo-relative path when possible, absolute otherwise.

    ``relative_to`` raises for a path outside the project, which an explicit
    ``--output`` may well be.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed badge does not match the report",
    )
    parser.add_argument("--output", type=Path, default=BADGE)
    args = parser.parse_args(argv)

    # The module attribute, not the default argument: bound at call time so
    # the path can be redirected (the tests rely on this).
    percentage = read_coverage(COVERAGE_XML)
    svg = render(percentage)

    if args.check:
        current = args.output.read_text() if args.output.is_file() else ""
        if current != svg:
            print(
                f"coverage badge is stale: the report says {percentage:.1f}%. "
                f"Run `python tools/coverage_badge.py` and commit "
                f"{_display(args.output)}.",
                file=sys.stderr,
            )
            return 1
        print(f"coverage badge is current ({percentage:.1f}%)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"{percentage:.1f}% -> {_display(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
