"""The coverage badge generator.

The badge is committed to the repository and rendered in the README, so a
wrong number is visible to everyone who visits the project. These tests pin
the parsing and the colour bands rather than the exact SVG geometry.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from coverage_badge import colour_for, main, read_coverage, render  # noqa: E402


def _report(tmp_path: Path, **attrs: str) -> Path:
    path = tmp_path / "coverage.xml"
    rendered = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    path.write_text(f'<?xml version="1.0"?><coverage {rendered}></coverage>')
    return path


class TestReadCoverage:
    def test_lines_and_branches_are_pooled(self, tmp_path):
        """Not the mean of two percentages: the weighted total, as coverage does."""
        path = _report(
            tmp_path,
            lines_valid="100",
            lines_covered="90",
            branches_valid="100",
            branches_covered="70",
        )
        assert read_coverage(path) == pytest.approx(80.0)

    def test_a_report_without_branches_still_works(self, tmp_path):
        path = _report(tmp_path, lines_valid="200", lines_covered="150")
        assert read_coverage(path) == pytest.approx(75.0)

    def test_a_missing_report_says_how_to_make_one(self, tmp_path):
        with pytest.raises(SystemExit, match="pytest --cov"):
            read_coverage(tmp_path / "absent.xml")

    def test_an_empty_report_is_refused_rather_than_shown_as_zero(self, tmp_path):
        path = _report(tmp_path, lines_valid="0", branches_valid="0")
        with pytest.raises(SystemExit, match="no measurable"):
            read_coverage(path)


class TestColourBands:
    @pytest.mark.parametrize(
        "percentage, colour",
        [
            (100.0, "#4c1"),
            (95.0, "#4c1"),
            (94.9, "#97ca00"),
            (90.0, "#97ca00"),
            (80.0, "#a4a61d"),
            (60.0, "#dfb317"),
            (40.0, "#fe7d37"),
            (0.0, "#e05d44"),
        ],
    )
    def test_each_band(self, percentage, colour):
        assert colour_for(percentage) == colour

    def test_the_colour_never_improves_as_coverage_falls(self):
        order = [colour_for(p) for p in (100, 92, 80, 65, 45, 10)]
        assert len(set(order)) == len(order)


class TestRender:
    def test_the_svg_parses_and_states_the_number(self):
        svg = render(87.4)
        ET.fromstring(svg)          # raises if malformed
        assert "87%" in svg
        assert "coverage" in svg

    def test_it_is_rounded_not_truncated(self):
        assert "88%" in render(87.6)

    def test_the_colour_reaches_the_svg(self):
        assert colour_for(97.0) in render(97.0)


class TestCheckMode:
    def test_check_fails_when_the_badge_is_stale(self, tmp_path, monkeypatch):
        import coverage_badge

        report = _report(
            tmp_path, lines_valid="100", lines_covered="90",
            branches_valid="0", branches_covered="0",
        )
        monkeypatch.setattr(coverage_badge, "COVERAGE_XML", report)
        badge = tmp_path / "coverage.svg"
        badge.write_text("<svg>stale</svg>")

        assert main(["--check", "--output", str(badge)]) == 1

    def test_check_passes_once_written(self, tmp_path, monkeypatch):
        import coverage_badge

        report = _report(
            tmp_path, lines_valid="100", lines_covered="90",
            branches_valid="0", branches_covered="0",
        )
        monkeypatch.setattr(coverage_badge, "COVERAGE_XML", report)
        badge = tmp_path / "coverage.svg"

        assert main(["--output", str(badge)]) == 0
        assert main(["--check", "--output", str(badge)]) == 0
