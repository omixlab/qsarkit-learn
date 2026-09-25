"""The committed ``docs/`` site must describe the version in the source tree.

``docs/`` holds the built HTML that GitHub Pages serves and is committed
deliberately, which means it can fall behind: bumping the version without
running ``make docs`` publishes a site labelled with the previous release,
and nothing about the package itself looks wrong. ``make bump-*`` rebuilds
the site for exactly this reason; this test is what notices when a bump
happened some other way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsarkit import __version__

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "docs"
INDEX = PUBLISHED / "index.html"


@pytest.mark.skipif(
    not INDEX.is_file(), reason="docs/ is not built in this checkout (e.g. an sdist)"
)
def test_published_site_states_the_current_version() -> None:
    """``conf.py`` renders ``html_title`` from ``qsarkit.__version__``."""
    text = INDEX.read_text(encoding="utf-8")
    expected = f"qsarkit {__version__}"

    assert expected in text, (
        f"docs/index.html does not mention {expected!r}, so the published site "
        f"was built from a different version than the source tree. "
        f"Run `make docs` and commit the result."
    )


@pytest.mark.skipif(
    not INDEX.is_file(), reason="docs/ is not built in this checkout (e.g. an sdist)"
)
def test_published_site_carries_no_stale_version() -> None:
    """A rebuild replaces the whole site, so no page may name another version.

    ``make docs`` builds from scratch into a clean directory, so a page left
    over from an earlier version would mean the copy step, not just one page,
    had gone wrong.
    """
    import re

    pattern = re.compile(r"qsarkit (\d+\.\d+\.\d+)")
    found = set()
    for page in PUBLISHED.rglob("*.html"):
        found.update(pattern.findall(page.read_text(encoding="utf-8", errors="ignore")))

    assert found <= {__version__}, (
        f"the published site names versions {sorted(found)} but the package is "
        f"{__version__}; run `make docs` to rebuild it from scratch."
    )
