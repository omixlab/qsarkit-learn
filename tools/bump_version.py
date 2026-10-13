#!/usr/bin/env python3
"""Bump the package version, in the one place it is defined.

``qsarkit/__init__.py`` holds the only copy of the version string.
``pyproject.toml`` reads it from there (``dynamic = ["version"]``) and
``docs-sphinx/source/conf.py`` imports it, so bumping this one line moves the
package metadata, the PyPI release and the documentation together. Keeping a
second copy in ``pyproject.toml`` is how a tag, a wheel and a rendered page
end up disagreeing about what version they are.

A bump also rebuilds ``docs/``, because that directory holds committed HTML
served by GitHub Pages: a bump without a rebuild publishes a site labelled with
the previous release, and nothing about the package itself looks wrong. Pass
``--no-docs`` to skip it when you intend to rebuild separately.

Usage::

    python tools/bump_version.py show
    python tools/bump_version.py patch | minor | major
    python tools/bump_version.py set 1.2.3
    python tools/bump_version.py patch --no-docs

Normally invoked through the Makefile: ``make bump-patch``, ``make bump-minor``,
``make bump-major``, ``make version``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "qsarkit" / "__init__.py"
CHANGELOG = ROOT / "docs-sphinx" / "source" / "changelog.rst"
# Prose that names the version it describes. Anything listed here is kept in
# step with the bump, because a document claiming to describe 0.3.0 while the
# package ships 0.4.0 is worse than one that does not say.
PROSE = ((ROOT / "package.md", re.compile(r"(Current at version )\d+\.\d+\.\d+")),)

VERSION_RE = re.compile(r'^(__version__\s*=\s*")(\d+)\.(\d+)\.(\d+)(")$', re.MULTILINE)
# A changelog section heading: "0.3.0 (unreleased)" or "0.3.0 (2026-09-25)",
# underlined with dashes. The marker decides whether the section is a draft we
# may relabel or a released one we must leave alone.
HEADING_RE = re.compile(
    r"^(?P<version>\d+\.\d+\.\d+) \((?P<marker>[^)]+)\)\n(?P<underline>-+)$",
    re.MULTILINE,
)
ANCHOR_RE = re.compile(r"^\.\. _changelog-[\d-]+:$", re.MULTILINE)

UNRELEASED = "unreleased"


def read_version() -> tuple[int, int, int]:
    match = VERSION_RE.search(INIT.read_text())
    if match is None:
        raise SystemExit(f"no `__version__ = \"X.Y.Z\"` line found in {INIT}")
    return int(match[2]), int(match[3]), int(match[4])


def format_version(parts: tuple[int, int, int]) -> str:
    return ".".join(str(p) for p in parts)


def bump(parts: tuple[int, int, int], part: str) -> tuple[int, int, int]:
    major, minor, patch = parts
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    if part == "patch":
        return major, minor, patch + 1
    raise SystemExit(f"unknown version part {part!r}")


def write_version(new: str) -> None:
    text = INIT.read_text()
    updated, count = VERSION_RE.subn(lambda m: f"{m[1]}{new}{m[5]}", text, count=1)
    if count != 1:
        raise SystemExit(f"failed to rewrite the version in {INIT}")
    INIT.write_text(updated)


def anchor_for(version: str) -> str:
    return f".. _changelog-{version.replace('.', '-')}:"


def sync_changelog(new: str) -> str:
    """Point the changelog's top section at the new version.

    A section still marked ``(unreleased)`` is a draft describing work that
    has not shipped, so it is relabelled — that draft is what the new version
    releases. A section carrying a date has shipped, so it is left untouched
    and an empty section for the new version is opened above it. Either way
    the prose is never rewritten.
    """
    if not CHANGELOG.exists():
        return "changelog not found, skipped"

    text = CHANGELOG.read_text()
    heading = HEADING_RE.search(text)
    if heading is None:
        return "no version heading found, skipped"

    old_version = heading["version"]
    if old_version == new:
        return f"top section is already {new}, left as is"

    if heading["marker"].strip().lower() == UNRELEASED:
        new_heading = f"{new} ({UNRELEASED})"
        replacement = f"{new_heading}\n{'-' * len(new_heading)}"
        text = text[: heading.start()] + replacement + text[heading.end() :]
        old_anchor = anchor_for(old_version)
        if old_anchor in text:
            text = text.replace(old_anchor, anchor_for(new), 1)
        CHANGELOG.write_text(text)
        return f"relabelled the unreleased section {old_version} -> {new}"

    anchor = ANCHOR_RE.search(text)
    insert_at = anchor.start() if anchor else heading.start()
    new_heading = f"{new} ({UNRELEASED})"
    section = f"{anchor_for(new)}\n\n{new_heading}\n{'-' * len(new_heading)}\n\n"
    text = text[:insert_at] + section + text[insert_at:]
    CHANGELOG.write_text(text)
    return f"opened a new {new} (unreleased) section above {old_version}"


def sync_prose(new: str) -> list[str]:
    """Update documents that state which version they describe."""
    messages = []
    for path, pattern in PROSE:
        if not path.exists():
            continue
        text = path.read_text()
        updated, count = pattern.subn(lambda m: f"{m[1]}{new}", text)
        if count and updated != text:
            path.write_text(updated)
            messages.append(f"{path.relative_to(ROOT)} now says {new}")
        elif not count:
            messages.append(f"{path.relative_to(ROOT)} has no version line, skipped")
    return messages


def rebuild_docs() -> int:
    """Rebuild the committed site in ``docs/`` so it names the new version.

    This runs here rather than only in the Makefile so that every route to a
    bump rebuilds -- calling this script directly used to leave `docs/` naming
    the previous release, and the two tests in
    ``tests/docs/test_published_site.py`` that catch it only report after a full
    suite run.
    """
    print("docs: rebuilding docs/ ...", flush=True)
    try:
        subprocess.run(["make", "docs"], cwd=ROOT, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        reason = (
            "`make` is not available"
            if isinstance(error, FileNotFoundError)
            else f"the build failed (exit {error.returncode})"
        )
        # The version is already written, so say so: re-running the bump would
        # move the version a second time.
        print(
            f"\ndocs: {reason}.\n"
            "docs: the version bump IS applied -- do not run the bump again.\n"
            "docs: rebuild with `make docs` (needs the `docs` extra:\n"
            '      pip install -e ".[docs]") and commit docs/ with it.',
            file=sys.stderr,
        )
        return 1
    print("docs: docs/ rebuilt")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["show", "patch", "minor", "major", "set"])
    parser.add_argument("value", nargs="?", help="the version to set, for `set`")
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="skip the docs/ rebuild; you are rebuilding it yourself",
    )
    args = parser.parse_args(argv)

    current = read_version()
    if args.action == "show":
        print(format_version(current))
        return 0

    if args.action == "set":
        if not args.value or not re.fullmatch(r"\d+\.\d+\.\d+", args.value):
            raise SystemExit("`set` needs a version of the form X.Y.Z")
        new = args.value
    else:
        new = format_version(bump(current, args.action))

    write_version(new)
    note = sync_changelog(new)
    print(f"version {format_version(current)} -> {new}")
    print(f"changelog: {note}")
    for message in sync_prose(new):
        print(f"prose: {message}")

    if args.no_docs:
        print("docs: skipped (--no-docs); docs/ still names the previous version")
        return 0
    return rebuild_docs()


if __name__ == "__main__":
    sys.exit(main())
