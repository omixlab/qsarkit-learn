# qsarkit-learn — documentation and release.
#
# The release flow, in the order you run it:
#
#   make bump-patch     bump the version and rebuild docs/  (before committing)
#   git commit -am "…"  commit the bump, the docs and your changes
#   make release        tag the commit and push the tag -> publishes to PyPI
#
# The version lives in exactly one place, qsarkit/__init__.py. pyproject.toml
# reads it from there and so does the documentation, so a bump moves all three
# together.

MAKEFLAGS += --no-print-directory
DOCS   = docs-sphinx
BUMP   = python tools/bump_version.py

.PHONY: help version bump-patch bump-minor bump-major tag release \
        docs preview doctest linkcheck strict clean

help:
	@echo "Documentation"
	@echo "  make docs       rebuild the site and copy it into docs/ (warnings are errors)"
	@echo "  make preview    build into $(DOCS)/build/html, leaving docs/ alone"
	@echo "  make doctest    execute every example in the documentation"
	@echo "  make linkcheck  verify external links resolve"
	@echo "  make strict     preview build with warnings as errors"
	@echo "  make clean      remove build artefacts (docs/ is left in place)"
	@echo
	@echo "Release"
	@echo "  make version    print the current version"
	@echo "  make bump-patch bump X.Y.Z -> X.Y.Z+1, then rebuild docs/"
	@echo "  make bump-minor bump X.Y.Z -> X.Y+1.0, then rebuild docs/"
	@echo "  make bump-major bump X.Y.Z -> X+1.0.0, then rebuild docs/"
	@echo "  make tag        tag the current commit v<version> (local only)"
	@echo "  make release    tag and push -> GitHub Actions publishes to PyPI"

# ------------------------------------------------------------------ release --

version:
	@$(BUMP) show

# One rule for all three; $(@:bump-%=%) turns the target name into the
# argument. The new version is read back inside the shell rather than through a
# make variable, because make expands the whole recipe before running any of
# it, so a $(VERSION) here would still hold the pre-bump number.
bump-patch bump-minor bump-major:
	@$(BUMP) $(@:bump-%=%)
	@$(MAKE) docs
	@v=$$($(BUMP) show); \
	echo; \
	echo "Bumped to $$v and rebuilt docs/. Next:"; \
	echo "  git add -A && git commit -m \"release $$v\""; \
	echo "  make release"

# Creating the tag is local and reversible; pushing it is neither, because the
# push starts a PyPI upload that cannot be replaced or undone. They are
# separate targets so `make tag` can never publish by accident.
tag:
	@v=$$($(BUMP) show); \
	if [ -n "$$(git status --porcelain)" ]; then \
		echo "The working tree has uncommitted changes."; \
		echo "Commit the version bump and the rebuilt docs/ first, so the tag"; \
		echo "points at exactly what gets published."; \
		exit 1; \
	fi; \
	if git rev-parse -q --verify "refs/tags/v$$v" >/dev/null; then \
		echo "Tag v$$v already exists. Bump the version before releasing again"; \
		echo "-- PyPI refuses a second upload of a version it already has."; \
		exit 1; \
	fi; \
	git tag -a "v$$v" -m "qsarkit-learn $$v"; \
	echo "Created tag v$$v. Push it to publish:"; \
	echo "  git push --follow-tags        (or: make release)"

release: tag
	@v=$$($(BUMP) show); \
	git push --follow-tags; \
	echo; \
	echo "Pushed v$$v. GitHub Actions is now building and publishing to PyPI:"; \
	echo "  https://github.com/omixlab/qsarkit-learn/actions"

# ------------------------------------------------------------ documentation --

docs:
	@$(MAKE) -C $(DOCS) publish

preview:
	@$(MAKE) -C $(DOCS) html
	@echo "Open $(DOCS)/build/html/index.html"

doctest:
	@$(MAKE) -C $(DOCS) doctest

linkcheck:
	@$(MAKE) -C $(DOCS) linkcheck

strict:
	@$(MAKE) -C $(DOCS) strict

clean:
	@$(MAKE) -C $(DOCS) clean
