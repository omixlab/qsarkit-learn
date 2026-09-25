# qsarkit-learn — documentation build.
#
# The sources live in docs-sphinx/. The built HTML is committed to docs/,
# which is the directory GitHub Pages serves, so rebuilding the site is part
# of preparing a commit rather than a step that happens on a server.
#
#   make docs      rebuild and refresh docs/ — the one to run before committing
#   make preview   build into docs-sphinx/build/html without touching docs/
#
# Every target forwards to docs-sphinx/Makefile, which holds the actual
# sphinx-build invocations; this file exists so the common case is one word
# from the repository root.

MAKEFLAGS += --no-print-directory
DOCS = docs-sphinx

# `docs` and `clean` name existing directories, so without .PHONY make would
# find docs/ already present, decide the target was up to date, and do nothing.
.PHONY: help docs preview doctest linkcheck strict clean

help:
	@echo "make docs       rebuild the site and copy it into docs/ (warnings are errors)"
	@echo "make preview    build into $(DOCS)/build/html, leaving docs/ alone"
	@echo "make doctest    execute every example in the documentation"
	@echo "make linkcheck  verify external links resolve"
	@echo "make strict     preview build with warnings as errors"
	@echo "make clean      remove build artefacts (docs/ is left in place)"

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
