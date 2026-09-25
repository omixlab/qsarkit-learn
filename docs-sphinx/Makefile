# Sphinx documentation for qsarkit.
#
# Two outputs, for two audiences:
#   `make html`     -> docs-sphinx/build/html, local scratch for previewing
#   `make publish`  -> build, then copy into ../docs, the committed Pages site
#
# The repository root has a Makefile that forwards here, so `make docs` from
# the top level and `make publish` from this directory do the same thing.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
HTMLDIR       = $(BUILDDIR)/html
PUBLISHDIR    = ../docs
# Keep the doctree cache out of the HTML output. Without -d Sphinx writes a
# ~22 MB .doctrees/ pickle cache into the build directory it is filling, which
# then gets copied into the published site, committed, and served to visitors.
DOCTREES      = $(BUILDDIR)/doctrees

.PHONY: help clean html publish linkcheck doctest livehtml strict

help:
	@echo "make html      preview build into $(HTMLDIR)"
	@echo "make publish   rebuild and copy into $(PUBLISHDIR) (warnings are errors)"
	@echo "make doctest   execute every example in the documentation"
	@echo "make linkcheck verify external links resolve"
	@echo "make strict    html build with warnings as errors"
	@echo "make clean     remove build artefacts"

clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/api/generated"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(HTMLDIR)" -d "$(DOCTREES)" $(SPHINXOPTS) $(O)

strict:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(HTMLDIR)" -d "$(DOCTREES)" -W --keep-going $(SPHINXOPTS) $(O)

# The published site.
#
# Built from scratch, for two reasons: a page deleted from the source must not
# linger in the output, and because the build directory is copied wholesale,
# anything stale left in it would be copied too. -W because a broken
# cross-reference must not reach the published docs.
#
# The copy happens only after a successful build, so a failure leaves the
# committed site untouched rather than half-replaced.
publish:
	rm -rf "$(HTMLDIR)"
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(HTMLDIR)" -d "$(DOCTREES)" -W --keep-going $(SPHINXOPTS) $(O)
	rm -rf "$(PUBLISHDIR)"
	mkdir -p "$(PUBLISHDIR)"
	cp -R "$(HTMLDIR)/." "$(PUBLISHDIR)/"
	@# sphinx.ext.githubpages writes this; re-touched so dropping that
	@# extension cannot silently break Pages (it skips directories with
	@# leading underscores, which is most of the output).
	@touch "$(PUBLISHDIR)/.nojekyll"
	@echo "Published to $(PUBLISHDIR). Review with git status, then commit."

doctest:
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" -d "$(DOCTREES)" $(SPHINXOPTS) $(O)

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" -d "$(DOCTREES)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(HTMLDIR)" -d "$(DOCTREES)" $(SPHINXOPTS) $(O)
