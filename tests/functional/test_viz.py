"""Tests for pipeline flowchart rendering."""

from __future__ import annotations

import os

import pytest

from qsarkit.base import OptionalDependencyError
from qsarkit.functional import (
    desalt,
    drop_invalid,
    fingerprint,
    fit,
    molecules,
    pipeline_nodes,
    plot_pipeline,
    render_pipeline,
    scale,
    to_dot,
)

SMILES = ["CCO", "CCN", "c1ccccc1"]


@pytest.fixture
def pipe():
    return desalt() >> drop_invalid() >> fingerprint(n_bits=32) >> scale() >> fit("rf")


class TestPipelineNodes:
    def test_one_node_per_step_plus_input(self, pipe):
        nodes = pipeline_nodes(pipe)
        assert len(nodes) == 6            # input + five steps

    def test_input_node_is_optional(self, pipe):
        assert len(pipeline_nodes(pipe, include_input=False)) == 5

    def test_domains_track_the_molecule_to_feature_transition(self, pipe):
        assert [n.domain for n in pipeline_nodes(pipe)] == [
            "input", "molecules", "molecules", "transition", "features", "terminal",
        ]

    def test_a_single_step_is_a_valid_pipeline(self):
        assert [n.domain for n in pipeline_nodes(desalt())] == ["input", "molecules"]

    def test_a_molecule_set_draws_its_history(self):
        ms = molecules(SMILES) >> desalt() >> drop_invalid()
        # The `molecules(n=...)` entry is the input node, not a step.
        assert [n.label for n in pipeline_nodes(ms)][1:] == [
            "desalt()", "drop_invalid()",
        ]

    def test_a_feature_set_draws_its_history(self):
        fs = molecules(SMILES) >> desalt() >> fingerprint(n_bits=16)
        assert [n.domain for n in pipeline_nodes(fs)] == [
            "input", "molecules", "transition",
        ]

    def test_rejects_something_that_is_not_a_pipeline(self):
        with pytest.raises(TypeError, match="Expected a Step"):
            pipeline_nodes("not a pipeline")

    def test_repr_is_informative(self, pipe):
        assert "desalt()" in repr(pipeline_nodes(pipe)[1])


class TestDot:
    def test_produces_a_digraph(self, pipe):
        dot = to_dot(pipe)
        assert dot.startswith("digraph qsarkit_pipeline {")
        assert dot.rstrip().endswith("}")

    def test_one_edge_between_each_pair_of_nodes(self, pipe):
        assert to_dot(pipe).count("->") == len(pipeline_nodes(pipe)) - 1

    def test_labels_every_step(self, pipe):
        dot = to_dot(pipe)
        for label in ("desalt()", "drop_invalid()", "featurize(", "scale()", "fit("):
            assert label in dot

    def test_edges_name_what_flows_along_them(self, pipe):
        dot = to_dot(pipe)
        assert "MoleculeSet" in dot and "FeatureSet" in dot

    def test_rankdir_is_configurable(self, pipe):
        assert "rankdir=LR;" in to_dot(pipe, rankdir="LR")

    def test_rejects_bad_rankdir(self, pipe):
        with pytest.raises(ValueError, match="rankdir"):
            to_dot(pipe, rankdir="sideways")

    def test_quotes_in_labels_are_escaped(self):
        # A step configured with a string argument puts quotes in its label;
        # unescaped they would terminate the DOT string early.
        dot = to_dot(scale("robust"))
        assert '\\"robust\\"' in dot or "'robust'" in dot
        assert dot.count("digraph") == 1

    def test_method_on_the_step_matches_the_function(self, pipe):
        assert pipe.to_dot() == to_dot(pipe)


class TestPlotly:
    def test_returns_a_figure(self, pipe):
        assert type(plot_pipeline(pipe).__class__).__name__ == "type"
        assert plot_pipeline(pipe).__class__.__name__ == "Figure"

    def test_one_shape_per_node(self, pipe):
        figure = plot_pipeline(pipe)
        assert len(figure.layout.shapes) == len(pipeline_nodes(pipe))

    def test_horizontal_orientation(self, pipe):
        figure = plot_pipeline(pipe, orientation="horizontal")
        assert figure.layout.width > figure.layout.height

    def test_vertical_orientation_is_taller(self, pipe):
        figure = plot_pipeline(pipe, orientation="vertical")
        assert figure.layout.height > figure.layout.width

    def test_rejects_bad_orientation(self, pipe):
        with pytest.raises(ValueError, match="orientation"):
            plot_pipeline(pipe, orientation="diagonal")

    def test_does_not_write_files_or_show(self, pipe, tmp_path):
        before = set(os.listdir(tmp_path))
        plot_pipeline(pipe)
        assert set(os.listdir(tmp_path)) == before

    def test_title_is_configurable(self, pipe):
        assert plot_pipeline(pipe, title="My workflow").layout.title.text == "My workflow"

    def test_method_on_the_step_matches_the_function(self, pipe):
        assert len(pipe.plot().layout.shapes) == len(plot_pipeline(pipe).layout.shapes)


class TestRender:
    def test_rejects_unsupported_formats(self, pipe, tmp_path):
        with pytest.raises(ValueError, match="Unsupported output format"):
            render_pipeline(pipe, str(tmp_path / "flow.jpeg"))

    def test_rejects_unknown_engine(self, pipe, tmp_path):
        with pytest.raises(ValueError, match="engine must be"):
            render_pipeline(pipe, str(tmp_path / "flow.png"), engine="imagination")

    def test_writes_a_file_or_reports_the_missing_dependency(self, pipe, tmp_path):
        # Neither graphviz nor kaleido is a hard dependency, so both
        # outcomes are correct; what must not happen is a silent no-op.
        target = str(tmp_path / "flow.png")
        try:
            written = render_pipeline(pipe, target)
        except OptionalDependencyError as exc:
            assert exc.package in {"graphviz", "kaleido"}
        else:
            assert os.path.exists(written) and os.path.getsize(written) > 0

    def test_graphviz_engine_reports_its_own_absence(self, pipe, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "qsarkit.functional._viz._try_graphviz", lambda *a, **k: None
        )
        with pytest.raises(OptionalDependencyError, match="graphviz"):
            render_pipeline(pipe, str(tmp_path / "flow.png"), engine="graphviz")
