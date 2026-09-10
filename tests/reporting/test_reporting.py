from __future__ import annotations

import json

import numpy as np
import pytest

from qsarkit.base import OptionalDependencyError
from qsarkit.reporting import (
    OECDReportBuilder,
    QSARReport,
    ReportSection,
    figure_to_html,
    plot_feature_importance,
    plot_learning_curve,
    plot_predicted_vs_observed,
    plot_residuals,
    plot_roc_curve,
    plot_williams,
)


@pytest.fixture(scope="module")
def predictions():
    rng = np.random.RandomState(0)
    y_true = rng.normal(size=60) * 2 + 5
    y_pred = y_true + rng.normal(scale=0.4, size=60)
    return y_true, y_pred


class TestPlots:
    def test_predicted_vs_observed_has_points_and_identity_line(self, predictions):
        import plotly.graph_objects as go

        fig = plot_predicted_vs_observed(*predictions)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_predicted_vs_observed_accepts_labels(self, predictions):
        y_true, y_pred = predictions
        fig = plot_predicted_vs_observed(
            y_true, y_pred, labels=[f"cpd_{i}" for i in range(len(y_true))]
        )
        assert fig.data[1].text is not None

    def test_residuals_are_centred_on_zero(self, predictions):
        y_true, y_pred = predictions
        fig = plot_residuals(y_true, y_pred, standardized=True)
        assert abs(float(np.mean(fig.data[0].y))) < 0.5

    def test_standardized_residuals_have_unit_spread(self, predictions):
        y_true, y_pred = predictions
        fig = plot_residuals(y_true, y_pred, standardized=True)
        assert float(np.std(fig.data[0].y, ddof=1)) == pytest.approx(1.0, abs=1e-6)

    def test_raw_residuals_keep_their_units(self, predictions):
        y_true, y_pred = predictions
        fig = plot_residuals(y_true, y_pred, standardized=False)
        assert np.allclose(fig.data[0].y, y_true - y_pred)

    def test_williams_plot_has_threshold_lines(self, predictions):
        y_true, y_pred = predictions
        leverage = np.random.RandomState(1).uniform(0.01, 0.3, len(y_true))
        fig = plot_williams(leverage, y_true, y_pred)
        # one vertical h* line plus two horizontal residual limits
        assert len(fig.layout.shapes) >= 3

    def test_williams_plot_accepts_an_explicit_h_star(self, predictions):
        y_true, y_pred = predictions
        leverage = np.full(len(y_true), 0.1)
        fig = plot_williams(leverage, y_true, y_pred, h_star=0.25)
        assert any(
            getattr(s, "x0", None) == 0.25 for s in fig.layout.shapes
        )

    def test_williams_rejects_mismatched_leverage(self, predictions):
        y_true, y_pred = predictions
        with pytest.raises(ValueError, match="leverage has shape"):
            plot_williams(np.zeros(5), y_true, y_pred)

    def test_roc_curve_reports_auc(self, predictions):
        y_true, y_pred = predictions
        fig = plot_roc_curve((y_true > 5).astype(int), y_pred)
        assert any("AUC" in (trace.name or "") for trace in fig.data)

    def test_roc_auc_is_high_for_a_good_score(self, predictions):
        y_true, y_pred = predictions
        fig = plot_roc_curve((y_true > 5).astype(int), y_pred)
        auc_text = [t.name for t in fig.data if "AUC" in (t.name or "")][0]
        assert float(auc_text.split("=")[1].rstrip(")")) > 0.9

    def test_roc_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="but y_score has"):
            plot_roc_curve([0, 1, 0], [0.1, 0.9])

    def test_learning_curve_plots_both_series(self):
        fig = plot_learning_curve(
            [10, 20, 40],
            [[0.9, 0.91], [0.93, 0.94], [0.95, 0.96]],
            [[0.7, 0.72], [0.8, 0.81], [0.85, 0.86]],
        )
        assert len(fig.data) == 2
        assert {t.name for t in fig.data} == {"training", "validation"}

    def test_learning_curve_accepts_flat_scores(self):
        fig = plot_learning_curve([10, 20], [0.9, 0.95], [0.7, 0.8])
        assert len(fig.data) == 2

    def test_learning_curve_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="train_sizes has"):
            plot_learning_curve([10, 20, 30], [0.9, 0.95], [0.7, 0.8])

    def test_feature_importance_shows_the_top_n(self):
        fig = plot_feature_importance(
            [f"d{i}" for i in range(30)], np.arange(30.0), top_n=5
        )
        assert len(fig.data[0].y) == 5

    def test_feature_importance_is_sorted(self):
        values = np.array([0.1, 0.9, 0.5])
        fig = plot_feature_importance(["a", "b", "c"], values)
        # the bar chart is drawn bottom-up, so the largest is last
        assert list(fig.data[0].y)[-1] == "b"

    def test_feature_importance_accepts_error_bars(self):
        fig = plot_feature_importance(
            ["a", "b"], [0.1, 0.9], errors=[0.01, 0.02]
        )
        assert fig.data[0].error_x is not None

    def test_feature_importance_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="names has length"):
            plot_feature_importance(["a", "b"], [0.1, 0.2, 0.3])

    def test_mismatched_prediction_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="but y_pred has"):
            plot_predicted_vs_observed([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_figure_to_html(self, predictions):
        fragment = figure_to_html(plot_predicted_vs_observed(*predictions))
        assert "<div" in fragment
        assert "plotly" in fragment.lower()


class TestQSARReport:
    def test_markdown_includes_the_title_and_endpoint(self):
        report = QSARReport(title="EGFR model", endpoint="pIC50")
        markdown = report.to_markdown()
        assert "# EGFR model" in markdown
        assert "pIC50" in markdown

    def test_sections_keep_their_order(self):
        report = QSARReport()
        report.add_section("First", {"a": 1})
        report.add_section("Second", {"b": 2})
        markdown = report.to_markdown()
        assert markdown.index("First") < markdown.index("Second")

    def test_add_section_is_chainable(self):
        report = QSARReport().add_section("A").add_section("B")
        assert len(report.sections) == 2

    def test_content_is_rendered_as_a_table(self):
        report = QSARReport().add_section("Dataset", {"n_compounds": 1200})
        markdown = report.to_markdown()
        assert "| n_compounds | 1200 |" in markdown

    def test_free_text_is_included(self):
        report = QSARReport().add_section("Notes", text="Curated from ChEMBL.")
        assert "Curated from ChEMBL." in report.to_markdown()

    def test_dataset_section_records_split_sizes(self):
        report = QSARReport().add_dataset_section(
            n_compounds=1000, n_train=800, n_test=200, source="ChEMBL 34"
        )
        content = report.sections[0].content
        assert content["n_train"] == 800 and content["source"] == "ChEMBL 34"

    def test_dataset_section_includes_a_curation_log(self):
        from qsarkit.data_quality import CurationReport

        curation = CurationReport(n_input=100, n_output=80)
        curation.stages = [{"stage": "validate", "n_before": 100, "n_after": 80}]
        report = QSARReport().add_dataset_section(
            n_compounds=80, curation=curation
        )
        content = report.sections[0].content
        assert content["n_removed"] == 20
        assert report.sections[0].table is not None

    def test_model_section_records_the_algorithm_and_parameters(self):
        from sklearn.ensemble import RandomForestRegressor

        report = QSARReport().add_model_section(
            RandomForestRegressor(n_estimators=42), descriptors="ECFP4"
        )
        content = report.sections[0].content
        assert content["algorithm"] == "RandomForestRegressor"
        assert content["descriptors"] == "ECFP4"
        assert content["param.n_estimators"] == 42

    def test_model_section_accepts_explicit_hyperparameters(self):
        report = QSARReport().add_model_section(
            object(), hyperparameters={"depth": 5}
        )
        assert report.sections[0].content["param.depth"] == 5

    def test_validation_section_includes_scrambling(self):
        report = QSARReport().add_validation_section(
            {"q2": 0.8}, y_scrambling={"p_value": 0.001}
        )
        content = report.sections[0].content
        assert content["q2"] == 0.8
        assert content["scrambling.p_value"] == 0.001

    def test_applicability_section_records_the_method(self):
        from qsarkit.applicability import LeverageAD

        domain = LeverageAD().fit(np.random.RandomState(0).normal(size=(40, 4)))
        report = QSARReport().add_applicability_section(domain, coverage=0.93)
        content = report.sections[0].content
        assert content["method"] == "LeverageAD"
        assert content["coverage"] == 0.93
        assert "threshold" in content

    def test_json_is_valid_and_round_trips(self):
        report = QSARReport(title="T", endpoint="E").add_section("S", {"k": 1})
        payload = json.loads(report.to_json())
        assert payload["title"] == "T"
        assert payload["sections"][0]["content"]["k"] == 1

    def test_json_handles_numpy_types(self):
        report = QSARReport().add_section(
            "S", {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1, 2])}
        )
        payload = json.loads(report.to_json())
        assert payload["sections"][0]["content"]["a"] == 1.5
        assert payload["sections"][0]["content"]["c"] == [1, 2]

    def test_json_converts_non_finite_to_null(self):
        report = QSARReport().add_section("S", {"a": np.float64("nan")})
        payload = json.loads(report.to_json())
        assert payload["sections"][0]["content"]["a"] is None

    def test_html_is_a_complete_document(self):
        report = QSARReport(title="T").add_section("S", {"k": 1})
        html = report.to_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<h2>S</h2>" in html

    def test_html_escapes_content(self):
        report = QSARReport(title="<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in report.to_html()
        assert "&lt;script&gt;" in report.to_html()

    def test_html_embeds_figures(self, predictions):
        report = QSARReport().add_validation_section(
            {"q2": 0.8}, figures=[plot_predicted_vs_observed(*predictions)]
        )
        assert "plotly" in report.to_html().lower()

    def test_markdown_without_figures_still_renders(self, predictions):
        report = QSARReport().add_validation_section(
            {"q2": 0.8}, figures=[plot_predicted_vs_observed(*predictions)]
        )
        markdown = report.to_markdown(include_figures=False)
        assert "q2" in markdown
        # The figure is named rather than dropped without trace.
        assert "[figure:" in markdown

    @pytest.mark.parametrize("fmt", ["markdown", "json", "html"])
    def test_writes_to_a_file(self, fmt, tmp_path):
        report = QSARReport(title="T").add_section("S", {"k": 1})
        path = tmp_path / f"report.{fmt}"
        getattr(report, f"to_{fmt}")(path)
        assert path.exists() and path.stat().st_size > 0

    def test_float_formatting(self):
        report = QSARReport().add_section("S", {"x": 0.123456789})
        assert "0.1235" in report.to_markdown()

    def test_non_finite_renders_as_not_available(self):
        report = QSARReport().add_section("S", {"x": float("nan")})
        assert "n/a" in report.to_markdown()

    def test_long_lists_are_truncated(self):
        report = QSARReport().add_section("S", {"x": list(range(20))})
        assert "20 items" in report.to_markdown()

    def test_repr(self):
        report = QSARReport(title="T").add_section("S")
        assert "QSARReport" in repr(report) and "1 sections" in repr(report)

    def test_section_to_dict_excludes_figures(self, predictions):
        section = ReportSection(
            title="S", content={"k": 1},
            figures=[plot_predicted_vs_observed(*predictions)],
        )
        assert "figures" not in section.to_dict()


class TestOECDReportBuilder:
    def test_lists_all_five_principles(self):
        markdown = OECDReportBuilder(title="M").build().to_markdown()
        for number in range(1, 6):
            assert f"Principle {number}" in markdown

    def test_unaddressed_principles_are_flagged(self):
        builder = OECDReportBuilder(title="M")
        builder.add_evidence(1, True, {"endpoint": "pIC50"})
        assert builder.unaddressed == [2, 3, 4, 5]
        assert "NOT ADDRESSED" in builder.build().to_markdown()

    def test_a_fully_documented_model_has_nothing_unaddressed(self):
        builder = OECDReportBuilder(title="M")
        for number in range(1, 6):
            builder.add_evidence(number, True, {"note": "done"})
        assert builder.unaddressed == []
        assert "NOT ADDRESSED" not in builder.build().to_markdown()

    def test_evidence_details_reach_the_report(self):
        builder = OECDReportBuilder(title="M")
        builder.add_evidence(2, True, {"algorithm": "RandomForest"})
        markdown = builder.build().to_markdown()
        assert "RandomForest" in markdown

    def test_add_evidence_is_chainable(self):
        builder = (
            OECDReportBuilder()
            .add_evidence(1, True, {})
            .add_evidence(2, True, {})
        )
        assert builder.unaddressed == [3, 4, 5]

    def test_rejects_an_out_of_range_principle(self):
        with pytest.raises(ValueError, match="principle must be 1-5"):
            OECDReportBuilder().add_evidence(9, True, {})

    def test_from_validation_populates_principle_four(self):
        builder = OECDReportBuilder().from_validation({"q2": 0.79, "rmse": 0.5})
        assert 4 not in builder.unaddressed

    def test_evidence_handles_numpy_values(self):
        builder = OECDReportBuilder()
        builder.add_evidence(4, True, {"q2": np.float64(0.79)})
        assert json.loads(builder.build().to_json())

    def test_report_carries_the_endpoint(self):
        report = OECDReportBuilder(title="M", endpoint="pIC50").build()
        assert report.endpoint == "pIC50"

    def test_summary_section_comes_first(self):
        report = OECDReportBuilder(title="M").build()
        assert report.sections[0].title == "OECD validation principles"


class TestOutputFormats:
    """Every report renders to text, Markdown, HTML, JSON and PDF."""

    @pytest.fixture
    def report(self, predictions):
        import pandas as pd

        return (
            QSARReport(title="Format demo", endpoint="pIC50", author="tests")
            .add_dataset_section(n_compounds=24, n_train=18, n_test=6)
            .add_validation_section({"q2": 0.81, "rmse": 0.47})
            .add_section(
                "A table",
                table=pd.DataFrame({"compound": ["a", "b"], "value": [1.5, 2.5]}),
                text="Some explanatory prose.",
            )
        )

    @pytest.fixture
    def report_with_figure(self, predictions):
        return QSARReport(title="With a plot").add_validation_section(
            {"q2": 0.8}, figures=[plot_predicted_vs_observed(*predictions)]
        )

    def test_text_contains_title_and_tables(self, report):
        text = report.to_text()
        assert "Format demo" in text
        assert "Endpoint: pIC50" in text
        assert "n_compounds" in text
        assert "compound" in text and "value" in text   # the DataFrame

    def test_text_has_no_markup(self, report):
        text = report.to_text()
        assert "<" not in text and "|" not in text and "**" not in text

    def test_text_wraps_to_the_requested_width(self, report):
        narrow = report.to_text(width=40)
        assert all(len(line) <= 60 for line in narrow.splitlines()[:3])

    def test_text_writes_to_a_file(self, report, tmp_path):
        path = tmp_path / "report.txt"
        report.to_text(path)
        assert path.read_text(encoding="utf-8") == report.to_text()

    def test_text_names_figures_rather_than_dropping_them(self, report_with_figure):
        assert "[figure:" in report_with_figure.to_text()

    def test_markdown_contains_tables(self, report):
        markdown = report.to_markdown()
        assert markdown.startswith("# Format demo")
        assert "| Property | Value |" in markdown

    def test_html_is_a_complete_document(self, report):
        html = report.to_html()
        assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
        assert "<table>" in html

    def test_json_round_trips(self, report):
        import json

        restored = json.loads(report.to_json())
        assert restored["title"] == "Format demo"
        assert len(restored["sections"]) == 3

    def test_pdf_is_written(self, report, tmp_path):
        pytest.importorskip("reportlab")
        path = tmp_path / "report.pdf"
        returned = report.to_pdf(path)
        assert str(path) == returned
        assert path.stat().st_size > 0
        assert path.read_bytes().startswith(b"%PDF")

    def test_pdf_accepts_letter_size(self, report, tmp_path):
        pytest.importorskip("reportlab")
        assert report.to_pdf(tmp_path / "letter.pdf", page_size="letter")

    def test_pdf_rejects_an_unknown_page_size(self, report, tmp_path):
        pytest.importorskip("reportlab")
        with pytest.raises(ValueError, match="page_size"):
            report.to_pdf(tmp_path / "x.pdf", page_size="A9")

    def test_pdf_without_figures_needs_no_kaleido(self, report_with_figure, tmp_path):
        pytest.importorskip("reportlab")
        path = tmp_path / "nofig.pdf"
        report_with_figure.to_pdf(path, include_figures=False)
        assert path.stat().st_size > 0

    def test_pdf_reports_a_missing_figure_backend(self, report_with_figure, tmp_path):
        """Dropping the plots silently would make an incomplete report look done."""
        pytest.importorskip("reportlab")
        try:
            report_with_figure.to_pdf(tmp_path / "fig.pdf")
        except OptionalDependencyError as exc:
            assert exc.package == "kaleido"
        else:
            assert (tmp_path / "fig.pdf").stat().st_size > 0

    def test_markdown_reports_a_missing_figure_backend(self, report_with_figure):
        try:
            markdown = report_with_figure.to_markdown()
        except OptionalDependencyError as exc:
            assert exc.package == "kaleido"
        else:
            assert "![" in markdown

    def test_every_format_carries_the_same_content(self, report):
        for rendered in (report.to_text(), report.to_markdown(), report.to_html()):
            assert "Format demo" in rendered
            assert "0.81" in rendered


class TestValueFormatting:
    def test_nested_dicts_are_flattened_not_repred(self):
        report = QSARReport().add_section(
            "S", {"criteria": {"passed": True, "r2": 0.95}})
        text = report.to_text()
        assert "passed=yes" in text and "r2=0.95" in text
        assert "{'passed'" not in text

    def test_booleans_read_as_words(self):
        report = QSARReport().add_section("S", {"ok": True, "bad": False})
        text = report.to_text()
        assert "yes" in text and "no" in text

    def test_non_finite_floats_are_marked_not_a_number(self):
        report = QSARReport().add_section("S", {"ratio": float("nan")})
        assert "n/a" in report.to_text()

    def test_none_is_marked_not_available(self):
        report = QSARReport().add_section("S", {"q2": None})
        assert "n/a" in report.to_text()
