"""QSAR model reports in Markdown, HTML and JSON, including QMRF format."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

__all__ = ["ReportSection", "QSARReport", "OECDReportBuilder"]

PathLike = Union[str, Path]


def _jsonable(value: Any) -> Any:
    """Convert numpy and other non-JSON types into serializable ones."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class ReportSection:
    """One titled section of a report.

    Attributes
    ----------
    title : str
        Section heading.
    content : dict
        Key-value pairs rendered as a definition table.
    text : str
        Free-form prose placed above the table.
    figures : list
        Plotly figures embedded in the HTML rendering.
    table : Any
        An optional pandas DataFrame rendered after the content.
    """

    title: str
    content: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    figures: List[Any] = field(default_factory=list)
    table: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form, excluding figures."""
        payload: Dict[str, Any] = {
            "title": self.title,
            "content": _jsonable(self.content),
        }
        if self.text:
            payload["text"] = self.text
        if self.table is not None:
            payload["table"] = _jsonable(self.table.to_dict(orient="records"))
        return payload


class QSARReport:
    """Assemble a complete, auditable report for a QSAR model.

    A model is only reproducible if the record says what data it was
    built on, how that data was curated, which descriptors and algorithm
    were used, how it was validated and where it applies. This collects
    all of that into one object that renders to Markdown, HTML or JSON.

    Sections are added in whatever order suits the model; the renderers
    preserve that order.

    Parameters
    ----------
    title : str
        Report title.
    author : str
        Who built the model.
    endpoint : str
        What is predicted, e.g. ``"pIC50 (CHEMBL204, IC50)"``. OECD
        principle 1 asks for exactly this.

    Examples
    --------
    >>> report = QSARReport(title="EGFR pIC50 model", endpoint="pIC50")
    >>> _ = report.add_section("Dataset", {"n_compounds": 1200})
    >>> "EGFR" in report.to_markdown()
    True

    References
    ----------
    - OECD (2007). "Guidance Document on the Validation of (Quantitative)
      Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
      Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    - Tropsha, A. (2010). "Best Practices for QSAR Model Development,
      Validation, and Exploitation." Mol. Inform., 29(6-7), 476-488.
      https://doi.org/10.1002/minf.201000061
    - Patlewicz, G. et al. (2008). "An Evaluation of the Implementation
      of the OECD (Q)SAR Application Toolbox." SAR QSAR Environ. Res.,
      19(5-6), 397-412. https://doi.org/10.1080/10629360802083848
    """

    def __init__(
        self,
        title: str = "QSAR model report",
        author: str = "",
        endpoint: str = "",
    ) -> None:
        self.title = title
        self.author = author
        self.endpoint = endpoint
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.sections: List[ReportSection] = []

    def add_section(
        self,
        title: str,
        content: Optional[Dict[str, Any]] = None,
        text: str = "",
        figures: Optional[Sequence[Any]] = None,
        table: Any = None,
    ) -> "QSARReport":
        """Append a section.

        Parameters
        ----------
        title : str
            Section heading.
        content : dict, optional
            Key-value pairs rendered as a table.
        text : str
            Prose placed above the table.
        figures : sequence, optional
            Plotly figures, embedded in HTML output only.
        table : pandas.DataFrame, optional
            Tabular data rendered after the content.

        Returns
        -------
        QSARReport
            ``self``, so calls can be chained.
        """
        self.sections.append(
            ReportSection(
                title=title,
                content=dict(content) if content else {},
                text=text,
                figures=list(figures) if figures else [],
                table=table,
            )
        )
        return self

    def add_dataset_section(
        self,
        n_compounds: int,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        source: str = "",
        curation: Optional[Any] = None,
    ) -> "QSARReport":
        """Add the dataset section, optionally including a curation log.

        Parameters
        ----------
        n_compounds : int
            Compounds after curation.
        n_train, n_test : int, optional
            Split sizes.
        source : str
            Where the data came from.
        curation : CurationReport, optional
            Output of
            :class:`~qsarkit.data_quality.DataCurationPipeline`, whose
            per-stage counts are what make the dataset auditable.

        Returns
        -------
        QSARReport
        """
        content: Dict[str, Any] = {"n_compounds": n_compounds}
        if n_train is not None:
            content["n_train"] = n_train
        if n_test is not None:
            content["n_test"] = n_test
        if source:
            content["source"] = source
        table = None
        if curation is not None:
            content["n_input_records"] = curation.n_input
            content["n_removed"] = curation.n_removed
            content["retention"] = round(curation.retention, 4)
            table = curation.to_dataframe()
        return self.add_section("Dataset", content, table=table)

    def add_model_section(
        self,
        model: Any,
        descriptors: str = "",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> "QSARReport":
        """Add the algorithm section (OECD principle 2).

        Parameters
        ----------
        model : estimator
            The fitted model; its class name and parameters are recorded.
        descriptors : str
            Description of the representation used.
        hyperparameters : dict, optional
            Overrides what is read from the model.

        Returns
        -------
        QSARReport
        """
        params = hyperparameters
        if params is None:
            params = (
                model.get_params() if hasattr(model, "get_params") else {}
            )
        content: Dict[str, Any] = {"algorithm": type(model).__name__}
        if descriptors:
            content["descriptors"] = descriptors
        content.update({f"param.{k}": v for k, v in sorted(params.items())})
        return self.add_section("Algorithm", content)

    def add_validation_section(
        self,
        metrics: Dict[str, Any],
        y_scrambling: Optional[Dict[str, Any]] = None,
        figures: Optional[Sequence[Any]] = None,
    ) -> "QSARReport":
        """Add the validation section (OECD principle 4).

        Parameters
        ----------
        metrics : dict
            Goodness-of-fit, robustness and predictivity measures.
        y_scrambling : dict, optional
            Output of a y-randomization run, which is the evidence that
            the fit is not chance correlation.
        figures : sequence, optional
            Diagnostic plots.

        Returns
        -------
        QSARReport
        """
        content = dict(metrics)
        if y_scrambling:
            content.update(
                {f"scrambling.{k}": v for k, v in y_scrambling.items()}
            )
        return self.add_section("Validation", content, figures=figures)

    def add_applicability_section(
        self,
        domain: Any,
        coverage: Optional[float] = None,
        report: Optional[Dict[str, Any]] = None,
        figures: Optional[Sequence[Any]] = None,
    ) -> "QSARReport":
        """Add the applicability-domain section (OECD principle 3).

        Parameters
        ----------
        domain : estimator
            The fitted domain; its class name and threshold are recorded.
        coverage : float, optional
            Fraction of the evaluation set inside the domain.
        report : dict, optional
            Output of :meth:`~qsarkit.applicability.ADAnalyzer.report`.
        figures : sequence, optional

        Returns
        -------
        QSARReport
        """
        content: Dict[str, Any] = {"method": type(domain).__name__}
        threshold = getattr(domain, "threshold_", None)
        if threshold is not None:
            content["threshold"] = threshold
        if coverage is not None:
            content["coverage"] = coverage
        if report:
            content.update(report)
        return self.add_section(
            "Applicability domain", content, figures=figures
        )

    def to_dict(self) -> Dict[str, Any]:
        """The whole report as a JSON-serializable dictionary."""
        return {
            "title": self.title,
            "author": self.author,
            "endpoint": self.endpoint,
            "created": self.created,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_json(self, path: Optional[PathLike] = None, indent: int = 2) -> str:
        """Render as JSON, optionally writing it to ``path``.

        Parameters
        ----------
        path : str or Path, optional
        indent : int, default 2

        Returns
        -------
        str
            The JSON text.
        """
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(
        self,
        path: Optional[PathLike] = None,
        include_figures: bool = True,
    ) -> str:
        """Render as Markdown, optionally writing it to ``path``.

        Parameters
        ----------
        path : str or Path, optional
            Write the Markdown here as well as returning it. When given,
            figures are written as PNG files in a sibling directory and
            linked relatively, which is what a Markdown file in a
            repository needs.
        include_figures : bool, default True
            Embed the report's figures. Requires ``kaleido``
            (``pip install qsarkit[reporting]``); pass ``False`` for a
            text-and-tables document without it.

        Returns
        -------
        str
            The Markdown text.

        Raises
        ------
        OptionalDependencyError
            If ``include_figures`` is set, the report has figures, and
            ``kaleido`` is missing.
        """
        figures = [f for section in self.sections for f in section.figures]
        images: Dict[int, bytes] = {}
        asset_dir: Optional[Path] = None
        if include_figures and figures:
            images = _rasterize_figures(figures)
            if path is not None:
                asset_dir = Path(path).with_suffix("").parent / (
                    Path(path).stem + "_figures"
                )
                asset_dir.mkdir(parents=True, exist_ok=True)
        figure_index = 0

        lines = [f"# {self.title}", ""]
        header = [
            ("Endpoint", self.endpoint),
            ("Author", self.author),
            ("Created", self.created),
        ]
        for label, value in header:
            if value:
                lines.append(f"**{label}:** {value}  ")
        lines.append("")

        for section in self.sections:
            lines.extend([f"## {section.title}", ""])
            if section.text:
                lines.extend([section.text, ""])
            if section.content:
                lines.extend(["| Property | Value |", "|---|---|"])
                for key, value in section.content.items():
                    lines.append(f"| {key} | {_format_value(value)} |")
                lines.append("")
            if section.table is not None:
                lines.extend([_dataframe_to_markdown(section.table), ""])
            for figure in section.figures:
                caption = _figure_title(figure)
                png = images.get(figure_index)
                figure_index += 1
                if png is None:
                    lines.extend([f"_[figure: {caption}]_", ""])
                elif asset_dir is not None:
                    name = f"figure_{figure_index:02d}.png"
                    (asset_dir / name).write_bytes(png)
                    lines.extend([f"![{caption}]({asset_dir.name}/{name})", ""])
                else:
                    # No file to sit beside, so inline it. Data URIs render
                    # in most Markdown viewers, though not on GitHub.
                    encoded = base64.b64encode(png).decode("ascii")
                    lines.extend(
                        [f"![{caption}](data:image/png;base64,{encoded})", ""]
                    )
        if path is not None:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        return "\n".join(lines)

    def to_html(
        self, path: Optional[PathLike] = None, include_plotlyjs: str = "cdn"
    ) -> str:
        """Render as a self-describing HTML document with embedded figures.

        Assembled directly rather than through a template engine, so the
        core report has no optional dependency at all.

        Parameters
        ----------
        path : str or Path, optional
        include_plotlyjs : str, default "cdn"
            ``"cdn"`` keeps the file small; ``True`` inlines Plotly so
            the report works with no network.

        Returns
        -------
        str
            The HTML document.
        """
        from qsarkit.reporting._plots import figure_to_html

        blocks: List[str] = []
        for section in self.sections:
            block = [f"<section><h2>{_escape(section.title)}</h2>"]
            if section.text:
                block.append(f"<p>{_escape(section.text)}</p>")
            if section.content:
                block.append(
                    "<table><thead><tr><th>Property</th><th>Value</th></tr>"
                    "</thead><tbody>"
                )
                for key, value in section.content.items():
                    block.append(
                        f"<tr><td>{_escape(str(key))}</td>"
                        f"<td>{_escape(_format_value(value))}</td></tr>"
                    )
                block.append("</tbody></table>")
            if section.table is not None:
                block.append(section.table.to_html(index=False, border=0))
            for i, figure in enumerate(section.figures):
                # Plotly's library is included once, with the first figure.
                first = not blocks and i == 0
                block.append(
                    figure_to_html(
                        figure,
                        include_plotlyjs=include_plotlyjs if first else False,
                    )
                )
            block.append("</section>")
            blocks.append("\n".join(block))

        header = "".join(
            f"<p><strong>{label}:</strong> {_escape(value)}</p>"
            for label, value in (
                ("Endpoint", self.endpoint),
                ("Author", self.author),
                ("Created", self.created),
            )
            if value
        )
        body = f"<h1>{_escape(self.title)}</h1>{header}" + "\n".join(blocks)
        html = _HTML_TEMPLATE.format(title=_escape(self.title), body=body)

        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def to_text(self, path: Optional[PathLike] = None, width: int = 78) -> str:
        """Render as plain text, optionally writing it to ``path``.

        The format for a terminal, a log, or an email: no markup, ASCII
        tables, and figures listed by title rather than dropped silently.

        Parameters
        ----------
        path : str or Path, optional
            Write the text here as well as returning it.
        width : int, default 78
            Column width for rules and wrapping.

        Returns
        -------
        str
            The plain-text report.

        Examples
        --------
        >>> from qsarkit.reporting import QSARReport
        >>> report = QSARReport(title="Demo", endpoint="pIC50")
        >>> _ = report.add_dataset_section(n_compounds=24, n_train=18, n_test=6)
        >>> print(report.to_text(width=40))
        ========================================
        Demo
        ========================================
        Endpoint: pIC50
        Created: ...
        <BLANKLINE>
        Dataset
        ----------------------------------------
          n_compounds  24
          n_train      18
          n_test       6
        <BLANKLINE>
        """
        import textwrap

        lines: List[str] = ["=" * width, self.title, "=" * width]
        for label, value in (
            ("Endpoint", self.endpoint),
            ("Author", self.author),
            ("Created", self.created),
        ):
            if value:
                lines.append(f"{label}: {value}")
        lines.append("")

        for section in self.sections:
            lines.extend([section.title, "-" * width])
            if section.text:
                lines.extend(textwrap.wrap(section.text, width=width) + [""])
            if section.content:
                pairs = [
                    (str(key), _format_value(value))
                    for key, value in section.content.items()
                ]
                key_width = max((len(k) for k, _ in pairs), default=0)
                for key, value in pairs:
                    lines.append(f"  {key.ljust(key_width)}  {value}")
                lines.append("")
            if section.table is not None:
                lines.extend([section.table.to_string(index=False), ""])
            for figure in section.figures:
                lines.append(f"  [figure: {_figure_title(figure)}]")
            if section.figures:
                lines.append("")

        text = "\n".join(lines)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_pdf(
        self,
        path: PathLike,
        include_figures: bool = True,
        page_size: str = "A4",
    ) -> str:
        """Render as a PDF with tables and embedded plots.

        Parameters
        ----------
        path : str or Path
            Output file. Unlike the other renderers this one is
            file-only: a PDF is binary and there is nothing useful to
            return as a string.
        include_figures : bool, default True
            Rasterize and embed the report's Plotly figures. Requires
            ``kaleido`` (``pip install qsarkit[reporting]``). Pass
            ``False`` to produce a tables-and-text PDF without it.
        page_size : {"A4", "letter"}, default "A4"

        Returns
        -------
        str
            The path written.

        Raises
        ------
        OptionalDependencyError
            If ``reportlab`` is missing, or if ``include_figures`` is set,
            the report has figures, and ``kaleido`` is missing. The second
            case raises rather than quietly dropping the plots: a report
            silently missing its evidence is worse than no report.
        ValueError
            If ``page_size`` is not recognized.

        Examples
        --------
        >>> import tempfile, os
        >>> from qsarkit.reporting import QSARReport
        >>> report = QSARReport(title="Demo", endpoint="pIC50")
        >>> _ = report.add_dataset_section(n_compounds=24)
        >>> out = os.path.join(tempfile.mkdtemp(), "report.pdf")
        >>> _ = report.to_pdf(out)
        >>> os.path.getsize(out) > 0
        True

        References
        ----------
        - ReportLab documentation:
          https://docs.reportlab.com/
        """
        from qsarkit.base import require

        require("reportlab")
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        sizes = {"A4": A4, "letter": letter}
        if page_size not in sizes:
            raise ValueError(
                f"page_size must be one of {sorted(sizes)}, got {page_size!r}."
            )

        figures = [f for s in self.sections for f in s.figures]
        images: Dict[int, bytes] = {}
        if include_figures and figures:
            images = _rasterize_figures(figures)

        styles = getSampleStyleSheet()
        body = ParagraphStyle(
            "qsarkit-body", parent=styles["BodyText"], spaceAfter=6, leading=13
        )
        story: List[Any] = [
            Paragraph(_escape(self.title), styles["Title"]),
        ]
        for label, value in (
            ("Endpoint", self.endpoint),
            ("Author", self.author),
            ("Created", self.created),
        ):
            if value:
                story.append(
                    Paragraph(f"<b>{label}:</b> {_escape(value)}", body)
                )
        story.append(Spacer(1, 6 * mm))

        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2933")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])

        figure_index = 0
        for section in self.sections:
            story.append(Paragraph(_escape(section.title), styles["Heading2"]))
            if section.text:
                story.append(Paragraph(_escape(section.text), body))
            if section.content:
                rows = [["Property", "Value"]] + [
                    [
                        Paragraph(_escape(str(key)), body),
                        Paragraph(_escape(_format_value(value)), body),
                    ]
                    for key, value in section.content.items()
                ]
                table = Table(rows, colWidths=[55 * mm, 105 * mm], repeatRows=1)
                table.setStyle(table_style)
                story.extend([table, Spacer(1, 4 * mm)])
            if section.table is not None:
                frame = section.table
                rows = [[str(c) for c in frame.columns]] + [
                    [_format_value(v) for v in row]
                    for row in frame.itertuples(index=False)
                ]
                table = Table(rows, repeatRows=1)
                table.setStyle(table_style)
                story.extend([table, Spacer(1, 4 * mm)])
            for _ in section.figures:
                png = images.get(figure_index)
                figure_index += 1
                if png is None:
                    continue
                import io

                story.extend([
                    Image(io.BytesIO(png), width=150 * mm, height=90 * mm,
                          kind="proportional"),
                    Spacer(1, 4 * mm),
                ])

        document = SimpleDocTemplate(
            str(path),
            pagesize=sizes[page_size],
            title=self.title,
            author=self.author or None,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
        )
        document.build(story)
        return str(path)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<QSARReport {self.title!r}: {len(self.sections)} sections>"


def _figure_title(figure: Any) -> str:
    """Best available name for a Plotly figure, for text listings."""
    try:
        title = figure.layout.title.text
    except AttributeError:
        title = None
    return str(title) if title else "untitled"


def _rasterize_figures(figures: Sequence[Any]) -> Dict[int, bytes]:
    """Render Plotly figures to PNG bytes, keyed by position.

    Raises
    ------
    OptionalDependencyError
        If kaleido is unavailable. Raising is deliberate: a report that
        silently lost its plots looks complete and is not.
    """
    from qsarkit.base.exceptions import OptionalDependencyError

    images: Dict[int, bytes] = {}
    for i, figure in enumerate(figures):
        try:
            images[i] = figure.to_image(format="png", scale=2)
        except Exception as exc:
            # Deliberately broad: a missing or misconfigured kaleido surfaces
            # as ImportError, ValueError, RuntimeError or a Plotly-internal
            # type depending on version, and all mean the same thing here.
            # The original is chained, so nothing is hidden.
            raise OptionalDependencyError("kaleido", "reporting") from exc
    return images



def _dataframe_to_markdown(frame: Any) -> str:
    """Render a DataFrame as a Markdown table.

    Written out rather than delegating to ``DataFrame.to_markdown``,
    which requires the optional ``tabulate`` package -- an undeclared
    hard dependency for anyone whose report contains a table.
    """
    columns = [str(c) for c in frame.columns]
    rows = [
        [_format_value(value) for value in record]
        for record in frame.itertuples(index=False)
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Render a value for a report table.

    Nested dicts are flattened into ``key=value`` pairs rather than
    printed as a Python repr: a validation report routinely nests one
    (the Golbraikh-Tropsha criteria, for instance), and a raw repr in a
    table cell is unreadable in every output format.
    """
    if isinstance(value, bool):
        # Checked before float, since bool is a subclass of int.
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "; ".join(
            f"{key}={_format_value(item)}"
            for key, item in value.items()
            if item is not None
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        shown = ", ".join(_format_value(v) for v in items[:5])
        return shown + (f", ... ({len(items)} items)" if len(items) > 5 else "")
    if value is None:
        return "n/a"
    return str(value)


def _escape(text: str) -> str:
    """Minimal HTML escaping for report text."""
    from html import escape

    return escape(str(text))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 60rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;
        color: #222; }}
 h1 {{ border-bottom: 2px solid #2874a6; padding-bottom: .3rem; }}
 h2 {{ margin-top: 2rem; color: #2874a6; }}
 table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
 th, td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
 th {{ background: #f4f6f7; }}
 section {{ margin-bottom: 2rem; }}
</style></head><body>
{body}
</body></html>"""


class OECDReportBuilder:
    """Build a QMRF-style report against the five OECD principles.

    The QSAR Model Reporting Format is the structure regulators expect,
    and its sections map onto the five validation principles. This
    assembles one from the artefacts the rest of the package produces,
    and — importantly — records explicitly when a principle has *not*
    been addressed, since a silent omission is what makes a submission
    fail review.

    Parameters
    ----------
    title : str
        Model name.
    author : str
        Who built it.
    endpoint : str
        The defined endpoint (principle 1).

    Examples
    --------
    >>> builder = OECDReportBuilder(title="EGFR model", endpoint="pIC50")
    >>> report = builder.build()
    >>> "OECD" in report.to_markdown()
    True

    Principles you have not addressed are tracked, and appear in the
    report as explicit gaps:

    >>> builder.unaddressed
    [1, 2, 3, 4, 5]
    >>> _ = builder.add_evidence(1, True, {"endpoint": "pIC50, CHEMBL204"})
    >>> builder.unaddressed
    [2, 3, 4, 5]

    :meth:`build` returns a :class:`QSARReport`, which renders to plain
    text, Markdown, HTML, JSON and PDF -- tables and plots included:

    >>> report = builder.build()
    >>> print(report.to_text(width=48).splitlines()[1])
    EGFR model
    >>> report.to_html().startswith("<!DOCTYPE html>")
    True
    >>> sorted(report.to_dict())
    ['author', 'created', 'endpoint', 'sections', 'title']

    References
    ----------
    - OECD (2007). "Guidance Document on the Validation of (Quantitative)
      Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
      Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    - OECD (2004). "The Report from the Expert Group on (Q)SARs on the
      Principles for the Validation of (Q)SARs." ENV/JM/MONO(2004)24.
    - European Commission Joint Research Centre. "QSAR Model Reporting
      Format (QMRF)."
      https://joint-research-centre.ec.europa.eu/scientific-tools-databases/qsar-toolbox_en
    """

    PRINCIPLES = (
        (1, "A defined endpoint"),
        (2, "An unambiguous algorithm"),
        (3, "A defined domain of applicability"),
        (
            4,
            "Appropriate measures of goodness-of-fit, robustness and "
            "predictivity",
        ),
        (5, "A mechanistic interpretation, if possible"),
    )

    def __init__(
        self,
        title: str = "QMRF report",
        author: str = "",
        endpoint: str = "",
    ) -> None:
        self.title = title
        self.author = author
        self.endpoint = endpoint
        self._evidence: Dict[int, Dict[str, Any]] = {}

    def add_evidence(
        self, principle: int, addressed: bool, details: Dict[str, Any]
    ) -> "OECDReportBuilder":
        """Record how one principle was addressed.

        Parameters
        ----------
        principle : int
            1 to 5.
        addressed : bool
            Whether the principle is satisfied.
        details : dict
            Supporting values.

        Returns
        -------
        OECDReportBuilder
            ``self``, so calls can be chained.
        """
        if principle not in {n for n, _ in self.PRINCIPLES}:
            raise ValueError(
                f"principle must be 1-5, got {principle}."
            )
        self._evidence[principle] = {
            "addressed": bool(addressed),
            **_jsonable(details),
        }
        return self

    def from_validation(self, validation: Dict[str, Any]) -> "OECDReportBuilder":
        """Populate principles 3 and 4 from a validation report.

        Parameters
        ----------
        validation : dict
            A mapping carrying validation metrics, e.g. from
            :mod:`qsarkit.validation`.

        Returns
        -------
        OECDReportBuilder
        """
        metrics = {
            k: v for k, v in validation.items() if isinstance(v, (int, float, str))
        }
        return self.add_evidence(4, bool(metrics), metrics)

    def build(self) -> QSARReport:
        """Assemble the report.

        Returns
        -------
        QSARReport
            With one section per OECD principle, each marked addressed or
            not.
        """
        report = QSARReport(
            title=self.title, author=self.author, endpoint=self.endpoint
        )
        report.add_section(
            "OECD validation principles",
            text=(
                "This report follows the QSAR Model Reporting Format (QMRF) "
                "and documents the model against the five OECD validation "
                "principles. A principle marked not addressed is an explicit "
                "gap, not an omission."
            ),
            content={
                f"Principle {number}": (
                    "addressed"
                    if self._evidence.get(number, {}).get("addressed")
                    else "NOT ADDRESSED"
                )
                for number, _ in self.PRINCIPLES
            },
        )
        for number, name in self.PRINCIPLES:
            evidence = dict(self._evidence.get(number, {"addressed": False}))
            addressed = evidence.pop("addressed", False)
            report.add_section(
                f"Principle {number}: {name}",
                content={"status": "addressed" if addressed else "NOT ADDRESSED",
                         **evidence},
            )
        return report

    @property
    def unaddressed(self) -> List[int]:
        """Principles with no recorded evidence.

        Returns
        -------
        list of int
            The principle numbers a reviewer will ask about.
        """
        return [
            number
            for number, _ in self.PRINCIPLES
            if not self._evidence.get(number, {}).get("addressed")
        ]
