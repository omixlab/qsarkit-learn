"""Flowchart rendering for functional pipelines.

A pipeline built with ``>>`` is a graph, and drawing it is the fastest
way to check that the stages are in the order you meant -- in the spirit
of Keras' ``plot_model`` and Bonobo's graph output.

Two renderers, because they fail in opposite directions:

* **Graphviz** (:func:`to_dot`) produces the classic boxes-and-arrows
  layout. It needs the ``dot`` binary, which is not a Python dependency.
* **Plotly** (:func:`plot_pipeline`) needs nothing beyond what qsarkit
  already requires, and matches the rest of the package's plotting.

:func:`render_pipeline` writes PNG or PDF using whichever is available,
preferring Graphviz for its layout quality.

Nodes are coloured by the domain they operate on -- molecules, features,
or a terminal result -- so the point where a pipeline crosses from
chemistry into a feature matrix is visible at a glance.

Examples
--------
>>> from qsarkit.functional import desalt, drop_invalid, fingerprint, fit
>>> pipe = desalt() >> drop_invalid() >> fingerprint() >> fit("rf")
>>> print(pipe.to_dot())          # doctest: +ELLIPSIS
digraph qsarkit_pipeline {
...
>>> figure = pipe.plot()
>>> type(figure).__name__
'Figure'

References
----------
- Gansner, E. R. & North, S. C. (2000). "An Open Graph Visualization
  System and Its Applications to Software Engineering." Softw. Pract.
  Exp., 30(11), 1203-1233.
  https://doi.org/10.1002/1097-024X(200009)30:11<1203::AID-SPE338>3.0.CO;2-N
- Graphviz documentation: https://graphviz.org/documentation/
- Plotly Python documentation: https://plotly.com/python/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

__all__ = ["PipelineNode", "pipeline_nodes", "to_dot", "plot_pipeline", "render_pipeline"]

#: Node fill colours by domain. Chosen to stay legible printed in
#: greyscale, which is where a flowchart usually ends up.
_DOMAIN_STYLE: Dict[str, Dict[str, str]] = {
    "input": {"fill": "#E8EEF7", "line": "#41618F", "shape": "ellipse"},
    "molecules": {"fill": "#DCEBDC", "line": "#3D7A3D", "shape": "box"},
    "transition": {"fill": "#FBEBD2", "line": "#B37A1E", "shape": "box"},
    "features": {"fill": "#E4E1F0", "line": "#5B4C93", "shape": "box"},
    "terminal": {"fill": "#F7DEDE", "line": "#9E3B3B", "shape": "ellipse"},
}

#: What each domain produces, drawn on the edge leaving that node.
_DOMAIN_OUTPUT: Dict[str, str] = {
    "input": "MoleculeSet",
    "molecules": "MoleculeSet",
    "transition": "FeatureSet",
    "features": "FeatureSet",
    "terminal": "",
}


class PipelineNode:
    """One stage of a pipeline, as it appears in the flowchart.

    Parameters
    ----------
    label : str
        Step name with its configured arguments.
    domain : {"input", "molecules", "transition", "features", "terminal"}
        What the step consumes and produces, which sets its colour and
        the label on its outgoing edge.
    detail : str, optional
        Second line of the node, e.g. the estimator or transformer class.

    Attributes
    ----------
    label : str
    domain : str
    detail : str or None
    """

    __slots__ = ("label", "domain", "detail")

    def __init__(self, label: str, domain: str, detail: Optional[str] = None) -> None:
        self.label = label
        self.domain = domain
        self.detail = detail

    def __repr__(self) -> str:
        return f"<PipelineNode {self.label!r} ({self.domain})>"


def _classify(step: Any) -> Tuple[str, Optional[str]]:
    """Domain and detail line for one step."""
    from qsarkit.functional._core import FeatureStep, Step
    from qsarkit.functional._model_steps import _Featurize, _Terminal

    if isinstance(step, _Featurize):
        # The label already names the transformer; a detail line would
        # just repeat it.
        return "transition", None
    if isinstance(step, _Terminal):
        estimator = getattr(step, "estimator", None)
        detail = None
        if estimator is not None and not isinstance(estimator, str):
            detail = type(estimator).__name__
        return "terminal", detail
    if isinstance(step, FeatureStep):
        return "features", None
    if isinstance(step, Step):
        return "molecules", None
    return "molecules", None


def pipeline_nodes(pipe: Any, include_input: bool = True) -> List[PipelineNode]:
    """Flatten a pipeline into the nodes of its flowchart.

    Parameters
    ----------
    pipe : PipeStep, MoleculeSet or FeatureSet
        A step, a composed pipeline, or a set whose ``history`` is drawn.
    include_input : bool, default True
        Prepend an input node representing the incoming molecules.

    Returns
    -------
    list of PipelineNode

    Examples
    --------
    >>> from qsarkit.functional import desalt, fingerprint, pipeline_nodes
    >>> [n.domain for n in pipeline_nodes(desalt() >> fingerprint())]
    ['input', 'molecules', 'transition']
    """
    from qsarkit.functional._core import FeatureSet, MoleculeSet, PipeStep, _Composed

    nodes: List[PipelineNode] = []
    if include_input:
        nodes.append(PipelineNode("input", "input", "molecules + y"))

    if isinstance(pipe, (MoleculeSet, FeatureSet)):
        # A set records what has already happened, as strings.
        for entry in pipe.history:
            if entry.startswith("molecules("):
                continue
            domain = "transition" if entry.startswith("featurize(") else "molecules"
            nodes.append(PipelineNode(entry, domain))
        return nodes

    if not isinstance(pipe, PipeStep):
        raise TypeError(
            f"Expected a Step, MoleculeSet or FeatureSet, got {type(pipe).__name__}."
        )

    steps = pipe.steps if isinstance(pipe, _Composed) else [pipe]
    for s in steps:
        domain, detail = _classify(s)
        nodes.append(PipelineNode(s._describe(), domain, detail))
    return nodes


def _escape(text: str) -> str:
    """Escape a label for inclusion in a DOT string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(
    pipe: Any,
    name: str = "qsarkit_pipeline",
    rankdir: str = "TB",
    include_input: bool = True,
) -> str:
    """Render a pipeline as Graphviz DOT source.

    Parameters
    ----------
    pipe : PipeStep, MoleculeSet or FeatureSet
        The pipeline to draw.
    name : str, default "qsarkit_pipeline"
        Graph name.
    rankdir : {"TB", "LR"}, default "TB"
        Layout direction: top-to-bottom or left-to-right.
    include_input : bool, default True
        Draw the input node.

    Returns
    -------
    str
        DOT source, renderable with ``dot -Tpng`` or by the ``graphviz``
        Python package.

    Examples
    --------
    >>> from qsarkit.functional import desalt, fingerprint, to_dot
    >>> dot = to_dot(desalt() >> fingerprint())
    >>> dot.splitlines()[0]
    'digraph qsarkit_pipeline {'
    >>> "desalt()" in dot
    True

    References
    ----------
    - Graphviz DOT language: https://graphviz.org/doc/info/lang.html
    """
    if rankdir not in ("TB", "LR"):
        raise ValueError(f"rankdir must be 'TB' or 'LR', got {rankdir!r}.")

    nodes = pipeline_nodes(pipe, include_input=include_input)
    lines = [
        f"digraph {name} {{",
        f"  rankdir={rankdir};",
        '  node [style="filled,rounded", shape=box, fontname="Helvetica", '
        'fontsize=11, margin="0.18,0.10"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#666666"];',
        '  bgcolor="transparent";',
    ]

    for i, node in enumerate(nodes):
        style = _DOMAIN_STYLE[node.domain]
        label = _escape(node.label)
        if node.detail:
            label += f"\\n{_escape(node.detail)}"
        lines.append(
            f'  n{i} [label="{label}", fillcolor="{style["fill"]}", '
            f'color="{style["line"]}", shape={style["shape"]}];'
        )

    for i in range(len(nodes) - 1):
        carried = _DOMAIN_OUTPUT[nodes[i].domain]
        edge_label = f' [label=" {carried}"]' if carried else ""
        lines.append(f"  n{i} -> n{i + 1}{edge_label};")

    lines.append("}")
    return "\n".join(lines)


def plot_pipeline(
    pipe: Any,
    title: str = "Pipeline",
    include_input: bool = True,
    orientation: str = "vertical",
) -> "go.Figure":
    """Render a pipeline as a Plotly flowchart.

    The dependency-free renderer: it needs only what qsarkit already
    requires, and returns a figure like every other plot in the package
    (never shown, never written to disk).

    Parameters
    ----------
    pipe : PipeStep, MoleculeSet or FeatureSet
        The pipeline to draw.
    title : str, default "Pipeline"
        Figure title.
    include_input : bool, default True
        Draw the input node.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Direction of flow.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> from qsarkit.functional import desalt, fingerprint, plot_pipeline
    >>> figure = plot_pipeline(desalt() >> fingerprint())
    >>> type(figure).__name__
    'Figure'

    Export needs kaleido (``pip install qsarkit[reporting]``):

    >>> figure.write_image("pipeline.png")     # doctest: +SKIP

    References
    ----------
    - Plotly Python documentation: https://plotly.com/python/
    """
    import plotly.graph_objects as go

    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got {orientation!r}."
        )

    nodes = pipeline_nodes(pipe, include_input=include_input)
    n = len(nodes)
    vertical = orientation == "vertical"

    # One unit of spacing per node; boxes occupy most of it so the arrows
    # between them stay visible.
    spacing = 1.0
    half_w, half_h = (0.42, 0.30) if vertical else (0.44, 0.34)

    shapes: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []

    for i, node in enumerate(nodes):
        # Draw top-to-bottom: the first step belongs at the top.
        pos = -i * spacing if vertical else i * spacing
        cx, cy = (0.0, pos) if vertical else (pos, 0.0)
        style = _DOMAIN_STYLE[node.domain]
        rounded = style["shape"] == "ellipse"

        shapes.append(
            {
                "type": "circle" if rounded else "rect",
                "x0": cx - half_w,
                "x1": cx + half_w,
                "y0": cy - half_h,
                "y1": cy + half_h,
                "fillcolor": style["fill"],
                "line": {"color": style["line"], "width": 1.6},
                "layer": "below",
            }
        )

        text = node.label
        if node.detail:
            text += f"<br><span style='font-size:10px'>{node.detail}</span>"
        annotations.append(
            {
                "x": cx,
                "y": cy,
                "text": text,
                "showarrow": False,
                "font": {"size": 11, "color": "#1F2933"},
                "align": "center",
            }
        )

        if i == n - 1:
            continue

        # Arrow to the next node, labelled with what flows along it.
        nxt = -(i + 1) * spacing if vertical else (i + 1) * spacing
        if vertical:
            ax, ay, bx, by = cx, cy - half_h, cx, nxt + half_h
        else:
            ax, ay, bx, by = cx + half_w, cy, nxt - half_w, cy
        annotations.append(
            {
                "x": bx,
                "y": by,
                "ax": ax,
                "ay": ay,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "text": "",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1.1,
                "arrowwidth": 1.4,
                "arrowcolor": "#7B8794",
            }
        )
        carried = _DOMAIN_OUTPUT[node.domain]
        if carried:
            annotations.append(
                {
                    "x": (ax + bx) / 2 + (0.06 if vertical else 0.0),
                    "y": (ay + by) / 2 + (0.0 if vertical else 0.10),
                    "text": carried,
                    "showarrow": False,
                    "font": {"size": 9, "color": "#7B8794"},
                    "xanchor": "left" if vertical else "center",
                }
            )

    figure = go.Figure()
    # An invisible trace pins the axes; shapes and annotations alone do not.
    figure.add_trace(
        go.Scatter(
            x=[0.0] * n if vertical else [i * spacing for i in range(n)],
            y=[-i * spacing for i in range(n)] if vertical else [0.0] * n,
            mode="markers",
            marker={"size": 0.1, "color": "rgba(0,0,0,0)"},
            hovertext=[f"{node.label} [{node.domain}]" for node in nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )

    span = (n - 1) * spacing
    if vertical:
        x_range, y_range = (-1.1, 1.1), (-span - 0.6, 0.6)
        width, height = 520, int(130 * n + 90)
    else:
        x_range, y_range = (-0.7, span + 0.7), (-1.0, 1.0)
        width, height = int(220 * n + 120), 340

    figure.update_layout(
        title=title,
        shapes=shapes,
        annotations=annotations,
        xaxis={"visible": False, "range": list(x_range), "fixedrange": True},
        yaxis={
            "visible": False,
            "range": list(y_range),
            "fixedrange": True,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        width=width,
        height=height,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_pipeline(
    pipe: Any,
    path: str,
    engine: str = "auto",
    include_input: bool = True,
    **kwargs: Any,
) -> str:
    """Write a pipeline flowchart to a PNG, PDF or SVG file.

    Parameters
    ----------
    pipe : PipeStep, MoleculeSet or FeatureSet
        The pipeline to draw.
    path : str
        Output file. The extension chooses the format: ``.png``,
        ``.pdf`` or ``.svg``.
    engine : {"auto", "graphviz", "plotly"}, default "auto"
        Renderer. ``"auto"`` uses Graphviz when the ``graphviz`` package
        and its ``dot`` binary are both present, and Plotly otherwise.
    include_input : bool, default True
        Draw the input node.
    **kwargs
        Passed to the chosen renderer (``rankdir`` for Graphviz,
        ``title``/``orientation`` for Plotly).

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If the extension is not a supported format, or ``engine`` is not
        one of the three accepted values.
    OptionalDependencyError
        If the requested engine's dependency is missing.

    Examples
    --------
    >>> from qsarkit.functional import desalt, fingerprint, render_pipeline
    >>> render_pipeline(desalt() >> fingerprint(), "pipeline.pdf")  # doctest: +SKIP
    'pipeline.pdf'

    Notes
    -----
    The Plotly path needs ``kaleido`` for static export
    (``pip install qsarkit[reporting]``); the Graphviz path needs the
    ``dot`` binary, which is a system package rather than a Python one.

    References
    ----------
    - Graphviz documentation: https://graphviz.org/documentation/
    - Kaleido: https://github.com/plotly/Kaleido
    """
    import os

    suffix = os.path.splitext(path)[1].lower().lstrip(".")
    if suffix not in ("png", "pdf", "svg"):
        raise ValueError(
            f"Unsupported output format {suffix!r}. Use .png, .pdf or .svg."
        )
    if engine not in ("auto", "graphviz", "plotly"):
        raise ValueError(
            f"engine must be 'auto', 'graphviz' or 'plotly', got {engine!r}."
        )

    if engine in ("auto", "graphviz"):
        rendered = _try_graphviz(pipe, path, suffix, include_input, kwargs)
        if rendered is not None:
            return rendered
        if engine == "graphviz":
            from qsarkit.base.exceptions import OptionalDependencyError

            raise OptionalDependencyError("graphviz")

    figure = plot_pipeline(
        pipe,
        include_input=include_input,
        **{k: v for k, v in kwargs.items() if k in ("title", "orientation")},
    )
    try:
        figure.write_image(path, format=suffix)
    except Exception as exc:  # kaleido missing, or no engine for this format
        from qsarkit.base.exceptions import OptionalDependencyError

        raise OptionalDependencyError("kaleido", "reporting") from exc
    return path


def _try_graphviz(
    pipe: Any,
    path: str,
    suffix: str,
    include_input: bool,
    kwargs: Dict[str, Any],
) -> Optional[str]:
    """Render with Graphviz, or return ``None`` if it is unavailable."""
    import os

    try:
        import graphviz
    except ImportError:
        return None

    source = to_dot(
        pipe,
        rankdir=kwargs.get("rankdir", "TB"),
        include_input=include_input,
    )
    stem, _ = os.path.splitext(path)
    try:
        graphviz.Source(source).render(
            filename=stem, format=suffix, cleanup=True, quiet=True
        )
    except Exception:
        # `dot` binary absent or failed; fall back to Plotly.
        return None
    return path
