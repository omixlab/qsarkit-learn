"""A functional, left-to-right pipe API for molecular preprocessing.

A second way to express a curation workflow, reading in the order the
work happens — in the spirit of R's ``%>%``::

    from qsarkit.functional import *

    X, y = (
        molecules(smiles, activities)
        >> desalt()
        >> deglycate(keep_originals=True)
        >> remove_duplicates(agg="mean")
        >> balance()
    )

:func:`molecules` returns a :class:`MoleculeSet` carrying the molecules,
the optional labels and a provenance log. Each step consumes one and
returns a new one, so nothing is mutated in place, and every step keeps
``y`` index-aligned with the molecules — dropping a molecule drops its
label too.

The result unpacks straight into ``X, y``.

Why ``>>`` and not ``>``
------------------------

``>`` cannot work as a pipe operator in Python. The language parses
``a > b > c`` as the *chained comparison* ``(a > b) and (b > c)``, so a
``>``-based pipe evaluates the first stage, throws the result away, and
returns the comparison of the last two stages. There is no way to
intercept that from ``__gt__``, and the failure is silent — you get a
result, just not the one you asked for.

``>>`` (and ``|``) are ordinary binary operators, left-associative, so
they chain correctly. Piping with ``>`` raises a
:class:`TypeError` explaining this rather than misbehaving quietly.

Dual-mode steps
---------------

Every step works two ways, so the same function serves the pipe API and
ordinary imperative code:

>>> from qsarkit.functional import desalt, molecules
>>> from rdkit import Chem
>>> mols, y = molecules(["CC(=O)[O-].[Na+]"]) >> desalt()     # deferred
>>> mols, y = desalt([Chem.MolFromSmiles("CC(=O)[O-].[Na+]")])  # immediate

Reusable pipelines
------------------

Steps compose with each other, so a curation protocol can be defined once
and applied to several datasets:

>>> from qsarkit.functional import drop_invalid, standardize
>>> curate = standardize() >> drop_invalid() >> remove_duplicates()
>>> train = molecules(train_smiles, train_y) >> curate   # doctest: +SKIP
>>> test = molecules(test_smiles, test_y) >> curate      # doctest: +SKIP

References
----------
- Bache, S. M. & Wickham, H. (2014). "magrittr: A Forward-Pipe Operator
  for R." https://CRAN.R-project.org/package=magrittr
- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify: On
  the Importance of Chemical Structure Curation in Cheminformatics and
  QSAR Modeling Research." J. Chem. Inf. Model., 50(7), 1189-1204.
  https://doi.org/10.1021/ci100176x
"""

from qsarkit.functional._core import (
    FeatureSet,
    FeatureStep,
    MoleculeSet,
    PipeStep,
    Step,
    feature_step,
    molecules,
    pipeline,
    step,
)
from qsarkit.functional._model_steps import (
    applicability_domain,
    collect,
    cross_validate,
    describe,
    drop_constant,
    drop_correlated,
    featurize,
    fingerprint,
    fit,
    impute,
    resample,
    scale,
    select_features,
    split,
)
from qsarkit.functional._steps import (
    apply,
    balance,
    canonicalize_tautomers,
    deglycate,
    desalt,
    drop_if,
    drop_invalid,
    filter_by_property,
    keep_if,
    neutralize,
    remove_duplicates,
    remove_protecting_groups,
    sample,
    shuffle,
    standardize,
    to_pactivity,
)
from qsarkit.functional._viz import (
    PipelineNode,
    pipeline_nodes,
    plot_pipeline,
    render_pipeline,
    to_dot,
)

__all__ = [
    # core
    "molecules",
    "MoleculeSet",
    "FeatureSet",
    "PipeStep",
    "Step",
    "FeatureStep",
    "step",
    "feature_step",
    "pipeline",
    # curation steps (molecules -> molecules)
    "standardize",
    "desalt",
    "neutralize",
    "canonicalize_tautomers",
    "deglycate",
    "remove_protecting_groups",
    "drop_invalid",
    "remove_duplicates",
    "balance",
    "keep_if",
    "drop_if",
    "filter_by_property",
    "to_pactivity",
    "sample",
    "shuffle",
    "apply",
    # representation (molecules -> features)
    "featurize",
    "fingerprint",
    "describe",
    # feature steps (features -> features)
    "scale",
    "impute",
    "drop_constant",
    "drop_correlated",
    "select_features",
    "resample",
    # terminals
    "split",
    "fit",
    "cross_validate",
    "applicability_domain",
    "collect",
    # flowchart
    "PipelineNode",
    "pipeline_nodes",
    "to_dot",
    "plot_pipeline",
    "render_pipeline",
]
