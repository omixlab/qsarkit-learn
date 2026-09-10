"""scikit-learn glue for composing molecule transformers into pipelines.

Building preprocessing as a ``Pipeline`` rather than applying it up front
is what keeps imputation and scaling fitted on training folds only --
the leak that otherwise inflates cross-validated QSAR scores.

Examples
--------
>>> from qsarkit.representation import MorganFingerprint
>>> from qsarkit.transform import make_qsar_pipeline
>>> from sklearn.ensemble import RandomForestRegressor
>>> pipe = make_qsar_pipeline(
...     MorganFingerprint(n_bits=64), RandomForestRegressor(n_estimators=5),
...     from_smiles=True,
... )
>>> _ = pipe.fit(["CCO", "CCN", "c1ccccc1"], [1.0, 2.0, 3.0])
>>> pipe.predict(["CCO"]).shape
(1,)

References
----------
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
  Python." J. Mach. Learn. Res., 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
- Cawley, G. C. & Talbot, N. L. C. (2010). "On Over-fitting in Model
  Selection and Subsequent Selection Bias in Performance Evaluation."
  J. Mach. Learn. Res., 11, 2079-2107.
  https://jmlr.org/papers/v11/cawley10a.html
"""

from qsarkit.transform._transforms import (
    DescriptorScaler,
    MolToSmiles,
    MoleculeFeatureUnion,
    NaNHandler,
    SmilesToMol,
    VarianceThresholdMol,
    make_qsar_pipeline,
)

__all__ = [
    "SmilesToMol",
    "MolToSmiles",
    "MoleculeFeatureUnion",
    "NaNHandler",
    "VarianceThresholdMol",
    "DescriptorScaler",
    "make_qsar_pipeline",
]
