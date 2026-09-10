"""QSAR modeling estimators: kernels, regressors, classifiers and ensembling.

This subpackage provides scikit-learn-compatible estimators tailored to
QSAR/QSPR modeling on molecular fingerprints and descriptors: a Tanimoto
kernel for Gaussian processes, thin QSAR-sane wrappers around common
scikit-learn regressors, a PLS regressor with VIP variable-importance
scores, a naive-baseline model for OECD-principle-4 comparisons, a
consensus/ensembling dispatcher, and unified ``name``-dispatching facades
(:class:`QSARRegressor`, :class:`QSARClassifier`).
"""

from qsarkit.models._baseline import BaselineModel
from qsarkit.models._consensus import ConsensusModel
from qsarkit.models._facades import QSARClassifier, QSARRegressor
from qsarkit.models._gaussian_process import GaussianProcessQSAR
from qsarkit.models._neural_network import NeuralNetworkQSAR
from qsarkit.models._pls import PLSRegressor
from qsarkit.models._random_forest import RandomForestQSAR
from qsarkit.models._svm import SVMQSAR
from qsarkit.models._tanimoto_kernel import TanimotoKernel

__all__ = [
    "QSARRegressor",
    "QSARClassifier",
    "PLSRegressor",
    "ConsensusModel",
    "TanimotoKernel",
    "GaussianProcessQSAR",
    "RandomForestQSAR",
    "SVMQSAR",
    "NeuralNetworkQSAR",
    "BaselineModel",
]
