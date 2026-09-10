"""Multi-layer perceptron regressor with QSAR-sane defaults."""

from __future__ import annotations

from typing import Optional, Tuple, Union

from sklearn.neural_network import MLPRegressor

__all__ = ["NeuralNetworkQSAR"]


class NeuralNetworkQSAR(MLPRegressor):
    """Feed-forward neural network regressor tuned with QSAR-sane defaults.

    A thin subclass of :class:`sklearn.neural_network.MLPRegressor` that
    keeps the full parent parameter set but defaults to a two-hidden-layer
    architecture with early stopping and moderate L2 regularization —
    settings that guard against the overfitting risk of neural networks
    on the small-to-medium (hundreds to low thousands of compounds) QSAR
    datasets typical of real drug-discovery projects, where a network
    with the scikit-learn defaults (a single 100-unit layer, no
    regularization, no early stopping) would otherwise happily memorize
    the training set.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default (100, 50)
        Sizes of the hidden layers. Two layers give the network enough
        capacity for non-linear structure-activity relationships without
        the data requirements of a deeper architecture.
    activation : {"identity", "logistic", "tanh", "relu"}, default "relu"
        Activation function of the hidden layers.
    alpha : float, default 1e-3
        L2 regularization strength, an order of magnitude above
        scikit-learn's default to counter overfitting on small QSAR
        datasets.
    early_stopping : bool, default True
        Hold out part of the training data and stop when validation
        score stops improving, which is a cheap and effective safeguard
        against overfitting on limited QSAR data.
    random_state : int, optional
        Seed for reproducible weight initialization and data shuffling.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=60, n_features=5, random_state=0)
    >>> model = NeuralNetworkQSAR(max_iter=200, random_state=0).fit(X, y)
    >>> model.predict(X).shape
    (60,)

    References
    ----------
    - Winkler, D. A. (2004). "Neural Networks as Robust Tools in Drug
      Design and Analysis." Mol. Biotechnol., 27(2), 139-167.
      https://doi.org/10.1385/MB:27:2:139
    """

    def __init__(
        self,
        loss: str = "squared_error",
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        activation: str = "relu",
        *,
        solver: str = "adam",
        alpha: float = 1e-3,
        batch_size: Union[str, int] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
    ) -> None:
        super().__init__(
            loss=loss,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
