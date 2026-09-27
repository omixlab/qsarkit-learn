"""Bridging the scikit-learn versions the package supports.

``requires-python`` is ``>=3.9`` and the dependency floor is
``scikit-learn>=1.2``, which spans several years of releases. Parameters come
and go across that range: ``MLPRegressor`` gained ``loss`` and ``MDS`` gained
``init`` in versions that require Python 3.10 or newer, so code passing them
unconditionally raises ``TypeError`` on exactly the interpreter the package
claims to support.

Forwarding such a parameter only when the installed estimator accepts it keeps
one code path working across the whole range. Silently dropping a value the
caller asked for would be worse than failing, so a non-default value on a
version that cannot honour it raises with a message naming the version.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict, Mapping

__all__ = ["accepts_parameter", "forward_supported"]


@lru_cache(maxsize=None)
def accepts_parameter(cls: type, name: str) -> bool:
    """Whether ``cls.__init__`` takes a parameter called ``name``.

    Cached, because it is consulted on every construction and the answer
    cannot change within a process.

    Parameters
    ----------
    cls : type
    name : str

    Returns
    -------
    bool

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from qsarkit.base import accepts_parameter
    >>> accepts_parameter(RandomForestRegressor, "n_estimators")
    True
    >>> accepts_parameter(RandomForestRegressor, "not_a_parameter")
    False
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - C-level __init__
        return False
    parameters = signature.parameters
    if name in parameters:
        return True
    # A constructor taking **kwargs accepts anything.
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def forward_supported(
    cls: type, candidates: Mapping[str, Any], defaults: Mapping[str, Any]
) -> Dict[str, Any]:
    """Keep the entries ``cls`` accepts; refuse to drop a non-default silently.

    Parameters
    ----------
    cls : type
        The estimator being constructed.
    candidates : mapping
        Parameters to forward if supported.
    defaults : mapping
        The value each candidate has by default. A candidate equal to its
        default is dropped quietly when unsupported, because the caller did
        not ask for it; anything else raises.

    Returns
    -------
    dict
        The subset of ``candidates`` that ``cls`` accepts.

    Raises
    ------
    TypeError
        If a candidate differs from its default and ``cls`` cannot accept it.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from qsarkit.base import forward_supported
    >>> forward_supported(
    ...     RandomForestRegressor,
    ...     {"n_estimators": 50, "nonexistent": "default"},
    ...     {"n_estimators": 100, "nonexistent": "default"},
    ... )
    {'n_estimators': 50}
    """
    import sklearn

    forwarded: Dict[str, Any] = {}
    for name, value in candidates.items():
        if accepts_parameter(cls, name):
            forwarded[name] = value
        elif value != defaults.get(name):
            raise TypeError(
                f"{cls.__name__} in scikit-learn {sklearn.__version__} has no "
                f"{name!r} parameter, so {name}={value!r} cannot be honoured. "
                f"Upgrade scikit-learn, or leave {name} at its default."
            )
    return forwarded
