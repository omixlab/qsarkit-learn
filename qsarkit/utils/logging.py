"""Package-wide logging configuration for qsarkit.

``qsarkit`` follows the standard library convention for libraries: every module
logs to a named logger under the ``qsarkit`` namespace and the package itself
attaches only a :class:`logging.NullHandler`, so an application that does not
configure logging sees no output. :func:`configure_logging` is the opt-in
helper that attaches a real handler.

References
----------
- Python Software Foundation. "Logging HOWTO - Configuring Logging for a
  Library." https://docs.python.org/3/howto/logging.html#library-config
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional, Union

_ROOT_LOGGER_NAME = "qsarkit"

#: Default record format used by :func:`configure_logging`.
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

#: Default timestamp format used by :func:`configure_logging`.
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _root_logger() -> logging.Logger:
    """Return the package root logger, ensuring it has a ``NullHandler``."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger inside the ``qsarkit`` namespace.

    Parameters
    ----------
    name : str, optional
        Logger name. A bare name (``"admet"``) or a dotted module name
        (``"qsarkit.admet._filters"``) are both accepted; the result is always
        rooted at ``qsarkit``. ``None`` returns the package root logger.

    Returns
    -------
    logging.Logger
        The requested logger. It has no handler of its own; records propagate
        to the ``qsarkit`` root logger, which carries a ``NullHandler`` unless
        :func:`configure_logging` was called.

    Examples
    --------
    >>> from qsarkit.utils import get_logger
    >>> log = get_logger(__name__)
    >>> log.name.startswith("qsarkit")
    True

    References
    ----------
    - Python Software Foundation. "Logging HOWTO."
      https://docs.python.org/3/howto/logging.html
    """
    root = _root_logger()
    if not name:
        return root
    if name == _ROOT_LOGGER_NAME:
        return root
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    level: Union[int, str] = logging.INFO,
    stream: Optional[object] = None,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> logging.Logger:
    """Attach a stream handler to the ``qsarkit`` root logger.

    This is an application-level convenience; libraries importing ``qsarkit``
    should not call it. Calling it repeatedly replaces the previously installed
    qsarkit handler rather than stacking duplicates.

    Parameters
    ----------
    level : int or str, default ``logging.INFO``
        Level for the ``qsarkit`` logger, e.g. ``"DEBUG"`` or ``logging.WARNING``.
    stream : file-like, optional
        Destination stream. Defaults to ``sys.stderr``.
    fmt : str, default :data:`DEFAULT_FORMAT`
        ``logging`` format string.
    datefmt : str, default :data:`DEFAULT_DATEFMT`
        ``time.strftime`` format for ``%(asctime)s``.

    Returns
    -------
    logging.Logger
        The configured ``qsarkit`` root logger.

    Examples
    --------
    >>> import io
    >>> from qsarkit.utils import configure_logging, get_logger
    >>> buf = io.StringIO()
    >>> _ = configure_logging("DEBUG", stream=buf)
    >>> get_logger("demo").debug("hello")
    >>> "hello" in buf.getvalue()
    True

    References
    ----------
    - Python Software Foundation. "Logging HOWTO."
      https://docs.python.org/3/howto/logging.html
    """
    logger = _root_logger()
    for existing in list(logger.handlers):
        if getattr(existing, "_qsarkit_handler", False):
            logger.removeHandler(existing)
    handler: logging.StreamHandler[Any] = logging.StreamHandler(
        stream if stream is not None else sys.stderr
    )
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handler._qsarkit_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger", "configure_logging", "DEFAULT_FORMAT", "DEFAULT_DATEFMT"]
