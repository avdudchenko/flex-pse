"""Custom logging system for flex-pse with per-level deduplication.

Provides a project-wide logger, a custom ``CONFIGURATION_SIMPLIFICATIONS`` log
level, and a deduplicating handler that suppresses repeated messages within a
sliding time window. Global runtime controls allow the application to adjust the
minimum log level and dedup behaviour without touching configuration files.

Note: This module was generated with AI assistance.
"""

import logging
import threading
import time
from collections import deque

CONFIGURATION_SIMPLIFICATIONS = 21
logging.addLevelName(CONFIGURATION_SIMPLIFICATIONS, "CONFIGURATION_SIMPLIFICATIONS")

DEFAULT_LOGGER_NAME = "flex-pse"
DEFAULT_LOGGER_LEVEL = CONFIGURATION_SIMPLIFICATIONS

DEFAULT_DEDUP_ENABLED = {
    logging.WARNING: True,
    CONFIGURATION_SIMPLIFICATIONS: True,
}

_GLOBAL_LOGGER_LEVEL = DEFAULT_LOGGER_LEVEL
_GLOBAL_DEDUP_ENABLED: dict[int, bool] | None = None


def get_global_level() -> int:
    """Return the current global minimum log level for flex-pse loggers.

    Note: This module was generated with AI assistance.

    Returns
    -------
    int
        The current global log level (defaults to
        ``CONFIGURATION_SIMPLIFICATIONS``). Individual loggers created via
        :func:`get_logger` inherit this level at creation time.
    """
    return _GLOBAL_LOGGER_LEVEL


def set_global_level(level: int) -> None:
    """Set the global minimum log level for subsequently created loggers.

    Note: This module was generated with AI assistance.

    Parameters
    ----------
    level : int
        The new global log level. Standard levels (e.g. ``logging.DEBUG``,
        ``logging.INFO``) or custom levels (e.g. ``CONFIGURATION_SIMPLIFICATIONS``)
        are accepted.
    """
    global _GLOBAL_LOGGER_LEVEL
    _GLOBAL_LOGGER_LEVEL = int(level)


def get_global_dedup_enabled() -> dict[int, bool]:
    """Return the global dedup configuration, keyed by log level.

    Note: This module was generated with AI assistance.

    Returns
    -------
    dict[int, bool]
        Mapping of ``levelno`` to ``True`` (dedup enabled) or ``False``
        (dedup disabled). Missing levels default to ``False``. The default
        mapping enables dedup for ``logging.WARNING`` and
        ``CONFIGURATION_SIMPLIFICATIONS``.
    """
    global _GLOBAL_DEDUP_ENABLED
    if _GLOBAL_DEDUP_ENABLED is None:
        _GLOBAL_DEDUP_ENABLED = dict(DEFAULT_DEDUP_ENABLED)
    return dict(_GLOBAL_DEDUP_ENABLED)


def set_global_dedup_enabled(dedup_enabled: dict[int, bool]) -> None:
    """Replace the global dedup configuration for subsequently created handlers.

    Note: This module was generated with AI assistance.

    Parameters
    ----------
    dedup_enabled : dict[int, bool]
        New mapping of ``levelno`` to dedup flag. Existing handlers keep their
        current state; call :func:`get_logger` again or mutate
        ``handler.dedup_enabled`` directly to update live handlers.
    """
    global _GLOBAL_DEDUP_ENABLED
    _GLOBAL_DEDUP_ENABLED = dict(dedup_enabled)


class FlexPseLogger(logging.Logger):
    """Project logger class that adds a ``configuration_simplifications`` method.

    Note: This module was generated with AI assistance.

    Registered via ``logging.setLoggerClass`` so that ``logging.getLogger``
    returns an instance of this class when the name matches. The added method
    emits log records at the custom ``CONFIGURATION_SIMPLIFICATIONS`` level,
    which is intended for user-facing physical-simplification notices that
    should appear in normal output but benefit from deduplication.
    """

    def configuration_simplifications(self, msg, *args, **kwargs):
        """Log a message at the ``CONFIGURATION_SIMPLIFICATIONS`` custom level.

        Note: This module was generated with AI assistance.

        Args:
            msg: The message to log. Supports ``%``-style formatting with ``args``.
            *args: Positional arguments interpolated into ``msg``.
            **kwargs: Extra keyword arguments forwarded to the underlying
                :class:`logging.Logger` emit call.
        """
        if self.isEnabledFor(CONFIGURATION_SIMPLIFICATIONS):
            self._log(CONFIGURATION_SIMPLIFICATIONS, msg, args, **kwargs)


logging.setLoggerClass(FlexPseLogger)


class DedupHandler(logging.StreamHandler):
    """Logging handler that suppresses duplicate messages within a sliding window.

    Note: this module was module were generated with AI assistance.

    Duplicates are keyed on ``(message, pathname, lineno, levelno)``. The
    handler is thread-safe and allows an optional ``target`` handler for
    forwarding, making it composable with other handlers (e.g. a file handler
    plus a dedup wrapper).
    """

    def __init__(self, target=None, window=10.0, dedup_enabled=None):
        """Initialise the dedup handler.

        Note: This module was generated with AI assistance.

        Parameters
        ----------
        target : logging.Handler, optional
            If provided, records are forwarded to this handler rather than
            emitted directly by this handler.
        window : float
            Sliding window duration in seconds. A duplicate message is allowed
            again after ``window`` seconds have elapsed since its first
            appearance.
        dedup_enabled : dict[int, bool], optional
            Mapping of ``levelno`` to dedup flag. When ``None``, the current
            global dedup configuration is used.
        """
        super().__init__()
        self._target = target
        self.window = window
        source = (
            dedup_enabled if dedup_enabled is not None else get_global_dedup_enabled()
        )
        self.dedup_enabled = dict(source)
        self._lock = threading.Lock()
        self._deque = deque()
        self._map = {}

    def emit(self, record):
        """Process a log record, emitting it if it is not a recent duplicate.

        Note: This module was generated with AI assistance.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to process.

        Returns
        -------
        None
            Returns ``None`` when the record is suppressed as a duplicate.
        """
        levelno = record.levelno
        if not self.dedup_enabled.get(levelno, False):
            if self._target is not None:
                self._target.handle(record)
            else:
                super().emit(record)
            return

        key = (record.getMessage(), record.pathname, record.lineno, levelno)
        now = time.monotonic()

        with self._lock:
            while self._deque and now - self._deque[0][1] > self.window:
                old_key, _ = self._deque.popleft()
                self._map.pop(old_key, None)

            if key in self._map:
                return

            self._deque.append((key, now))
            self._map[key] = now

        if self._target is not None:
            self._target.handle(record)
        else:
            super().emit(record)


def get_logger(name=None, dedup_enabled=None):
    """Return a project logger with a ``DedupHandler`` attached.

    Note: This module was generated with AI assistance.

    The logger is configured with the global log level and a single
    ``DedupHandler``. If the logger already has a ``DedupHandler``, no new
    handler is added, allowing callers to re-acquire the same logger safely.

    Parameters
    ----------
    name : str, optional
        Logger name. Defaults to ``DEFAULT_LOGGER_NAME`` (``"flex-pse"``).
    dedup_enabled : dict[int, bool], optional
        Dedup mapping for the attached ``DedupHandler``. When ``None``, the
        current global dedup configuration is used.

    Returns
    -------
    logging.Logger
        A logger instance with dedup-aware output.
    """
    if name is None:
        name = DEFAULT_LOGGER_NAME
    logger = logging.getLogger(name)
    logger.setLevel(_GLOBAL_LOGGER_LEVEL)
    logger.propagate = False

    if not any(isinstance(h, DedupHandler) for h in logger.handlers):
        effective_dedup = (
            dedup_enabled if dedup_enabled is not None else get_global_dedup_enabled()
        )
        dedup_handler = DedupHandler(dedup_enabled=effective_dedup)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        dedup_handler.setFormatter(formatter)
        logger.addHandler(dedup_handler)

    return logger
