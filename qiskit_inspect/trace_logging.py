"""Logging helpers for qiskit_inspect."""

from __future__ import annotations

import logging
import sys
from typing import Optional, TextIO

_TRACE_LOGGER_NAME = "qiskit_inspect.trace"
_TRACE_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _trace_formatter() -> logging.Formatter:
    """Return a fresh formatter for the trace logger."""

    return logging.Formatter(_TRACE_FORMAT)


def enable_trace_logging(
    level: int = logging.INFO, stream: Optional[TextIO] = None, propagate: bool = False
) -> logging.Logger:
    """Configure and return the ``qiskit_inspect.trace`` logger.

    Args:
        level: Logging level (for example, ``logging.INFO``).
        stream: Optional stream to attach via ``logging.StreamHandler``. If ``None``,
            existing stream handlers are retargeted to the current ``sys.stderr`` and a
            new handler is created if necessary.
        propagate: Whether the logger should propagate messages to ancestor loggers.

    Returns:
        logging.Logger: The configured logger instance.

    Notes:
        Idempotent: if a handler already exists, it is reused, its level is updated, and
        its stream is refreshed.
    """
    logger = logging.getLogger(_TRACE_LOGGER_NAME)
    logger.setLevel(level)

    target_stream = stream if stream is not None else sys.stderr

    stream_handlers = [
        handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)
    ]

    if not stream_handlers:
        handler = logging.StreamHandler(target_stream)
        handler.setLevel(level)
        handler.setFormatter(_trace_formatter())
        logger.addHandler(handler)
        stream_handlers = [handler]

    for handler in stream_handlers:
        handler.setLevel(level)
        if stream is not None:
            handler.setStream(stream)
        else:
            handler.setStream(target_stream)
        if handler.formatter is None:
            handler.setFormatter(_trace_formatter())
    logger.propagate = propagate
    return logger
