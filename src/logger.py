"""
Structured logging setup — used across training, API, and inference.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Return a logger with console + optional file handler.

    Parameters
    ----------
    name     : Module name (use __name__ in callers).
    log_file : Optional path for file output. If None, console only.
    """
    from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_DIR

    logger = logging.getLogger(name)

    if logger.handlers:           # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (append mode)
    target = log_file or (LOG_DIR / "app.log")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except OSError:
        logger.warning("Could not create log file at %s; logging to console only.", target)

    return logger
