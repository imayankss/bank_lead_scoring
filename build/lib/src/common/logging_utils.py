"""
Central logging utilities for the bank_lead_scoring_project.

Usage:
    from src.common.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("message")
"""

from __future__ import annotations

import logging
from typing import Optional

# Internal flag so we configure the root logger only once
_LOGGER_CONFIGURED = False


def configure_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a consistent format.

    Call this once at process startup. `get_logger` will call it lazily
    the first time you request a logger if it hasn't been configured yet.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _LOGGER_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the given name, ensuring the root logger is configured.

    Example:
        logger = get_logger(__name__)
        logger.info("ETL started")
    """
    configure_root_logger()
    return logging.getLogger(name)


__all__ = ["get_logger", "configure_root_logger"]
