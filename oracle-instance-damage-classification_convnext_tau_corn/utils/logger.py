"""Simple project logger."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, output_dir: str | None = None, filename: str = "log.txt") -> logging.Logger:
    """Create a console + file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

