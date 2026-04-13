"""xBD label mappings for damage subtype classification."""

from __future__ import annotations


DAMAGE_LABEL_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

LABEL_TO_DAMAGE = {value: key for key, value in DAMAGE_LABEL_MAP.items()}
CLASS_NAMES = [LABEL_TO_DAMAGE[idx] for idx in range(len(LABEL_TO_DAMAGE))]

