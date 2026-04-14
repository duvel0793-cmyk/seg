from __future__ import annotations

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {idx: name for name, idx in LABEL_TO_INDEX.items()}

