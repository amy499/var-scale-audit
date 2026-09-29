"""Manifest reading: the one place that knows the manifest file format.

Current format: CSV with header `class_id,seed`, one row per image.
Standard library only.
"""

import csv
from pathlib import Path
from typing import NamedTuple

COLUMNS = ("class_id", "seed")


class ManifestError(ValueError):
    pass


class Row(NamedTuple):
    class_id: int
    seed: int

    @property
    def stem(self) -> str:
        """Output file name (no extension) for this image."""
        return f"class{self.class_id:04d}_seed{self.seed}"


def load_manifest(path) -> list[Row]:
    """Return the manifest rows in file order. Duplicate (class_id, seed) rows are an error."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or tuple(h.strip() for h in header) != COLUMNS:
            raise ManifestError(f"{path}: header must be {','.join(COLUMNS)}, got {header}")
        rows, seen = [], set()
        for lineno, rec in enumerate(reader, start=2):
            if not rec or all(not c.strip() for c in rec):
                continue
            if len(rec) != len(COLUMNS):
                raise ManifestError(f"{path}:{lineno}: expected {len(COLUMNS)} columns, got {rec}")
            try:
                row = Row(*(int(c) for c in rec))
            except ValueError:
                raise ManifestError(f"{path}:{lineno}: class_id and seed must be integers, got {rec}") from None
            if row.seed < 0:
                raise ManifestError(f"{path}:{lineno}: seed must be >= 0, got {row.seed}")
            if row in seen:
                raise ManifestError(f"{path}:{lineno}: duplicate row {row}")
            seen.add(row)
            rows.append(row)
    if not rows:
        raise ManifestError(f"{path}: no rows")
    return rows
