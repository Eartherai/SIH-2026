"""Translate a detector's native class labels into the AQUA-SHIELD taxonomy.

Keeping this separate from the detector means swapping or retraining a model can
never silently change what a hazard report *means*. The mapping lives in
data/class_mapping.yaml and is validated on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_MAPPING = Path("data/class_mapping.yaml")


@dataclass(frozen=True)
class TaxonEntry:
    native_name: str
    level1: str          # MAN_MADE | AMBIGUOUS
    level2: str          # e.g. mine_like_object
    note: str

    @property
    def is_man_made(self) -> bool:
        return self.level1 == "MAN_MADE"


@lru_cache(maxsize=8)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"class mapping not found at {p}. AQUA-SHIELD refuses to guess a taxonomy."
        )
    return yaml.safe_load(p.read_text())


class Taxonomy:
    """Maps native detector class ids -> AQUA-SHIELD semantics for one source domain."""

    def __init__(self, source: str, mapping_path: str | Path = DEFAULT_MAPPING):
        cfg = _load(str(mapping_path))
        sources = cfg.get("sources", {})
        if source not in sources:
            raise KeyError(f"unknown source '{source}'. known: {sorted(sources)}")
        self.source = source
        self.meta = sources[source]
        self.level2_defs = cfg["level2"]
        self._by_id: dict[int, TaxonEntry] = {}
        for cid, spec in self.meta["native_classes"].items():
            l2 = spec["aqs_level2"]
            if l2 not in self.level2_defs:
                raise ValueError(f"{source} class {cid} maps to undefined level2 '{l2}'")
            self._by_id[int(cid)] = TaxonEntry(
                native_name=spec["native_name"],
                level1=spec["aqs_level1"],
                level2=l2,
                note=spec.get("note", ""),
            )

    def __getitem__(self, class_id: int) -> TaxonEntry:
        if int(class_id) not in self._by_id:
            # Never invent a category for an id the mapping does not cover.
            return TaxonEntry(f"UNMAPPED_{class_id}", "AMBIGUOUS", "unknown_anomaly",
                              "Detector emitted a class id absent from class_mapping.yaml.")
        return self._by_id[int(class_id)]

    @property
    def names(self) -> dict[int, str]:
        return {k: v.native_name for k, v in self._by_id.items()}

    @property
    def citation(self) -> str:
        return self.meta.get("citation", "")

    @property
    def license(self) -> str:
        return self.meta.get("license", "unknown")

    def describe(self, class_id: int) -> str:
        e = self[class_id]
        return f"{e.native_name} -> {e.level2} ({e.level1})"
