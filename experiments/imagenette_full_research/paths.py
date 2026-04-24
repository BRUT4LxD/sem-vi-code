"""
Directory layout for the full ImageNette research pipeline (default root: ``final_research/``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FullResearchPaths:
    """All roots used by :mod:`experiments.imagenette_full_research.runner`."""

    root: str = "final_research"
    imagenette_train: str = "./data/imagenette/train"
    imagenette_val: str = "./data/imagenette/val"

    def __post_init__(self) -> None:
        self.root = os.path.normpath(self.root)

    @property
    def runs(self) -> str:
        """TensorBoard log root (e.g. ``final_research/runs``)."""
        return os.path.join(self.root, "runs")

    @property
    def models_normal(self) -> str:
        return os.path.join(self.root, "models", "normal")

    @property
    def results_normal(self) -> str:
        return os.path.join(self.root, "results", "normal")

    @property
    def data_attacks_normal(self) -> str:
        return os.path.join(self.root, "data", "attacks", "normal")

    @property
    def results_attacks_normal(self) -> str:
        return os.path.join(self.root, "results", "attacks", "normal")

    @property
    def models_progressive_active(self) -> str:
        return os.path.join(self.root, "models", "progressive", "active")

    @property
    def data_progressive(self) -> str:
        return os.path.join(self.root, "data", "progressive")

    @property
    def data_attacks_progressive_active(self) -> str:
        return os.path.join(self.root, "data", "attacks", "progressive", "active")

    @property
    def results_progressive_active(self) -> str:
        return os.path.join(self.root, "results", "progressive", "active")

    @property
    def results_attacks_progressive_active(self) -> str:
        return os.path.join(self.root, "results", "attacks", "progressive", "active")

    @property
    def models_progressive_passive(self) -> str:
        return os.path.join(self.root, "models", "progressive", "passive")

    @property
    def results_progressive_passive(self) -> str:
        return os.path.join(self.root, "results", "progressive", "passive")

    @property
    def data_attacks_progressive_passive(self) -> str:
        return os.path.join(self.root, "data", "attacks", "progressive", "passive")

    @property
    def results_attacks_progressive_passive(self) -> str:
        return os.path.join(self.root, "results", "attacks", "progressive", "passive")

    @property
    def models_noise_detection(self) -> str:
        return os.path.join(self.root, "models", "noise_detection")

    @property
    def results_transferability_normal(self) -> str:
        return os.path.join(self.root, "results", "transferability", "from_normal")

    @property
    def results_transferability_passive(self) -> str:
        return os.path.join(self.root, "results", "transferability", "from_passive")

    def ensure_dirs(self) -> None:
        for p in (
            self.runs,
            self.models_normal,
            self.results_normal,
            self.data_attacks_normal,
            self.results_attacks_normal,
            self.models_progressive_active,
            self.data_progressive,
            self.data_attacks_progressive_active,
            self.results_progressive_active,
            self.results_attacks_progressive_active,
            self.models_progressive_passive,
            self.results_progressive_passive,
            self.data_attacks_progressive_passive,
            self.results_attacks_progressive_passive,
            self.models_noise_detection,
            self.results_transferability_normal,
            self.results_transferability_passive,
        ):
            os.makedirs(p, exist_ok=True)


def _parse_yaml_scalar(val: str) -> Any:
    """Parse a single YAML scalar (bools, numbers, or plain string)."""
    val = val.strip()
    if not val:
        return ""
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    try:
        if "." in val or "e" in low:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _load_simple_indented_yaml(path: str) -> Dict[str, Any]:
    """
    Minimal YAML reader for configs like ``config.yaml`` (top-level sections + 2-space keys).
    Used when PyYAML is not installed. Ignores ``#`` comments and blank lines.
    """
    root: Dict[str, Any] = {}
    section: Optional[str] = None
    section_key_re = re.compile(r"^([A-Za-z_]\w*):\s*$")
    kv_re = re.compile(r"^([ \t]+)([A-Za-z_]\w*):\s*(.*)$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith((" ", "\t")):
                m = section_key_re.match(line)
                if m:
                    section = m.group(1)
                    root[section] = {}
                    continue
                continue
            m = kv_re.match(line)
            if m and section is not None:
                _, key, rest = m.group(1), m.group(2), m.group(3)
                root[section][key] = _parse_yaml_scalar(rest)
    return root


def paths_from_mapping(data: Dict[str, Any], base: Optional[FullResearchPaths] = None) -> FullResearchPaths:
    """Overlay non-empty keys from ``data`` onto a :class:`FullResearchPaths` instance."""
    p = base or FullResearchPaths()
    if not data:
        return p
    for k, v in data.items():
        if hasattr(p, k) and v is not None and v != "":
            setattr(p, k, v)
    return p


def load_yaml_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    lower = path.lower()
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        try:
            import yaml  # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            return _load_simple_indented_yaml(path)
    if lower.endswith(".json"):
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config extension: {path}")
