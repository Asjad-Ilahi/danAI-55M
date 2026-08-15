"""
Dataset mixture system per §9 & §7.

Supports upsampling high-quality sources relative to raw text,
and separating main training phase mixture from WSD annealing phase mixture.
"""

import random
from pathlib import Path
from typing import List, Dict, Any, Iterator

from src.utils.config import Config


class DataMixture:
    """Manages weighted sampling across raw data sources."""

    def __init__(self, config_path: str | Path = "configs/data_mixture.yaml"):
        self.config = Config.from_yaml(config_path)
        self.main_sources = self.config.data_mixture.main
        self.annealing_sources = self.config.data_mixture.get("annealing", self.main_sources)

    def _normalize_weights(self, sources: List[Any]) -> Tuple_and_Weights:
        paths = []
        weights = []
        for s in sources:
            p = Path(s["path"])
            w = float(s.get("weight", 1.0)) * float(s.get("upsample_weight", 1.0))
            if p.exists():
                paths.append(p)
                weights.append(w)

        if not weights:
            raise FileNotFoundError("No valid data paths found in data_mixture.yaml configuration!")

        total = sum(weights)
        norm_weights = [w / total for w in weights]
        return paths, norm_weights

    def get_document_stream(self, phase: str = "main", seed: int = 42) -> Iterator[str]:
        """Stream documents sampled proportionally according to mixture weights."""
        sources = self.main_sources if phase == "main" else self.annealing_sources
        paths, weights = self._normalize_weights(sources)

        rng = random.Random(seed)

        # Gather files per source
        file_lists = []
        for p in paths:
            exts = ["*.txt", "*.jsonl", "*.md"]
            files = []
            for ext in exts:
                files.extend(list(p.glob(f"**/{ext}")))
            file_lists.append(files)

        # Infinite generator streaming documents weighted by source
        while True:
            source_idx = rng.choices(range(len(paths)), weights=weights, k=1)[0]
            files = file_lists[source_idx]
            if not files:
                continue

            chosen_file = rng.choice(files)
            try:
                with open(chosen_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if content.strip():
                        yield content
            except Exception:
                continue
