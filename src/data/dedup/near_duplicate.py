"""
Cross-dataset MinHash 13-gram near-duplicate detector per §15.
"""

import hashlib
import re
from typing import Set, List


class MinHashDedup:
    """
    MinHash near-duplicate detector using 13-grams.
    Memory-efficient index across dataset sources.
    """

    def __init__(self, num_perm: int = 64, ngram_size: int = 13, threshold: float = 0.80):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.threshold = threshold

        # Generate linear hash coefficients (a * x + b) % p
        self.prime = 4294967311
        import random
        rng = random.Random(42)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

        # Hash buckets for band-based LSH
        self.num_bands = 8
        self.rows_per_band = num_perm // self.num_bands
        self.band_buckets: List[Set[int]] = [set() for _ in range(self.num_bands)]

    def _get_ngrams(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.ngram_size:
            return [" ".join(words)]
        return [" ".join(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]

    def _compute_minhash(self, text: str) -> List[int]:
        ngrams = self._get_ngrams(text)
        ngram_hashes = [int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF for ng in ngrams]

        minhash = []
        for i in range(self.num_perm):
            a_i = self.a[i]
            b_i = self.b[i]
            min_val = min((a_i * h + b_i) % self.prime for h in ngram_hashes)
            minhash.append(min_val)
        return minhash

    def is_near_duplicate(self, text: str) -> bool:
        if len(text) < 100:
            return False

        minhash = self._compute_minhash(text)

        # Check band collision
        is_dup = False
        band_hashes = []

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(minhash[start:end])
            band_h = hash(band_slice)
            band_hashes.append(band_h)

            if band_h in self.band_buckets[band_idx]:
                is_dup = True

        if not is_dup:
            for band_idx, band_h in enumerate(band_hashes):
                self.band_buckets[band_idx].add(band_h)

        return is_dup
