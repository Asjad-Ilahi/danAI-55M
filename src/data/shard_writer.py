"""
Binary uint16 token shard writer per §10.

Token IDs (for vocab size <= 65536) fit in unsigned 16-bit integers (uint16).
Writes token arrays to data/processed/{split}/shard_XXXXX.bin with metadata JSON.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class ShardWriter:
    """Writes uint16 token shards to disk."""

    def __init__(
        self,
        output_dir: str | Path,
        split: str = "train",
        max_tokens_per_shard: int = 10_000_000,
        vocab_size: int = 16384,
    ):
        self.output_dir = Path(output_dir) / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.max_tokens_per_shard = max_tokens_per_shard
        self.vocab_size = vocab_size

        if vocab_size > 65536:
            raise ValueError(f"Vocab size {vocab_size} exceeds uint16 maximum (65536).")

        self.current_shard_idx = 0
        self.current_buffer: List[int] = []
        self.current_segment_buffer: List[int] = []
        self.total_tokens_written = 0

    def add_packed_sequence(self, tokens: List[int], segment_ids: List[int]) -> None:
        """Add a packed sequence to buffer and flush to shard if buffer is full."""
        self.current_buffer.extend(tokens)
        self.current_segment_buffer.extend(segment_ids)

        if len(self.current_buffer) >= self.max_tokens_per_shard:
            self.flush()

    def flush(self) -> None:
        """Write buffer contents to binary shard file."""
        if not self.current_buffer:
            return

        shard_filename = f"shard_{self.current_shard_idx:05d}.bin"
        seg_filename = f"shard_{self.current_shard_idx:05d}_seg.bin"
        shard_path = self.output_dir / shard_filename
        seg_path = self.output_dir / seg_filename

        # Convert to numpy uint16
        tokens_np = np.array(self.current_buffer, dtype=np.uint16)
        segments_np = np.array(self.current_segment_buffer, dtype=np.uint16)

        # Save binary memory-map compatible arrays
        tokens_np.tofile(str(shard_path))
        segments_np.tofile(str(seg_path))

        meta = {
            "shard_idx": self.current_shard_idx,
            "num_tokens": len(tokens_np),
            "dtype": "uint16",
            "token_file": shard_filename,
            "segment_file": seg_filename,
        }
        with open(self.output_dir / f"shard_{self.current_shard_idx:05d}.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.total_tokens_written += len(tokens_np)
        print(f"[{self.split}] Wrote {len(tokens_np):,} tokens to {shard_filename} (total: {self.total_tokens_written:,})")

        self.current_shard_idx += 1
        self.current_buffer.clear()
        self.current_segment_buffer.clear()

    def close(self) -> Dict[str, Any]:
        """Flush remaining buffer and return summary statistics."""
        self.flush()
        return {
            "split": self.split,
            "total_shards": self.current_shard_idx,
            "total_tokens": self.total_tokens_written,
            "output_dir": str(self.output_dir),
        }
