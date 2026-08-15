"""
Memory-mapped shard dataset for PyTorch training loop per §10.

Reads uint16 binary shards via numpy.memmap without loading entire dataset into RAM.
Supports document-aware causal attention masking per §6.
"""

import glob
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.packing import create_block_diagonal_causal_mask


class ShardDataset(Dataset):
    """
    Dataset over binary uint16 shards mapped via numpy.memmap.
    """

    def __init__(
        self,
        data_dir: str | Path | List[str | Path],
        seq_len: int = 1024,
        pack_with_document_mask: bool = True,
    ):
        if isinstance(data_dir, (str, Path)):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [Path(d) for d in data_dir]
            
        self.seq_len = seq_len
        self.pack_with_document_mask = pack_with_document_mask

        # Find all token shard files across all directories
        self.token_files: List[Path] = []
        for d in self.data_dirs:
            shards = sorted(list(d.glob("shard_*.bin")))
            shards = [f for f in shards if not f.name.endswith("_seg.bin")]
            self.token_files.extend(shards)

        if not self.token_files:
            raise FileNotFoundError(f"No binary shards found in {self.data_dirs}")

        # Load memmaps
        self.shards: List[np.ndarray] = []
        self.segment_shards: List[Optional[np.ndarray]] = []
        self.shard_lengths: List[int] = []

        total_tokens = 0
        for f in self.token_files:
            mmap = np.memmap(str(f), dtype=np.uint16, mode="r")
            self.shards.append(mmap)

            # Check for segment file
            seg_file = f.parent / f"{f.stem}_seg.bin"
            if seg_file.exists():
                seg_mmap = np.memmap(str(seg_file), dtype=np.uint16, mode="r")
                self.segment_shards.append(seg_mmap)
            else:
                self.segment_shards.append(None)

            num_tokens = len(mmap)
            self.shard_lengths.append(num_tokens)
            total_tokens += num_tokens

        self.total_tokens = total_tokens

        # We step sequence by sequence of size seq_len + 1 (since x is seq_len, y is shifted by 1)
        # Sequence length in shard needed for each sample is seq_len + 1 tokens
        self.sample_len = seq_len + 1

        # Pre-compute sample offsets per shard
        self.sample_offsets: List[Tuple[int, int]] = []  # (shard_idx, start_pos)
        for shard_idx, num_tokens in enumerate(self.shard_lengths):
            num_samples = num_tokens // self.sample_len
            for s in range(num_samples):
                self.sample_offsets.append((shard_idx, s * self.sample_len))

    def __len__(self) -> int:
        return len(self.sample_offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, start_pos = self.sample_offsets[idx]
        tokens_mmap = self.shards[shard_idx]
        seg_mmap = self.segment_shards[shard_idx]

        # Extract seq_len + 1 tokens
        chunk = tokens_mmap[start_pos : start_pos + self.sample_len].astype(np.int64)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        item = {
            "x": x,
            "y": y,
        }

        if self.pack_with_document_mask and seg_mmap is not None:
            seg_chunk = seg_mmap[start_pos : start_pos + self.seq_len].astype(np.int64)
            seg_tensor = torch.tensor(seg_chunk, dtype=torch.long)
            # Create (1, seq_len, seq_len) block diagonal mask
            mask = create_block_diagonal_causal_mask(seg_tensor)  # (1, 1, seq_len, seq_len)
            item["attn_mask"] = mask.squeeze(0)  # (1, seq_len, seq_len)

        return item
