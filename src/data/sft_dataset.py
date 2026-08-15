"""
SFT Dataset Module for Supervised Fine-Tuning with Response-Only Loss Masking.

Features:
- Standard OpenAI/HuggingFace multi-turn JSONL conversation parser.
- Intelligent sequence truncation preserving the latest complete user/assistant turn.
- Response-only loss masking: all System/User prompt tokens have target ID = -100.
- Per-sample domain tracking for stratified multi-domain validation loss reporting.
- Dynamic batch padding with attention masking for high throughput.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset with response-only loss masking and
    intelligent turn-aware truncation.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []
        self.total_tokens: int = 0
        
        # Determine EOS token ID
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            eos_id = tokenizer.token_to_id("</s>")
        self.eos_id = eos_id if eos_id is not None else 0
        
        self._load_dataset()

    def _load_dataset(self):
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"SFT dataset file not found: {self.jsonl_path}")
        
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data and len(data["messages"]) >= 2:
                        self.samples.append(data)
                        self.total_tokens += data.get("provenance", {}).get("num_tokens", 0)
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        messages = sample["messages"]
        domain = sample.get("provenance", {}).get("domain", "general")
        
        # 1. Parse into structured user-assistant pairs
        pairs: List[Tuple[str, str]] = []
        current_user = None
        for msg in messages:
            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()
            if role in ["user", "system"]:
                current_user = content
            elif role == "assistant" and current_user is not None:
                pairs.append((current_user, content))
                current_user = None

        if not pairs:
            # Fallback if roles were irregular
            first_user = messages[0].get("content", "")
            first_asst = messages[1].get("content", "")
            pairs = [(first_user, first_asst)]

        # 2. Encode pairs from newest to oldest (Intelligent Truncation)
        encoded_pairs = []
        for u, a in reversed(pairs):
            u_enc = self.tokenizer.encode(f"User: {u}\n\nAssistant: ").ids
            a_enc = self.tokenizer.encode(f"{a}").ids + [self.eos_id]
            encoded_pairs.append((u_enc, a_enc))

        final_input_ids: List[int] = []
        final_target_ids: List[int] = []

        for u_enc, a_enc in encoded_pairs:
            pair_len = len(u_enc) + len(a_enc)
            if len(final_input_ids) + pair_len <= self.max_seq_len:
                # Prepend older complete turn before newer turn
                final_input_ids = u_enc + a_enc + final_input_ids
                final_target_ids = ([-100] * len(u_enc)) + a_enc + final_target_ids
            else:
                # If no pair fits yet (the latest single turn alone exceeds max_seq_len)
                if not final_input_ids:
                    avail_for_user = self.max_seq_len - len(a_enc)
                    if avail_for_user >= 20:
                        u_enc_trunc = u_enc[:avail_for_user]
                        final_input_ids = u_enc_trunc + a_enc
                        final_target_ids = ([-100] * len(u_enc_trunc)) + a_enc
                    else:
                        avail_for_asst = max(10, self.max_seq_len - len(u_enc) - 1)
                        a_enc_trunc = a_enc[:avail_for_asst] + [self.eos_id]
                        final_input_ids = u_enc + a_enc_trunc
                        final_target_ids = ([-100] * len(u_enc)) + a_enc_trunc
                break

        # Prepare causal LM shifted x and y
        # x: input_ids[:-1], y: target_ids[1:]
        x = torch.tensor(final_input_ids[:-1], dtype=torch.long)
        y = torch.tensor(final_target_ids[1:], dtype=torch.long)

        return {
            "x": x,
            "y": y,
            "domain": domain,
            "seq_len": torch.tensor(len(x), dtype=torch.long),
        }


def sft_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    """
    Collate function with dynamic padding and attention mask generation.
    """
    max_len = max(item["x"].size(0) for item in batch)
    
    batch_x = []
    batch_y = []
    attn_masks = []
    domains = []
    
    for item in batch:
        x = item["x"]
        y = item["y"]
        domains.append(item["domain"])
        seq_len = x.size(0)
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            padded_x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            padded_y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded_x = x
            padded_y = y
            mask = torch.ones(seq_len, dtype=torch.bool)
        
        batch_x.append(padded_x)
        batch_y.append(padded_y)
        attn_masks.append(mask)
    
    return {
        "x": torch.stack(batch_x, dim=0),
        "y": torch.stack(batch_y, dim=0),
        "attention_mask": torch.stack(attn_masks, dim=0),
        "domains": domains,
    }
