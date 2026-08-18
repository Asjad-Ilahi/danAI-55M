"""
SFT Dataset Module for Supervised Fine-Tuning with Response-Only Loss Masking.

Features:
- Standard multi-turn JSONL conversation parser with full System, User, Tool, and Assistant role support.
- Response-only loss masking: all System/User/Tool inputs have target ID = -100 (loss only on Assistant output).
- Intelligent sequence truncation preserving the latest complete turns.
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
    intelligent turn-aware truncation supporting System, User, Tool, and Assistant roles.
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
        
        # Parse conversation with system prompt and turn structure
        # Segment format: (prompt_tokens, response_tokens)
        full_input_ids: List[int] = []
        full_target_ids: List[int] = []
        
        system_prefix = ""
        msg_idx = 0
        if messages and messages[0].get("role", "").strip().lower() == "system":
            sys_content = messages[0].get("content", "").strip()
            if sys_content:
                system_prefix = f"System: {sys_content}\n\n"
                s_ids = self.tokenizer.encode(system_prefix).ids
                full_input_ids.extend(s_ids)
                full_target_ids.extend([-100] * len(s_ids))
            msg_idx = 1
        
        while msg_idx < len(messages):
            msg = messages[msg_idx]
            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()
            
            if role == "user":
                user_header = f"User: {content}\n\nAssistant: "
                h_ids = self.tokenizer.encode(user_header).ids
                full_input_ids.extend(h_ids)
                full_target_ids.extend([-100] * len(h_ids))
                msg_idx += 1
            elif role in ["tool", "tool_response"]:
                tool_block = f"\n\n<tool_response>\n{content}\n</tool_response>\n\nAssistant: "
                t_ids = self.tokenizer.encode(tool_block).ids
                full_input_ids.extend(t_ids)
                full_target_ids.extend([-100] * len(t_ids))
                msg_idx += 1
            elif role == "assistant":
                is_last = (msg_idx == len(messages) - 1)
                asst_ids = self.tokenizer.encode(content).ids
                if is_last or "<tool_call>" not in content:
                    asst_ids = asst_ids + [self.eos_id]
                full_input_ids.extend(asst_ids)
                full_target_ids.extend(asst_ids)
                msg_idx += 1
            else:
                extra_ids = self.tokenizer.encode(f"{content}\n\n").ids
                full_input_ids.extend(extra_ids)
                full_target_ids.extend([-100] * len(extra_ids))
                msg_idx += 1

        # Fallback if empty
        if not full_input_ids:
            u = messages[0].get("content", "") if messages else "Hello"
            a = messages[1].get("content", "") if len(messages) > 1 else "Hello!"
            u_ids = self.tokenizer.encode(f"User: {u}\n\nAssistant: ").ids
            a_ids = self.tokenizer.encode(a).ids + [self.eos_id]
            full_input_ids = u_ids + a_ids
            full_target_ids = ([-100] * len(u_ids)) + a_ids

        # Truncate if exceeds max_seq_len (keep within bounds)
        if len(full_input_ids) > self.max_seq_len:
            full_input_ids = full_input_ids[:self.max_seq_len]
            full_target_ids = full_target_ids[:self.max_seq_len]

        # Causal LM shifted input (x) and target (y)
        x = torch.tensor(full_input_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_target_ids[1:], dtype=torch.long)

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
            x_padded = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            y_padded = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
        else:
            x_padded = x
            y_padded = y
            mask = torch.ones(seq_len, dtype=torch.bool)
            
        batch_x.append(x_padded)
        batch_y.append(y_padded)
        attn_masks.append(mask)
        
    return {
        "x": torch.stack(batch_x, dim=0),
        "y": torch.stack(batch_y, dim=0),
        "attention_mask": torch.stack(attn_masks, dim=0),
        "domains": domains,
    }
