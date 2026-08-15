"""
Text generation engine per §19 & §42.

Supports:
- KV Cache acceleration for sequential decoding
- Greedy decoding, Temperature scaling
- Top-k sampling, Top-p (nucleus) sampling
- Context length truncation
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any


class TextGenerator:
    """Text generation engine supporting KV cache, nucleus sampling, and repetition penalty."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> str:
        """
        Generate text from prompt.
        """
        if eos_token_id is None:
            eos_token_id = self.tokenizer.token_to_id("<eos>")

        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)

        seq_len = input_ids.shape[1]
        max_context = getattr(self.model, "max_seq_len", 1024)

        if seq_len >= max_context:
            input_ids = input_ids[:, -max_context + max_new_tokens:]

        generated_ids = list(encoded.ids)
        kv_caches = None

        if use_kv_cache:
            # Initial forward pass over full prompt to build KV cache
            logits, _, kv_caches = self.model(input_ids, kv_caches=None, use_cache=True)
            next_token_logits = logits[:, -1, :]

        for step in range(max_new_tokens):
            if not use_kv_cache:
                curr_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                if curr_ids.shape[1] > max_context:
                    curr_ids = curr_ids[:, -max_context:]
                logits, _, _ = self.model(curr_ids, kv_caches=None, use_cache=False)
                next_token_logits = logits[:, -1, :]
            else:
                if step > 0:
                    curr_input = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=self.device)
                    logits, _, kv_caches = self.model(curr_input, kv_caches=kv_caches, use_cache=True)
                    next_token_logits = logits[:, -1, :]

            # Apply repetition penalty per CTRL / Hugging Face formula
            if repetition_penalty != 1.0 and len(generated_ids) > 0:
                unique_tokens = list(set(generated_ids))
                token_logits = next_token_logits[0, unique_tokens]
                # If logit > 0: divide by penalty (lower probability)
                # If logit < 0: multiply by penalty (lower probability)
                penalized = torch.where(
                    token_logits > 0,
                    token_logits / repetition_penalty,
                    token_logits * repetition_penalty
                )
                next_token_logits[0, unique_tokens] = penalized

            # Apply temperature scaling
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
            else:
                scaled_logits = next_token_logits

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = -float("Inf")

            # Sample token
            if temperature > 0:
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(scaled_logits, dim=-1).item()

            generated_ids.append(next_token)

            if next_token == eos_token_id:
                break

        # Decode generated IDs to text
        return self.tokenizer.decode(generated_ids)
