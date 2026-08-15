"""
Unit tests for text generation and KV cache equivalence.
"""

import unittest
import torch
from src.utils.config import Config
from src.model.gpt import CausalLM


class TestGeneration(unittest.TestCase):

    def test_kv_cache_equivalence(self):
        """KV cache output must match non-cache forward pass output exactly."""
        torch.manual_seed(42)
        config = Config.from_yaml("configs/debug.yaml")
        model = CausalLM(config)
        model.eval()

        input_ids = torch.tensor([[10, 20, 30, 40, 50]])

        # Non-cache forward pass
        with torch.no_grad():
            full_logits, _, _ = model(input_ids, kv_caches=None)

        # Cache forward pass step by step
        with torch.no_grad():
            kv_caches = None
            step_logits_list = []
            for t in range(input_ids.shape[1]):
                token_t = input_ids[:, t:t+1]
                step_logits, _, kv_caches = model(token_t, kv_caches=kv_caches, use_cache=True)
                step_logits_list.append(step_logits[:, -1, :])

            cached_full_logits = torch.stack(step_logits_list, dim=1)

        self.assertTrue(torch.allclose(full_logits, cached_full_logits, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
