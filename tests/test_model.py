"""
Unit tests for full CausalLM model and parameter counter matching per §4.
"""

import torch
import unittest
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.model.parameter_count import compute_parameter_count, verify_against_model


class TestModel(unittest.TestCase):

    def test_parameter_count_exact_match(self):
        """Analytical param formula must match instantiated model numel() exactly."""
        config = Config.from_yaml("configs/model.yaml")
        model = CausalLM(config)

        mc = config.model
        counts = compute_parameter_count(
            vocab_size=mc.vocab_size,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            num_query_heads=mc.num_query_heads,
            num_kv_heads=mc.num_kv_heads,
            intermediate_size=mc.get("intermediate_size", "auto"),
            tie_embeddings=mc.get("tie_embeddings", True),
            use_bias=mc.get("use_bias", False),
        )

        actual_total = sum(p.numel() for p in model.parameters())
        self.assertEqual(actual_total, counts["total"])
        self.assertTrue(verify_against_model(model, counts["total"]))

    def test_model_forward_loss(self):
        config = Config.from_yaml("configs/debug.yaml")
        model = CausalLM(config)

        x = torch.randint(0, config.model.vocab_size, (2, 32))
        y = torch.randint(0, config.model.vocab_size, (2, 32))

        logits, loss, _ = model(x, targets=y)

        self.assertEqual(logits.shape, (2, 32, config.model.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
