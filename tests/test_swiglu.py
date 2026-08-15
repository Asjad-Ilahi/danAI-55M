"""
Unit tests for SwiGLU module.
"""

import torch
import unittest
from src.model.swiglu import SwiGLU


class TestSwiGLU(unittest.TestCase):

    def test_swiglu_shape(self):
        mlp = SwiGLU(hidden_size=512, intermediate_size=1536)
        x = torch.randn(2, 16, 512)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_swiglu_parameter_count(self):
        h = 512
        inter = 1536
        mlp = SwiGLU(hidden_size=h, intermediate_size=inter, bias=False)
        expected_params = 3 * h * inter  # gate + up + down
        actual_params = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(actual_params, expected_params)


if __name__ == "__main__":
    unittest.main()
