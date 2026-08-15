"""
Unit tests for RMSNorm module.
"""

import torch
import unittest
from src.model.rmsnorm import RMSNorm


class TestRMSNorm(unittest.TestCase):

    def test_rmsnorm_output_shape(self):
        norm = RMSNorm(hidden_size=512)
        x = torch.randn(2, 16, 512)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_rmsnorm_math(self):
        hidden_size = 4
        norm = RMSNorm(hidden_size=hidden_size, eps=1e-5)
        norm.weight.data.fill_(1.0)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        # RMS = sqrt(mean(1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.73861278
        expected_rms = torch.sqrt(torch.mean(x**2) + 1e-5)
        expected_out = x / expected_rms

        out = norm(x)
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-5))

    def test_dtype_preservation(self):
        norm = RMSNorm(hidden_size=128)
        x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        out = norm(x)
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
