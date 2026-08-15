"""
Unit tests for RoPE module.
"""

import torch
import unittest
from src.model.rope import RotaryEmbedding


class TestRoPE(unittest.TestCase):

    def test_rope_rotation_shape(self):
        head_dim = 64
        rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=512)
        q = torch.randn(1, 8, 32, head_dim)
        rotated_q = rope(q)
        self.assertEqual(rotated_q.shape, q.shape)

    def test_relative_position_property(self):
        head_dim = 16
        rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=100)

        # Vector q at pos m and vector k at pos n
        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        pos_m = torch.tensor([[10]])
        pos_n = torch.tensor([[5]])

        q_rot = rope(q, position_ids=pos_m)
        k_rot = rope(k, position_ids=pos_n)

        dot_product_1 = torch.sum(q_rot * k_rot)

        # Shift both positions by delta (e.g. +20)
        pos_m_shifted = torch.tensor([[30]])
        pos_n_shifted = torch.tensor([[25]])

        q_rot_shifted = rope(q, position_ids=pos_m_shifted)
        k_rot_shifted = rope(k, position_ids=pos_n_shifted)

        dot_product_2 = torch.sum(q_rot_shifted * k_rot_shifted)

        # Dot product depends ONLY on relative distance (m - n = 5)
        self.assertTrue(torch.allclose(dot_product_1, dot_product_2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
