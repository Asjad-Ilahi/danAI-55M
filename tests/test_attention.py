"""
Unit tests for GQA Attention and Document Boundary Masking per §18 & §6.
"""

import torch
import unittest
from src.model.attention import GroupedQueryAttention
from src.data.packing import create_block_diagonal_causal_mask


class TestAttention(unittest.TestCase):

    def test_gqa_shapes(self):
        attn = GroupedQueryAttention(
            hidden_size=512,
            num_query_heads=8,
            num_kv_heads=2,
            max_seq_len=1024,
        )
        x = torch.randn(2, 16, 512)
        out, cache = attn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(cache)

    def test_causal_mask_gradient_isolation(self):
        """Prove token i cannot affect logits/representations of tokens before i."""
        attn = GroupedQueryAttention(hidden_size=64, num_query_heads=4, num_kv_heads=2)
        attn.eval()

        # Batch=1, seq_len=4
        x = torch.randn(1, 4, 64, requires_grad=True)
        out, _ = attn(x)

        # Gradient of token 0's output w.r.t input token 3 must be exactly 0
        loss = out[0, 0, :].sum()  # Loss dependent ONLY on token 0 output
        loss.backward()

        grad_token_3 = x.grad[0, 3, :]
        self.assertTrue(torch.all(grad_token_3 == 0.0), "Token 3 leaked gradient back to Token 0 output!")

    def test_document_boundary_mask_correctness(self):
        """
        REQUIRED TEST (§18):
        Prove tokens in document B never receive gradient signal from / attend to document A
        when packed together in the same sequence.
        """
        attn = GroupedQueryAttention(hidden_size=64, num_query_heads=4, num_kv_heads=2)
        attn.eval()

        # Sequence of length 6 packed with 2 documents:
        # Doc 0: positions 0, 1, 2
        # Doc 1: positions 3, 4, 5
        segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        doc_mask = create_block_diagonal_causal_mask(segment_ids)  # (1, 1, 6, 6)

        x = torch.randn(1, 6, 64, requires_grad=True)
        out, _ = attn(x, attention_mask=doc_mask)

        # Token at position 3 is the first token of Document 1.
        # It must NOT attend to positions 0, 1, 2 (Document 0).
        loss_doc1_start = out[0, 3, :].sum()
        loss_doc1_start.backward()

        # Check gradients for Document 0 tokens (pos 0, 1, 2)
        grad_doc0 = x.grad[0, :3, :]
        self.assertTrue(
            torch.all(grad_doc0 == 0.0),
            "Document 1 start token received gradient signal from Document 0 tokens! Document mask is leaking attention!"
        )


if __name__ == "__main__":
    unittest.main()
