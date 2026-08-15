"""
Unit tests for packing and dataset indexing.
"""

import unittest
from src.data.packing import pack_documents, create_block_diagonal_causal_mask
import torch


class TestDataset(unittest.TestCase):

    def test_document_packing(self):
        docs = [
            [10, 11, 12],
            [20, 21],
            [30, 31, 32, 33],
        ]
        eos_id = 99
        seq_len = 5

        packed = list(pack_documents(docs, max_seq_len=seq_len, eos_token_id=eos_id))
        self.assertGreater(len(packed), 0)

        first_seq, first_segs = packed[0]
        self.assertEqual(len(first_seq), seq_len)
        self.assertEqual(len(first_segs), seq_len)

    def test_block_diagonal_causal_mask_shape(self):
        segs = torch.tensor([[0, 0, 1, 1, 1]])
        mask = create_block_diagonal_causal_mask(segs)
        self.assertEqual(mask.shape, (1, 1, 5, 5))


if __name__ == "__main__":
    unittest.main()
