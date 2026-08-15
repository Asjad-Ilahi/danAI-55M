"""
Unit tests for Tokenizer integration.
"""

import unittest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


class TestTokenizer(unittest.TestCase):

    def test_bpe_tokenizer_special_tokens(self):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask >"],
        )

        tokenizer.train_from_iterator(["Hello world, this is a test."], trainer=trainer)

        self.assertEqual(tokenizer.token_to_id("<pad>"), 0)
        self.assertEqual(tokenizer.token_to_id("<unk>"), 1)
        self.assertIsNotNone(tokenizer.token_to_id("<eos>"))


if __name__ == "__main__":
    unittest.main()
