import unittest

from src.common.dataset import TranslationDataset, collate_translation_batch
from src.common.tokenizer import load_tokenizer


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer("data/vocab/tokenizer_config.json")
        cls.dataset = TranslationDataset("data/processed/test.jsonl", tokenizer=cls.tokenizer)

    def test_length(self):
        self.assertGreater(len(self.dataset), 0)

    def test_item_structure(self):
        sample = self.dataset[0]
        self.assertIn("src_ids", sample)
        self.assertIn("tgt_ids", sample)
        self.assertIsInstance(sample["src_ids"], list)
        self.assertGreater(len(sample["src_ids"]), 0)

    def test_collate(self):
        batch = [self.dataset[i] for i in range(4)]
        collated = collate_translation_batch(batch, self.tokenizer)
        self.assertEqual(collated["src"].shape[0], 4)
        self.assertEqual(collated["tgt"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()