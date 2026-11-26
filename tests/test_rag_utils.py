import unittest

from src.pdf_processing import split_into_chunks, create_chunks
from src.evaluation import precision_at_k, recall_at_k, reciprocal_rank


class TestChunking(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(split_into_chunks(""), [])

    def test_chunks_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = split_into_chunks(text, chunk_size=10, overlap=2)
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertEqual(chunks[1], "ijklmnopqr")
        self.assertEqual("".join([chunks[0], chunks[1]]).count("i"), 2)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_into_chunks("hello", chunk_size=10, overlap=10)

    def test_metadata_is_preserved(self):
        pages = [{"text": "A" * 25, "source": "notes.pdf", "page": 3}]
        chunks = create_chunks(pages, chunk_size=10, overlap=2)
        self.assertTrue(all(item["source"] == "notes.pdf" for item in chunks))
        self.assertTrue(all(item["page"] == 3 for item in chunks))
        self.assertEqual([item["id"] for item in chunks], list(range(len(chunks))))


class TestRetrievalMetrics(unittest.TestCase):
    def test_precision(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3, 4], {2, 4}, 4), 0.5)

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k([1, 2, 3], {2, 4}, 3), 0.5)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank([7, 8, 9], {8}), 0.5)
        self.assertEqual(reciprocal_rank([1, 2], {4}), 0.0)


if __name__ == "__main__":
    unittest.main()
