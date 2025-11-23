import unittest
from ingest.chunker import chunk_text

class TestChunker(unittest.TestCase):
    def test_simple_chunking(self):
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        self.assertTrue(len(chunks) > 1)
        # Check if chunks respect the size
        self.assertTrue(all(len(c) <= 50 for c in chunks))

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text(self):
        text = "Short text"
        self.assertEqual(chunk_text(text, chunk_size=100), ["Short text"])

    def test_overlap(self):
        text = "1234567890"
        # Chunk size 5, overlap 2
        # Expected: "12345", "45678", "7890" (roughly)
        chunks = chunk_text(text, chunk_size=5, overlap=2)
        self.assertTrue(len(chunks) >= 2)
