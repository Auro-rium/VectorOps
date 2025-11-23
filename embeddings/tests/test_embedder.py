import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from embeddings.embedder import Embedder

class TestEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize once to save time on model loading
        print("Initializing Embedder model (this may take a moment)...")
        cls.embedder = Embedder()

    def test_embed_text(self):
        text = "This is a test sentence."
        embedding = self.embedder.embed_text(text)
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0)
        # all-MiniLM-L6-v2 has 384 dimensions
        self.assertEqual(len(embedding), 384)

    def test_embed_batch(self):
        texts = ["Sentence one.", "Sentence two."]
        embeddings = self.embedder.embed_batch(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 384)
        self.assertEqual(len(embeddings[1]), 384)

if __name__ == '__main__':
    unittest.main()
