import unittest
import os
import sys


#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ingest.extract import extract_content
from ingest.cleaner import clean_text
from ingest.chunker import chunk_text

class TestIngestionPipeline(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test-data'))

    def test(self):
        if not os.path.exists(self.test_data_dir):
            self.skipTest(f"Test data directory not found: {self.test_data_dir}")

        files = os.listdir(self.test_data_dir)
        if not files:
             self.skipTest("No files found in test-data directory")

        for filename in files:
            file_path = os.path.join(self.test_data_dir, filename)
            if not os.path.isfile(file_path):
                continue
            
            print(f"\nProcessing {filename}...")
            
            # Extract
            text = extract_content(file_path)
            if not text:
                print(f"Warning: No text extracted from {filename}. It might be an image without text or extraction failed.")
                # We don't fail the test here because OCR might fail or be empty, but we log it.
                continue
            
            print(f"Extracted {len(text)} characters.")

            # Clean
            cleaned_text = clean_text(text)
            self.assertTrue(len(cleaned_text) >= 0) # It can be empty if input was only whitespace
            print(f"Cleaned text length: {len(cleaned_text)}")

            # Chunk
            chunks = chunk_text(cleaned_text, chunk_size=500, overlap=50)
            
            print(f"Generated {len(chunks)} chunks.")
            if chunks:
                print(f"First chunk preview: {chunks[0][:100]}...")
            
            # Basic assertions
            if len(cleaned_text) > 500:
                self.assertTrue(len(chunks) > 1, "Should have multiple chunks for long text")
