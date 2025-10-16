import unittest
import sys
import os
from unittest.mock import Mock, MagicMock
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunking_strategies import (
    chunk_fixed_size,
    chunk_by_sentence,
    chunk_recursive,
    chunk_semantic
)


class TestChunkingStrategies(unittest.TestCase):
    """Comprehensive unit tests for all chunking strategies."""

    def setUp(self):
        """Set up test data for all tests."""
        # Simple predictable text for exact testing
        self.simple_text = "First sentence. Second sentence. Third sentence."

        # Predictable text for overlap testing
        self.overlap_test_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Text with clear paragraph structure
        self.paragraph_text = "First paragraph first sentence. First paragraph second sentence.\n\nSecond paragraph first sentence. Second paragraph second sentence.\n\nThird paragraph only sentence."

        # Very short text
        self.short_text = "Short."

        # Empty text
        self.empty_text = ""

class TestFixedSizeChunking(TestChunkingStrategies):
    """Tests for fixed size chunking strategy."""

    def test_fixed_size_exact_chunks(self):
        """Test exact chunk content with fixed size chunking."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_fixed_size(text, chunk_size=10, chunk_overlap=3)

        expected_chunks = [
            "ABCDEFGHIJ",  # 0-10
            "HIJKLMNOPQ",  # 7-17 (overlap of 3)
            "OPQRSTUVWX",  # 14-24
            "VWXYZ"  # 21-26
        ]

        self.assertEqual(chunks, expected_chunks)

    def test_fixed_size_no_overlap(self):
        """Test fixed size chunking with no overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_fixed_size(text, chunk_size=5, chunk_overlap=0)

        expected_chunks = [
            "ABCDE",
            "FGHIJ",
            "KLMNO",
            "PQRST",
            "UVWXY",
            "Z"
        ]

        self.assertEqual(chunks, expected_chunks)

    def test_fixed_size_larger_than_text(self):
        """Test when chunk size is larger than text."""
        chunks = chunk_fixed_size(self.short_text, chunk_size=100, chunk_overlap=10)

        expected_chunks = ["Short."]
        self.assertEqual(chunks, expected_chunks)

    def test_fixed_size_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_fixed_size(self.empty_text, chunk_size=100, chunk_overlap=10)
        self.assertEqual(chunks, [])

    def test_fixed_size_sentence_text(self):
        """Test fixed size chunking on sentence text."""
        chunks = chunk_fixed_size(self.simple_text, chunk_size=20, chunk_overlap=5)

        expected_chunks = ['First sentence. Seco', ' Second sentence. Th', 'e. Third sentence.', 'ce.']
        self.assertEqual(chunks, expected_chunks)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

        # Verify first chunk starts with beginning of text
        self.assertTrue(chunks[0].startswith("First"))


class TestSentenceBasedChunking(TestChunkingStrategies):
    """Tests for sentence-based chunking strategy."""

    def test_sentence_chunking_exact_content(self):
        """Test exact content of sentence-based chunks."""
        chunks = chunk_by_sentence(self.simple_text, sentences_per_chunk=1)

        expected_chunks = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]

        self.assertEqual(chunks, expected_chunks)

    def test_sentence_chunking_two_per_chunk(self):
        """Test sentence chunking with two sentences per chunk."""
        chunks = chunk_by_sentence(self.simple_text, sentences_per_chunk=2)

        expected_chunks = [
            "First sentence. Second sentence.",
            "Third sentence."
        ]

        self.assertEqual(chunks, expected_chunks)

    def test_sentence_chunking_more_than_available(self):
        """Test when requesting more sentences per chunk than available."""
        chunks = chunk_by_sentence(self.simple_text, sentences_per_chunk=5)

        expected_chunks = ["First sentence. Second sentence. Third sentence."]
        self.assertEqual(chunks, expected_chunks)

    def test_sentence_chunking_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_by_sentence(self.empty_text, sentences_per_chunk=3)
        self.assertEqual(chunks, [])

    def test_sentence_chunking_single_sentence(self):
        """Test chunking single sentence."""
        chunks = chunk_by_sentence(self.short_text, sentences_per_chunk=1)

        expected_chunks = ["Short."]
        self.assertEqual(chunks, expected_chunks)

# Todo add other strategies here...

class TestSentenceBasedChunking(TestChunkingStrategies):
    """Tests for sentence-based chunking strategy."""

    def test_sentence_chunking_exact_content(self):
        """Test exact content of sentence-based chunks."""
        chunks = chunk_by_sentence(self.simple_text, sentences_per_chunk=1)

        expected_chunks = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]

        self.assertEqual(chunks, expected_chunks)

    def test_sentence_chunking_two_per_chunk(self):
        """Test sentence chunking with two sentences per chunk."""
        chunks = chunk_by_sentence(self.simple_text, sentences_per_chunk=2)

        expected_chunks = [
            "First sentence. Second sentence.",
            "Third sentence."
        ]

        self.assertEqual(chunks, expected_chunks)

    def test_sentence_chunking_more_than_available(self):
        """Test when requesting more sentences per chunk than available."""
        chunks = chunk_by_sentence(self.simple_text, sentences_per_chunk=5)

        expected_chunks = ["First sentence. Second sentence. Third sentence."]
        self.assertEqual(chunks, expected_chunks)

    def test_sentence_chunking_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_by_sentence(self.empty_text, sentences_per_chunk=3)
        self.assertEqual(chunks, [])

    def test_sentence_chunking_single_sentence(self):
        """Test chunking single sentence."""
        chunks = chunk_by_sentence(self.short_text, sentences_per_chunk=1)

        expected_chunks = ["Short."]
        self.assertEqual(chunks, expected_chunks)

class TestRecursiveChunking(TestChunkingStrategies):
    """Tests for recursive character splitting chunking strategy."""

    def test_recursive_chunking_paragraphs(self):
        """Test chunking with paragraph separators."""
        chunks = chunk_recursive(self.paragraph_text, chunk_size=50, chunk_overlap=5)
        # Should split at paragraph boundaries first, then sentences if needed
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertTrue(any("First paragraph" in chunk for chunk in chunks))
        self.assertTrue(any("Second paragraph" in chunk for chunk in chunks))
        self.assertTrue(any("Third paragraph" in chunk for chunk in chunks))

    def test_recursive_chunking_overlap(self):
        """Test overlap is present in recursive chunking."""
        text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        chunks = chunk_recursive(text, chunk_size=10, chunk_overlap=3)
        # Overlap: last 3 chars of previous chunk should be at start of next
        for i in range(1, len(chunks)):
            self.assertTrue(chunks[i].startswith(chunks[i-1][-3:]))

    def test_recursive_chunking_short_text(self):
        """Test chunking with very short text."""
        chunks = chunk_recursive(self.short_text, chunk_size=100, chunk_overlap=10)
        self.assertEqual(chunks, ["Short."])

    def test_recursive_chunking_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_recursive(self.empty_text, chunk_size=100, chunk_overlap=10)
        self.assertEqual(chunks, [])

    def test_recursive_chunking_exact_size(self):
        """Test chunking when chunk size matches text length."""
        text = "1234567890"
        chunks = chunk_recursive(text, chunk_size=10, chunk_overlap=2)
        self.assertEqual(chunks, ["1234567890"])

class TestSemanticChunking(TestChunkingStrategies):
    """Tests for semantic chunking strategy."""

    def setUp(self):
        super().setUp()


        # Mock vectorizer for semantic chunking tests
        self.mock_vectorizer = Mock()
        # Create embeddings that will result in predictable similarity patterns
        self.mock_vectorizer.embed_documents.return_value = [
            np.array([1.0, 0.0, 0.0]),  # Similar to next
            np.array([0.9, 0.1, 0.0]),  # Similar to previous
            np.array([0.0, 0.0, 1.0]),  # Different - should cause split
            np.array([0.0, 0.1, 0.9])  # Similar to previous
        ]

        # Patch sent_tokenize to return predictable sentences
        patcher = unittest.mock.patch('chunking_strategies.sent_tokenize', return_value=["A.", "B.", "C.", "D."])
        self.mock_sent_tokenize = patcher.start()
        self.addCleanup(patcher.stop)

    def test_semantic_chunking_basic_split(self):
        """Unit test: verify function calls and logic for chunk_semantic."""
        text = "A. B. C. D."
        chunks = chunk_semantic(text, self.mock_vectorizer, similarity_threshold=0.5)

        self.mock_sent_tokenize.assert_called_once_with(text)
        self.mock_vectorizer.embed_documents.assert_called_once_with(["A.", "B.", "C.", "D."])
        self.assertEqual(chunks, ["A. B.", "C. D."])


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)

    # Run the tests
    unittest.main(verbosity=2)
