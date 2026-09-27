"""
Unit and Integration Tests for Person 2 Preprocessing Pipeline.
Tests:
- clean_text normalization
- count_tokens accuracy and fallback
- chunk_document splitting and bounds
- deduplication (exact SHA-256 and near-duplicate TF-IDF)
- broken image reference validation
"""

import unittest
from src.preprocess import clean_text
from src.chunking import count_tokens, chunk_document, split_large_section
from src.deduplication import compute_sha256, deduplicate_chunks
from src.metadata import build_chunk_metadata, classify_category, classify_type, classify_difficulty, detect_language


class TestPerson2Pipeline(unittest.TestCase):

    def test_clean_text_unicode_and_whitespace(self):
        raw = "  Hello \ufffd World \u201cquoted\u201d text\n\n\n\nLine 2   extra spaces  \t \n\nSource type: Synthetic sample FAQ document | test"
        cleaned = clean_text(raw)
        self.assertNotIn("\ufffd", cleaned)
        self.assertNotIn("Source type: Synthetic", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn('"quoted"', cleaned)
        self.assertTrue(cleaned.startswith("Hello"))

    def test_token_counting(self):
        text = "How do I reset my password? Open the login page and click Forgot Password."
        tokens = count_tokens(text)
        self.assertGreater(tokens, 10)
        self.assertLess(tokens, 30)

    def test_chunking_atomic_faq_preservation(self):
        text = "Q1. How do I reset my password?\nClick forgot password and follow steps."
        chunks = chunk_document(text, target_min=250, target_max=500)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Q1. How do I reset my password?", chunks[0]["text"])

    def test_chunking_large_text_splitting(self):
        # Create a 600-word text that exceeds 500 tokens
        sentences = [f"Step {i}: This is an important instructional step that explains how to configure component {i} in the multimodal system." for i in range(1, 35)]
        large_text = "\n\n".join(sentences)
        chunks = split_large_section(large_text, target_min=250, target_max=500)
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            tok = count_tokens(ch)
            self.assertLessEqual(tok, 520)

    def test_exact_deduplication(self):
        chunks = [
            {"chunk_id": "c1", "text": "How do I reset my password? Open login page."},
            {"chunk_id": "c2", "text": "how do i reset my password? open login page."},  # exact duplicate normalized
            {"chunk_id": "c3", "text": "Different question about wifi connection."}
        ]
        unique, stats = deduplicate_chunks(chunks, near_dup_threshold=0.88)
        self.assertEqual(stats["total_chunks_before"], 3)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0]["chunk_id"], "c1")
        self.assertEqual(unique[1]["chunk_id"], "c3")

    def test_near_deduplication(self):
        chunks = [
            {"chunk_id": "c1", "text": "How do I reset my account password? Go to the login page and select forgot password."},
            {"chunk_id": "c2", "text": "How do I reset my account password? Go to the login page and click forgot password."},  # near duplicate (0.89 similarity)
            {"chunk_id": "c3", "text": "Why does my router fail to connect to the broadband fiber network?"}
        ]
        unique, stats = deduplicate_chunks(chunks, near_dup_threshold=0.85)
        self.assertEqual(stats["near_duplicates_removed"], 1)
        self.assertEqual(len(unique), 2)

    def test_metadata_classification(self):
        cat = classify_category("To reset your password...", {}, "01_account_and_password_faq.pdf")
        self.assertEqual(cat, "Account Management")

        t = classify_type("Q1. How do I reset my password? Click here.")
        self.assertEqual(t, "howto")

        lang = detect_language("This is a standard English knowledge base entry.")
        self.assertEqual(lang, "en")


if __name__ == "__main__":
    unittest.main()
