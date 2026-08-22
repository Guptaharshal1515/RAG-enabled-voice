import unittest
from src.chunking.token_counter import estimate_tokens
from src.chunking.sentence_chunker import sentence_chunk, split_sentences
from src.chunking.recursive_chunker import recursive_chunk
from src.chunking.adaptive_chunker import AdaptiveChunker


class TestAdaptiveChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = AdaptiveChunker(
            short_threshold=200,
            medium_threshold=800,
            long_threshold=2000
        )

    def test_token_counter_indic_and_english(self):
        eng = "The Eiffel Tower is located in Paris, France."
        hin = "एफिल टॉवर पेरिस, फ्रांस में स्थित एक प्रसिद्ध स्मारक है।"
        self.assertGreater(estimate_tokens(eng), 5)
        self.assertGreater(estimate_tokens(hin), 5)
        self.assertEqual(estimate_tokens(""), 0)

    def test_strategy_1_short_document(self):
        """Test 1: <= 200 tokens -> whole_document strategy, 1 chunk"""
        text = "भारत एक विशाल और विविध देश है। इसकी राजधानी नई दिल्ली है।"
        result = self.chunker.chunk(text)
        self.assertEqual(result["strategy"], "whole_document")
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0], text)

    def test_strategy_2_medium_document(self):
        """Test 2: 201-800 tokens -> sentence_aware strategy, preserves sentence boundaries"""
        sentences = [
            f"यह वाक्य क्रमांक {i} है जो दस्तावेज़ की संरचना को दर्शाता है।"
            for i in range(1, 15)
        ]
        text = " ".join(sentences)
        tokens = estimate_tokens(text)
        self.assertTrue(200 < tokens <= 800)

        result = self.chunker.chunk(text)
        self.assertEqual(result["strategy"], "sentence_aware")
        self.assertGreater(len(result["chunks"]), 1)
        for chunk in result["chunks"]:
            # Check sentence boundary preserved (ends with viram or punctuation)
            self.assertTrue(chunk.endswith("।") or chunk.endswith("."))

    def test_strategy_3_long_document(self):
        """Test 3: 801-2000 tokens -> recursive strategy, respects max tokens"""
        paragraphs = [
            f"अनुच्छेद {p}: " + " ".join([
                f"इस अनुच्छेद में विवरण {i} प्रस्तुत किया गया है।" for i in range(1, 6)
            ])
            for p in range(1, 6)
        ]
        text = "\n\n".join(paragraphs)
        tokens = estimate_tokens(text)
        self.assertTrue(800 < tokens <= 2000)

        result = self.chunker.chunk(text)
        self.assertEqual(result["strategy"], "recursive")
        self.assertGreater(len(result["chunks"]), 1)

    def test_strategy_4_very_long_document(self):
        """Test 4: > 2000 tokens -> semantic_fallback_recursive strategy"""
        long_text = "लंबे दस्तावेज़ का पाठ । " * 600
        tokens = estimate_tokens(long_text)
        self.assertGreater(tokens, 2000)

        result = self.chunker.chunk(long_text)
        self.assertEqual(result["strategy"], "semantic_fallback_recursive")
        self.assertGreater(len(result["chunks"]), 1)


if __name__ == "__main__":
    unittest.main()
