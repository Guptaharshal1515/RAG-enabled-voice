import unittest
import os
import shutil
from pathlib import Path

from src.voice.audio import validate_audio, generate_synthetic_wav
from src.voice.sarvam_stt import MockSarvamSTT, normalize_transcript
from src.voice.schemas import TranscriptionResult, VoiceRAGResponse
from src.harness.voice_rag_pipeline import VoiceRAGPipeline
from src.harness.rag_pipeline import RAGPipeline
from src.generation.schemas import StructuredAnswer
from src.generation.llm import LLM
from src.generation.generator import Generator


class MockFastLLM(LLM):
    def generate(self, prompt: str, timeout_sec: float = 5.0) -> str:
        return '{"answer": "Relevant fact for voice query. [c1]", "source_ids": ["c1"]}'


class MockRetriever:
    def search(self, query: str, top_k: int = 5, enable_reranking: bool = False):
        return {
            "query": query,
            "retrieval_method": "hybrid_rrf",
            "results": [
                {
                    "chunk_id": "c1",
                    "document": {"document_id": "d1", "text": "Relevant fact for voice query.", "language": "en"},
                    "score": 0.90
                }
            ]
        }


class TestVoicePipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/test_audio"
        os.makedirs(self.test_dir, exist_ok=True)
        self.valid_wav = os.path.join(self.test_dir, "sample.wav")
        generate_synthetic_wav(self.valid_wav, duration_sec=0.5)

        # Setup standard RAG components
        self.retriever = MockRetriever()
        self.generator = Generator(MockFastLLM())
        self.rag_pipeline = RAGPipeline(self.retriever, self.generator)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_1_english_voice_rag(self):
        """Test 1: English audio transcription flows end-to-end to grounded answer"""
        stt = MockSarvamSTT(simulated_transcript="What causes earthquakes?", simulated_lang="en-IN")
        voice_rag = VoiceRAGPipeline(stt, self.rag_pipeline)

        resp = voice_rag.run(self.valid_wav)
        self.assertIsInstance(resp, VoiceRAGResponse)
        self.assertEqual(resp.transcription.text, "What causes earthquakes?")
        self.assertFalse(resp.rag_response.refusal)
        self.assertTrue(resp.rag_response.grounded)
        self.assertIn("total_e2e", resp.latencies_ms)
        self.assertIn("stt", resp.latencies_ms)

    def test_2_indic_voice_rag(self):
        """Test 2: Indian language voice query transcription"""
        stt = MockSarvamSTT(simulated_transcript="কৰ্পোৰেচন কি?", simulated_lang="as-IN")
        voice_rag = VoiceRAGPipeline(stt, self.rag_pipeline)

        resp = voice_rag.run(self.valid_wav, language_code="as-IN")
        self.assertEqual(resp.transcription.text, "কৰ্পোৰেচন কি?")
        self.assertEqual(resp.transcription.language, "as-IN")
        self.assertFalse(resp.rag_response.refusal)

    def test_3_normalization_and_noisy_audio(self):
        """Test 3: Extra whitespaces normalized properly"""
        raw = "   what   causes    earthquakes ?   "
        self.assertEqual(normalize_transcript(raw), "what causes earthquakes ?")

    def test_4_empty_audio_rejection(self):
        """Test 4: 0-byte audio file rejected by audio validator"""
        empty_wav = os.path.join(self.test_dir, "empty.wav")
        Path(empty_wav).touch()

        is_valid, err = validate_audio(empty_wav)
        self.assertFalse(is_valid)
        self.assertEqual(err, "empty_audio_file")

        stt = MockSarvamSTT()
        voice_rag = VoiceRAGPipeline(stt, self.rag_pipeline)
        resp = voice_rag.run(empty_wav)
        self.assertTrue(resp.rag_response.refusal)

    def test_5_unsupported_format_rejection(self):
        """Test 5: Unsupported audio extension rejected"""
        bad_file = os.path.join(self.test_dir, "test.exe")
        Path(bad_file).write_text("not audio")

        is_valid, err = validate_audio(bad_file)
        self.assertFalse(is_valid)
        self.assertIn("unsupported_format", err)

    def test_6_oversized_audio_rejection(self):
        """Test 6: Audio exceeding max size constraint rejected"""
        is_valid, err = validate_audio(self.valid_wav, max_size_bytes=10)
        self.assertFalse(is_valid)
        self.assertEqual(err, "audio_file_too_large")

    def test_7_stt_failure_graceful_handling(self):
        """Test 7: STT failure produces friendly refusal message, not 500 crash"""
        failing_stt = MockSarvamSTT(should_fail=True)
        voice_rag = VoiceRAGPipeline(failing_stt, self.rag_pipeline)

        resp = voice_rag.run(self.valid_wav)
        self.assertTrue(resp.rag_response.refusal)
        self.assertIn("couldn't understand", resp.rag_response.answer.lower())


if __name__ == "__main__":
    unittest.main()
