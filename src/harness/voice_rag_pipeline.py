import time
import uuid
import logging
from typing import Dict, Any, Optional

from src.voice.sarvam_stt import SpeechToText
from src.voice.schemas import TranscriptionResult, VoiceRAGResponse
from src.voice.audio import validate_audio
from src.generation.schemas import RAGResponse
from src.harness.rag_pipeline import RAGPipeline
from src.guardrails.policy import get_refusal_message

logger = logging.getLogger("VoiceRAGPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [VoiceRAG] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VoiceRAGPipeline:
    """
    End-to-End Multilingual Voice RAG Pipeline.
    Orchestrates:
    Audio File -> Validation -> Sarvam STT -> Text Normalization -> Guarded RAG Harness -> Spoken Context.
    Tracks complete latency budget across STT, Retrieval, Generation, and Guardrails.
    """

    def __init__(
        self,
        stt: SpeechToText,
        rag_pipeline: RAGPipeline
    ):
        self.stt = stt
        self.rag = rag_pipeline
        self.voice_logs = []

    def run(
        self,
        audio_path: str,
        language_code: str = "unknown",
        top_k: int = 5,
        enable_reranking: bool = False
    ) -> VoiceRAGResponse:
        """
        Execute Voice-to-Answer RAG query.
        """
        if language_code == "auto" or not language_code:
            language_code = "unknown"
        request_id = str(uuid.uuid4())[:8]
        t_e2e_start = time.perf_counter()

        # 1. Audio Validation
        t_val_start = time.perf_counter()
        is_valid, audio_err = validate_audio(audio_path)
        t_val_end = time.perf_counter()
        audio_val_ms = (t_val_end - t_val_start) * 1000

        if not is_valid:
            refusal_text = "I could not process the provided audio file. Please check the format and try again."
            dummy_transcription = TranscriptionResult(text="", error=f"audio_validation_failed: {audio_err}")
            dummy_rag_response = RAGResponse(
                request_id=request_id,
                query="",
                answer=refusal_text,
                sources=[],
                refusal=True,
                error=f"audio_validation_failed: {audio_err}",
                latencies_ms={"total": round(audio_val_ms, 2)}
            )
            return VoiceRAGResponse(
                request_id=request_id,
                transcription=dummy_transcription,
                rag_response=dummy_rag_response,
                latencies_ms={
                    "audio_validation": round(audio_val_ms, 2),
                    "stt": 0.0,
                    "retrieval": 0.0,
                    "generation": 0.0,
                    "total_e2e": round(audio_val_ms, 2)
                }
            )

        # 2. Sarvam Speech-to-Text Transcription
        t_stt_start = time.perf_counter()
        transcription = self.stt.transcribe(audio_path, language_code=language_code)
        t_stt_end = time.perf_counter()
        stt_elapsed_ms = (t_stt_end - t_stt_start) * 1000

        if transcription.error or not transcription.text:
            refusal_text = "I couldn't understand the audio. Please try speaking again."
            dummy_rag_response = RAGResponse(
                request_id=request_id,
                query="",
                answer=refusal_text,
                sources=[],
                refusal=True,
                error=transcription.error or "stt_empty_transcript",
                latencies_ms={"total": round(stt_elapsed_ms, 2)}
            )
            t_e2e_end = time.perf_counter()
            return VoiceRAGResponse(
                request_id=request_id,
                transcription=transcription,
                rag_response=dummy_rag_response,
                latencies_ms={
                    "audio_validation": round(audio_val_ms, 2),
                    "stt": round(stt_elapsed_ms, 2),
                    "retrieval": 0.0,
                    "generation": 0.0,
                    "total_e2e": round((t_e2e_end - t_e2e_start) * 1000, 2)
                }
            )

        # 3. Guarded Text RAG Execution
        rag_response = self.rag.run(
            query=transcription.text,
            top_k=top_k,
            enable_reranking=enable_reranking
        )

        t_e2e_end = time.perf_counter()
        total_e2e_ms = (t_e2e_end - t_e2e_start) * 1000

        latencies = {
            "audio_validation": round(audio_val_ms, 2),
            "stt": round(transcription.stt_latency_ms or stt_elapsed_ms, 2),
            "retrieval": rag_response.latencies_ms.get("retrieval", 0.0),
            "generation": rag_response.latencies_ms.get("generation", 0.0),
            "total_e2e": round(total_e2e_ms, 2)
        }

        voice_response = VoiceRAGResponse(
            request_id=request_id,
            transcription=transcription,
            rag_response=rag_response,
            latencies_ms=latencies
        )

        self.voice_logs.append({
            "request_id": request_id,
            "transcription": transcription.to_dict(),
            "latencies_ms": latencies,
            "grounded": rag_response.grounded,
            "refusal": rag_response.refusal
        })

        return voice_response
