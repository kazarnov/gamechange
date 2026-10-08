"""NVIDIA Nemotron 3.5 ASR (cache-aware streaming FastConformer-RNNT) via transformers."""

import logging
import re
import threading
from collections.abc import Callable

import numpy as np
import torch
import torchaudio
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

log = logging.getLogger(__name__)

LANG_TAG = re.compile(r"<[a-z]{2,3}(-[A-Za-z]{2,4})?>")


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    return torchaudio.functional.resample(torch.from_numpy(audio), orig_sr, target_sr).numpy()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", LANG_TAG.sub("", text)).strip()


class ASREngine:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype, language: str, lookahead: int):
        self.language = language
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForRNNT.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
        self.processor.set_num_lookahead_tokens(lookahead)
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        log.info("ASR loaded (%s, streaming latency %d ms)", model_id, self.processor.streaming_latency_ms)

    def _inputs(self, audio: np.ndarray, **kwargs):
        inputs = self.processor(
            audio, sampling_rate=self.sample_rate, language=self.language, return_tensors="pt", **kwargs
        )
        return inputs.to(self.model.device, dtype=self.model.dtype)

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        """Offline transcription of a full utterance."""
        out = self.model.generate(**self._inputs(audio), return_dict_in_generate=True)
        return clean(self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0])

    def stream(self, on_partial: Callable[[str], None]) -> "StreamingTranscription":
        return StreamingTranscription(self, on_partial)


class StreamingTranscription:
    """One utterance of cache-aware streaming recognition.

    Audio is pushed with feed(); a background thread runs model.generate over a
    generator that blocks until each next chunk of audio is available. finish()
    pads the tail with silence so the last words get their lookahead context.
    """

    def __init__(self, engine: ASREngine, on_partial: Callable[[str], None]):
        self.engine = engine
        self.on_partial = on_partial
        self.audio = np.zeros(0, dtype=np.float32)
        self.cond = threading.Condition()
        self.closed = False
        self.text = ""
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def feed(self, samples: np.ndarray):
        with self.cond:
            self.audio = np.concatenate([self.audio, samples])
            self.cond.notify_all()

    def finish(self) -> str:
        """Blocking: closes the stream and returns the final transcript."""
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.thread.join()
        if self.error:
            # Fall back to offline recognition on the whole utterance
            log.warning("streaming ASR failed (%s); falling back to offline", self.error)
            return self.engine.transcribe(self.audio) if len(self.audio) else ""
        return clean(self.text)

    def _slice(self, start: int, end: int) -> np.ndarray | None:
        """Waits for audio[start:end]; once closed, zero-pads. None when the stream is drained."""
        p = self.engine.processor
        with self.cond:
            self.cond.wait_for(lambda: len(self.audio) >= end or self.closed)
            audio = self.audio
            # After close, emit one extra chunk of silence so the tail gets lookahead
            if self.closed and start >= len(audio) + p.num_samples_per_audio_chunk:
                return None
        chunk = audio[max(start, 0) : end]
        if len(chunk) < end - max(start, 0):
            chunk = np.pad(chunk, (0, end - max(start, 0) - len(chunk)))
        return chunk

    def _run(self):
        e, p = self.engine, self.engine.processor
        try:
            first_audio = self._slice(0, p.num_samples_first_audio_chunk)
            if first_audio is None:
                return
            first = e._inputs(first_audio, is_streaming=True, is_first_audio_chunk=True)

            def features():
                yield first.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
                hop, n_fft = p.feature_extractor.hop_length, p.feature_extractor.n_fft
                mel_idx = p.num_mel_frames_first_audio_chunk
                while True:
                    start = mel_idx * hop - n_fft // 2
                    chunk = self._slice(start, start + p.num_samples_per_audio_chunk)
                    if chunk is None:
                        return
                    yield e._inputs(chunk, is_streaming=True, is_first_audio_chunk=False).input_features
                    mel_idx += p.num_mel_frames_per_audio_chunk

            streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=True)

            def generate():
                try:
                    with torch.inference_mode():
                        e.model.generate(**{**first, "input_features": features(), "streamer": streamer})
                except Exception as exc:  # surface to finish()
                    self.error = exc
                    streamer.end()

            gen = threading.Thread(target=generate, daemon=True)
            gen.start()
            for piece in streamer:
                if piece:
                    self.text += piece
                    self.on_partial(clean(self.text))
            gen.join()
        except Exception as exc:
            self.error = exc
