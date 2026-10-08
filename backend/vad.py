"""Server-side voice activity detection with Silero VAD.

Turns a continuous 16 kHz mic stream into speech start / audio / speech end events.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME = 512  # Silero needs exactly 512 samples per call at 16 kHz (32 ms)
FRAME_MS = FRAME * 1000 // SAMPLE_RATE


@dataclass
class VadEvent:
    kind: str  # "start" | "audio" | "end"
    audio: np.ndarray | None = None


class TurnDetector:
    def __init__(self, threshold: float, min_silence_ms: int, min_speech_ms: int, preroll_ms: int):
        self.model = load_silero_vad()
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.min_silence_frames = max(1, min_silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self.pending = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.candidate: list[np.ndarray] = []

    def process(self, samples: np.ndarray) -> list[VadEvent]:
        """Feed float32 samples; returns events in order."""
        self.pending = np.concatenate([self.pending, samples])
        events: list[VadEvent] = []
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            events.extend(self._frame(frame))
        return events

    def _frame(self, frame: np.ndarray) -> list[VadEvent]:
        with torch.no_grad():
            prob = self.model(torch.from_numpy(frame), SAMPLE_RATE).item()

        if not self.in_speech:
            if prob >= self.threshold:
                # Accumulate until speech lasts long enough to not be a click/cough
                self.candidate.append(frame)
                if len(self.candidate) >= self.min_speech_frames:
                    self.in_speech = True
                    self.silence_frames = 0
                    audio = np.concatenate([*self.preroll, *self.candidate])
                    self.preroll.clear()
                    self.candidate = []
                    return [VadEvent("start"), VadEvent("audio", audio)]
            else:
                for f in self.candidate:
                    self.preroll.append(f)
                self.candidate = []
                self.preroll.append(frame)
            return []

        events = [VadEvent("audio", frame)]
        if prob < self.neg_threshold:
            self.silence_frames += 1
            if self.silence_frames >= self.min_silence_frames:
                self.in_speech = False
                self.silence_frames = 0
                events.append(VadEvent("end"))
        else:
            self.silence_frames = 0
        return events
