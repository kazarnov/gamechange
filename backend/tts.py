"""OmniVoice TTS (k2-fsa) with a fixed voice across sentences."""

import difflib
import hashlib
import logging
import re
import threading
from collections.abc import Callable

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, VoiceClonePrompt

from .config import ROOT

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
VOICES_DIR = ROOT / "voices"
REFERENCE_TEXT = "Hello, it is nice to meet you. I am your assistant, and I am happy to help you with anything today."
DESIGN_ATTEMPTS = 8


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class TTSEngine:
    """One voice, designed from TTS_VOICE_INSTRUCT or cloned from TTS_REF_AUDIO: nothing to pick
    (voicestudio.py offers a choice). The voice calls match VoiceStudioTTS's."""

    choosable = False

    def __init__(self, model_id: str, device: str, dtype: torch.dtype, num_step: int, speed: float,
                 voice_instruct: str, ref_audio: str = "", ref_text: str = "",
                 transcribe: Callable[[np.ndarray], str] | None = None):
        """`transcribe` (24 kHz audio -> text) is used to verify a designed voice is intelligible."""
        self.model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
        self.num_step = num_step
        self.speed = speed
        self.transcribe = transcribe
        self.lock = threading.Lock()  # one synthesis at a time on the GPU
        self.voice = self._load_voice(voice_instruct, ref_audio, ref_text)
        self.default = {"id": "local", "name": "This computer’s voice", "language": "", "archetype": None,
                        "description": "Cloned from your recording" if ref_audio else voice_instruct}
        self.synthesize("Warming up.")  # first call pays CUDA init costs
        log.info("TTS loaded (%s)", model_id)

    def _load_voice(self, instruct: str, ref_audio: str, ref_text: str) -> VoiceClonePrompt:
        """Without a fixed prompt OmniVoice picks a new random voice per call, so we
        always synthesize from a voice-clone prompt: either the user's reference clip
        or a reference we design once from the instruct text and cache on disk."""
        if ref_audio:
            return self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

        key = hashlib.sha1(f"{instruct}|{self.model.dtype}".encode()).hexdigest()[:10]
        prompt_path = VOICES_DIR / f"designed-{key}.pt"
        if prompt_path.exists():
            return VoiceClonePrompt.load(str(prompt_path))

        VOICES_DIR.mkdir(exist_ok=True)
        wav_path = VOICES_DIR / f"designed-{key}.wav"
        sf.write(wav_path, self._design(instruct), SAMPLE_RATE)
        prompt = self.model.create_voice_clone_prompt(ref_audio=str(wav_path), ref_text=REFERENCE_TEXT)
        prompt.save(str(prompt_path))
        return prompt

    def _design(self, instruct: str) -> np.ndarray:
        """Voice design is stochastic and sometimes yields mumbling; since every later
        sentence clones this reference, retry with new seeds until it transcribes well."""
        best, best_score = None, -1.0
        for seed in range(DESIGN_ATTEMPTS):
            torch.manual_seed(seed)
            audio = self.model.generate(text=REFERENCE_TEXT, instruct=instruct, num_step=32)[0]
            if self.transcribe is None:
                return audio
            heard = self.transcribe(audio)
            score = difflib.SequenceMatcher(None, _words(REFERENCE_TEXT), _words(heard)).ratio()
            log.info("Designing voice %r, attempt %d: intelligibility %.2f", instruct, seed + 1, score)
            if score > best_score:
                best, best_score = audio, score
            if score >= 0.9:
                break
        if best_score < 0.9:
            log.warning("Best designed voice only scored %.2f; consider TTS_REF_AUDIO", best_score)
        return best

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        """Returns float32 mono audio at 24 kHz, always in the one voice."""
        with self.lock, torch.inference_mode():
            audio = self.model.generate(
                text=text, voice_clone_prompt=self.voice, num_step=self.num_step, speed=self.speed
            )[0]
        return np.asarray(audio, dtype=np.float32)

    def profiles(self) -> list[dict]:
        return [self.default]

    def catalog(self, q: str = "", lang: str = "", exclude: set[str] = frozenset()) -> tuple[list[dict], int, list[str]]:
        return [], 0, []

    def adopt(self, archetype: str) -> dict:
        raise ValueError("voices can only be picked with a VoiceStudio server (TTS_URL)")

    def sample(self, voice: str | None = None, archetype: str | None = None) -> np.ndarray:
        return self.synthesize(REFERENCE_TEXT)


_MARKDOWN = re.compile(r"[*_`#>|~]+|\[([^\]]*)\]\([^)]*\)")


def speakable(text: str) -> str:
    """Strips markdown the LLM may still emit so it isn't read aloud."""
    return re.sub(r"\s+", " ", _MARKDOWN.sub(lambda m: m.group(1) or " ", text)).strip()


class SentenceChunker:
    """Splits a token stream into speakable chunks as early as possible."""

    END = re.compile(r"(.+?[.!?。！？…]+[\"')\]]*)(\s+|$)", re.S)

    def __init__(self, max_chars: int = 220, first_clause_chars: int = 20):
        self.buf = ""
        self.max_chars = max_chars
        # Speak the first clause as soon as it's complete so audio starts sooner
        self.first_clause = re.compile(rf"(.{{{first_clause_chars},}}?[,;:，、—–])\s", re.S)

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            m = self.END.match(self.buf)
            if m and m.group(2):  # require whitespace after the punctuation
                out.append(m.group(1))
                self.buf = self.buf[m.end():]
                continue
            if self.first_clause and not out and (m := self.first_clause.match(self.buf)):
                out.append(m.group(1))
                self.buf = self.buf[m.end():]
                continue
            if "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                if line.strip():
                    out.append(line)
                continue
            if len(self.buf) > self.max_chars:
                cut = max(self.buf.rfind(", ", 0, self.max_chars), self.buf.rfind(" ", 0, self.max_chars))
                cut = cut if cut > 0 else self.max_chars
                out.append(self.buf[: cut + 1])
                self.buf = self.buf[cut + 1:]
                continue
            break
        if out:
            self.first_clause = None
        return [s for s in (speakable(x) for x in out) if s]

    def flush(self) -> list[str]:
        rest, self.buf = speakable(self.buf), ""
        return [rest] if rest else []
