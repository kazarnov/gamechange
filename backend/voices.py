"""The voice a video is read in: the user's own, recorded on the page and cloned, or one that fits the
video, picked from the voice server's catalog by a description ("young female, upbeat, for an ad").

Recording: the user reads one of SCRIPTS out loud, which says they agree to their voice being cloned.
The recording is checked (long and loud enough, and the assistant's speech recognition hears that
script in it), cleaned up and cloned on VoiceStudio, with the reading kept as its consent. A copy stays
with the user's conversations, so the voice can be made again when the voice server loses it (a new pod
starts with only its demo voice). A recorded voice is its owner's: nobody else sees or uses it.
"""

import base64
import difflib
import hashlib
import io
import re
import unicodedata
import wave

import numpy as np

RATE = 24000
SCRIPTS = {
    "en": {"label": "English",
           "text": "Hi, this is my own voice, and I agree to have it cloned to narrate my brand's videos. "
                   "Fresh ideas, warm colours and good stories: that's what we share with you every day."},
    "fr": {"label": "Français",
           "text": "Bonjour, c'est ma propre voix, et j'accepte qu'elle soit clonée pour raconter les vidéos de "
                   "ma marque. De nouvelles idées, de belles couleurs et de bonnes histoires : voilà ce que nous "
                   "partageons avec vous chaque jour."},
}
SAMPLES = {"en": "Here's your voice, ready for your videos.", "fr": "Voici votre voix, prête pour vos vidéos."}
MIN_SECONDS, MAX_SECONDS = 5.0, 40.0
HEARD, WORD_FOR_WORD = 0.6, 0.85  # how much of the script must be heard; above this, the script is the reference


class RecordingError(ValueError):
    """What is wrong with a recording, said to the user."""


def owner_tag(owner: str) -> str:
    """What marks a recorded voice as this user's on the voice server (its personality field)."""
    return "clone:" + hashlib.sha1(owner.encode()).hexdigest()[:10]


def words(text: str) -> list[str]:
    """Lower-case words without accents: "Clonée" and "clonee" are the same word."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", plain.lower())


def heard(transcript: str, script: str) -> float:
    """How much of the script is in what was heard, 0 to 1."""
    return difflib.SequenceMatcher(None, words(script), words(transcript)).ratio()


# --- a recording ----------------------------------------------------------------------

def read_recording(data: str) -> np.ndarray:
    """A 16-bit WAV, as a data URL or base64, at any rate -> 24 kHz float32 mono."""
    try:
        raw = base64.b64decode(data.split(",", 1)[1] if data.startswith("data:") else data)
        with wave.open(io.BytesIO(raw)) as w:
            rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    except Exception as exc:
        raise RecordingError("the recording couldn't be read; record it again") from exc
    if width != 2:
        raise RecordingError("the recording should be 16-bit WAV")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(audio):
        n = int(len(audio) * RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _trim(audio: np.ndarray) -> np.ndarray:
    """Without the quiet before the first word and after the last."""
    step = RATE // 50  # 20 ms
    frames = len(audio) // step
    if not frames:
        return audio
    rms = np.sqrt((audio[: frames * step].reshape(frames, step) ** 2).mean(axis=1))
    voiced = np.flatnonzero(rms > max(0.006, 0.12 * np.percentile(rms, 95)))
    if not len(voiced):
        return audio[:0]
    start = max(0, voiced[0] * step - int(0.15 * RATE))
    end = min(len(audio), (voiced[-1] + 1) * step + int(0.3 * RATE))
    return audio[start:end]


def check(audio: np.ndarray) -> np.ndarray:
    """The reading, trimmed and evened out; RecordingError if it can't make a good voice."""
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak < 0.02:
        raise RecordingError("I could barely hear you. Move closer to the microphone and read it again")
    if (np.abs(audio) > 0.99).mean() > 0.002:
        raise RecordingError("it's too loud and crackles. Move back a little from the microphone and read it again")
    audio = _trim(audio)
    seconds = len(audio) / RATE
    if seconds < MIN_SECONDS:
        raise RecordingError("that was short. Read the whole text, then stop")
    if seconds > MAX_SECONDS:
        raise RecordingError("that was long. Read just the text, then stop")
    return (audio * (0.9 / max(float(np.abs(audio).max()), 1e-6))).astype(np.float32)


def wav(audio: np.ndarray, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# --- a voice that fits a video ------------------------------------------------------------

GENDERS = {"female": {"female", "woman", "women", "girl", "lady", "feminine"},
           "male": {"male", "man", "men", "guy", "boy", "masculine"}}
# Words a description may use, and the catalog's words for them
ALSO = {
    "deep": ["low"], "bass": ["low"], "baritone": ["low"], "kid": ["child"], "kids": ["child"],
    "children": ["child"], "teen": ["teenager"], "teenage": ["teenager"], "old": ["elderly"],
    "senior": ["elderly"], "older": ["elderly"], "mature": ["middle"],
    "ad": ["advertisement"], "ads": ["advertisement"], "advert": ["advertisement"],
    "advertising": ["advertisement"], "commercial": ["advertisement", "promo"], "promo": ["advertisement"],
    "promotion": ["advertisement", "promo"], "sale": ["advertisement", "promo"],
    "sales": ["advertisement", "promo"], "marketing": ["advertisement"],
    "vlog": ["social", "vlogger"], "influencer": ["social", "vlogger"], "tiktok": ["social"],
    "reel": ["social"], "reels": ["social"], "instagram": ["social"], "podcast": ["podcaster"],
    "narrator": ["narration"], "story": ["narration", "storyteller"], "documentary": ["narration"],
    "audiobook": ["narration"], "explainer": ["informative"], "tutorial": ["informative"],
    "educational": ["informative"], "friendly": ["conversational"], "casual": ["conversational"],
    "luxury": ["luxe"], "luxurious": ["luxe"], "premium": ["luxe"], "elegant": ["luxe"], "classy": ["luxe"],
    "energetic": ["upbeat", "hype"], "energy": ["upbeat"], "excited": ["hype", "upbeat"],
    "exciting": ["hype"], "hyped": ["hype"], "lively": ["upbeat"], "fun": ["upbeat"], "cheerful": ["upbeat"],
    "uk": ["british"], "england": ["british"], "us": ["american"], "usa": ["american"],
}
STOP = {"a", "an", "the", "and", "or", "with", "voice", "voices", "for", "of", "in", "on", "to", "like", "who",
        "that", "sounds", "sounding", "speaker", "tone", "style", "accent", "pitch", "very", "bit", "slightly",
        "slight", "quite", "some", "one", "please", "someone", "person", "video", "videos", "our", "my", "brand"}


def _plain_name(name: str) -> str:
    return " ".join(w for w in words(name) if w != "the")


def named(voices: list[dict], name: str) -> dict | None:
    """The voice with this id or name ("The Luxe", "luxe", "My voice")."""
    want = _plain_name(name)
    for v in voices:
        if name.strip() == v["id"] or (want and want == _plain_name(v["name"])):
            return v
    return None


def match(catalog: list[dict], description: str, n: int = 4) -> list[dict]:
    """The catalog voices closest to a description, best first. A gender asked for is a must; then
    the more of its words a voice has (in its name, description, use and language), the better, and
    featured voices first among equals."""
    asked = [w for w in words(description) if w not in STOP]
    gender = next((g for g, said in GENDERS.items() if said & set(asked)), None)
    wanted = [{w, *ALSO.get(w, [])} for w in asked if not any(w in said for said in GENDERS.values())]
    scored = []
    for i, v in enumerate(catalog):
        have = set(words(f"{v['name']} {v['description']} {v.get('use', '')} {v.get('language', '')}"))
        if gender and gender not in have:
            continue
        score = sum(1 for group in wanted
                    if any(t in have or (len(t) >= 4 and any(h.startswith(t) for h in have)) for t in group))
        scored.append((-(score + (0.3 if v.get("featured") else 0)), i, v))
    scored.sort(key=lambda x: x[:2])
    return [v for _, _, v in scored[:n]]
