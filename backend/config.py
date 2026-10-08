from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM
    ollama_api_key: str = ""
    ollama_host: str = "https://ollama.com"
    llm_model: str = "glm-5.3-flash"
    # GLM 5.3 Flash leaks its reasoning into the reply with think=false; "low" keeps it
    # separate and is the fastest clean option. Accepts true | false | low | medium | high
    llm_think: bool | Literal["low", "medium", "high"] = "low"
    llm_max_tool_rounds: int = 5
    system_prompt: str = (
        "You are a friendly voice assistant having a spoken conversation. "
        "Your replies are converted to speech, so keep them short and natural: "
        "one to three sentences, no markdown, no lists, no emojis, no code. "
        "Spell out numbers and symbols the way you would say them. "
        "Use the available tools when they help you answer or act for the user, "
        "and briefly say what you did. Reply in the language the user speaks."
    )

    # ASR
    asr_model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    asr_language: str = "auto"
    asr_mode: str = "streaming"  # streaming | offline
    asr_lookahead: int = 3
    asr_device: str = "cuda"
    asr_dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    # TTS
    tts_model: str = "k2-fsa/OmniVoice"
    tts_device: str = "cuda"
    # GTX 16xx cards have no tensor cores: fp16 is ~6x slower there and OmniVoice
    # produces unintelligible audio in fp16, so default to float32
    tts_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    tts_num_step: int = 8
    tts_speed: float = 1.0
    tts_voice_instruct: str = "female, moderate pitch, american accent"
    tts_ref_audio: str = ""
    tts_ref_text: str = ""

    # Browser agent (Playwright)
    browser_enabled: bool = True
    website_url: str = "https://demo.playwright.dev/todomvc/"
    browser_headless: bool = False  # forced on when there is no display
    browser_profile_dir: str = "browser-profile"  # cookies/logins persist here
    browser_llm_model: str = ""  # empty = same as LLM_MODEL
    browser_llm_think: bool | Literal["low", "medium", "high"] = "low"
    browser_max_steps: int = 25
    browser_screenshots: bool = True  # send a screenshot to the web UI after each step
    skills_dir: str = "skills"

    # Pictures and videos: the ComfyUI manager API on the RunPod pod (comfyui/API.md)
    comfyui_url: str = ""  # https://<pod-id>-8000.proxy.runpod.net; empty = off
    comfyui_api_key: str = ""  # MANAGER_API_KEY from comfyui/pod.env
    comfyui_descriptions: str = "comfyui/descriptions.yaml"  # what each workflow is for
    comfyui_timeout_minutes: int = 30  # give up waiting for one generation after this
    media_dir: str = "media"  # generated and attached files, served at /media

    # VAD / turn taking
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600
    vad_min_speech_ms: int = 250
    vad_preroll_ms: int = 300


settings = Settings()
