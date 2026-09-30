"""
TTS Service using Kokoro-82M.
Handles high-quality speech synthesis, voice blending, audio caching, and streaming sentence generation.
"""

import os
import io
import re
import time
import hashlib
import logging
import soundfile as sf
import numpy as np
from typing import Generator, List, Dict, Optional, Tuple

from markdown_to_speech import normalizer

logger = logging.getLogger("onmychat.tts")

# Known Kokoro American and British English voices
DEFAULT_VOICES = [
    {"id": "af_heart", "name": "Heart (Default / Warm Female)", "lang": "en-US", "gender": "female"},
    {"id": "af_nicole", "name": "Nicole (Clear & Articulate)", "lang": "en-US", "gender": "female"},
    {"id": "af_sarah", "name": "Sarah (Casual & Soft)", "lang": "en-US", "gender": "female"},
    {"id": "af_nova", "name": "Nova (Dynamic & Friendly)", "lang": "en-US", "gender": "female"},
    {"id": "af_alloy", "name": "Alloy (Neutral & Modern)", "lang": "en-US", "gender": "female"},
    {"id": "af_bella", "name": "Bella (Energetic & Expressive)", "lang": "en-US", "gender": "female"},
    {"id": "af_sky", "name": "Sky (Youthful & Bright)", "lang": "en-US", "gender": "female"},
    {"id": "af_aoede", "name": "Aoede (Gentle & Soothing)", "lang": "en-US", "gender": "female"},
    {"id": "af_jessica", "name": "Jessica (Crisp Professional)", "lang": "en-US", "gender": "female"},
    {"id": "af_kore", "name": "Kore (Calm & Deep)", "lang": "en-US", "gender": "female"},
    {"id": "af_river", "name": "River (Grounded Female)", "lang": "en-US", "gender": "female"},
    {"id": "bf_emma", "name": "Emma (Warm British)", "lang": "en-GB", "gender": "female"},
    {"id": "bf_alice", "name": "Alice (Sophisticated British)", "lang": "en-GB", "gender": "female"},
    {"id": "am_adam", "name": "Adam (Warm Male)", "lang": "en-US", "gender": "male"},
    {"id": "am_michael", "name": "Michael (Deep Male)", "lang": "en-US", "gender": "male"},
    {"id": "am_fenrir", "name": "Fenrir (Rich Male)", "lang": "en-US", "gender": "male"},
    {"id": "am_puck", "name": "Puck (Playful Male)", "lang": "en-US", "gender": "male"},
    {"id": "am_echo", "name": "Echo (Male)", "lang": "en-US", "gender": "male"},
    {"id": "june", "name": "June (Blend Heart + Bella)", "lang": "en-US", "gender": "female"},
    {"id": "calm", "name": "Calm (Blend Heart + Nicole)", "lang": "en-US", "gender": "female"},
]

# Blended voice profiles for enhanced expressiveness, inflections, and naturalness
VOICE_PROFILES = {
    # Melodic warmth of Heart (60%) blended with lively inflection of Bella (40%)
    "june": [("af_heart", 0.60), ("af_bella", 0.40)],
    "expressive": [("af_heart", 0.50), ("af_bella", 0.35), ("af_sarah", 0.15)],
    "lively": [("af_bella", 0.65), ("af_sky", 0.35)],
    "calm": [("af_heart", 0.75), ("af_nicole", 0.25)],
    "warm_male": [("am_adam", 0.65), ("am_michael", 0.35)],
}

class TTSService:
    _instance = None

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(base_dir, "data", "tts_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self._pipeline = None
        self._device = None
        self._voice_cache = {}
        self._micro_cues = {}

        try:
            from config import SETTINGS
            self.default_voice = SETTINGS.get("TTS_DEFAULT_VOICE", "af_heart")
            self.default_speed = float(SETTINGS.get("TTS_SPEED", "0.95"))
        except Exception:
            self.default_voice = "af_heart"
            self.default_speed = 0.95

        self._load_micro_cues()

    @classmethod
    def get_instance(cls) -> "TTSService":
        if cls._instance is None:
            cls._instance = TTSService()
        return cls._instance

    def _get_device(self) -> str:
        if self._device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
            except Exception:
                self._device = "cpu"
            logger.info(f"[TTS] Using device: {self._device}")
        return self._device

    def _get_pipeline(self):
        if self._pipeline is None:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

            from kokoro import KPipeline
            device = self._get_device()
            logger.info(f"[TTS] Initializing Kokoro KPipeline on {device}...")
            self._pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=device)
            logger.info("[TTS] Kokoro KPipeline initialized successfully.")
        return self._pipeline

    def _resolve_voice(self, voice_name: str):
        """Resolves a voice name or blend preset into a voice object/tensor."""
        pipeline = self._get_pipeline()
        voice_name = voice_name or self.default_voice

        # Check in-memory cache
        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]

        # Check blend presets
        if voice_name in VOICE_PROFILES:
            components = VOICE_PROFILES[voice_name]
            blended = None
            for v_id, weight in components:
                v_tensor = pipeline.load_voice(v_id)
                if blended is None:
                    blended = weight * v_tensor
                else:
                    blended = blended + (weight * v_tensor)
            self._voice_cache[voice_name] = blended
            return blended

        # Custom blend syntax: e.g. "af_heart:0.6+af_bella:0.4" or "af_heart+af_bella"
        if "+" in voice_name:
            parts = voice_name.split("+")
            blended = None
            for p in parts:
                p = p.strip()
                if ":" in p:
                    v_id, w_str = p.split(":", 1)
                    w = float(w_str)
                else:
                    v_id = p
                    w = 1.0 / len(parts)
                v_tensor = pipeline.load_voice(v_id.strip())
                if blended is None:
                    blended = w * v_tensor
                else:
                    blended = blended + (w * v_tensor)
            self._voice_cache[voice_name] = blended
            return blended

        # Standard single voice
        v = pipeline.load_voice(voice_name)
        self._voice_cache[voice_name] = v
        return v

    def _cache_path(self, clean_text: str, voice: str, speed: float) -> str:
        key_src = f"{voice}_{speed:.2f}_{clean_text}"
        md5_hash = hashlib.md5(key_src.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{md5_hash}.wav")

    def get_voices(self) -> List[Dict[str, str]]:
        return DEFAULT_VOICES

    def _load_micro_cues(self):
        """Pre-loads 24kHz micro cues into memory for 0ms latency audio splicing."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cues_dirs = [
            os.path.join(base_dir, "micro_cues"),
            os.path.join(base_dir, "data", "micro_cues")
        ]
        for cdir in cues_dirs:
            if os.path.exists(cdir):
                for fname in os.listdir(cdir):
                    if fname.endswith(".wav"):
                        cname = os.path.splitext(fname)[0]
                        if cname not in self._micro_cues:
                            try:
                                audio, sr = sf.read(os.path.join(cdir, fname))
                                if len(audio.shape) > 1:
                                    audio = audio[0]
                                self._micro_cues[cname] = np.asarray(audio, dtype=np.float32)
                                logger.info(f"[TTS] Loaded micro-cue '{cname}' ({len(audio)/sr:.2f}s)")
                            except Exception as e:
                                logger.warning(f"[TTS] Failed to load micro-cue {fname}: {e}")

    def _get_micro_cue(self, name: str) -> Optional[np.ndarray]:
        return self._micro_cues.get(name)

    def _synthesize_audio(self, clean_text: str, voice: str, speed: float) -> bytes:
        cache_file = self._cache_path(clean_text, voice, speed)
        if os.path.exists(cache_file):
            logger.info(f"[TTS] Cache hit for voice={voice} len={len(clean_text)}")
            with open(cache_file, "rb") as f:
                return f.read()

        logger.info(f"[TTS] Synthesizing speech with voice={voice}, speed={speed}...")
        t0 = time.time()

        # Parse any micro-cues like [[cue:chuckle]], [[cue:sigh]]
        cues_found = re.findall(r'\[\[cue:([a-zA-Z0-9_]+)\]\]', clean_text)
        speech_text = re.sub(r'\[\[cue:[a-zA-Z0-9_]+\]\]\s*', '', clean_text).strip()

        audio_parts = []
        for cue_name in cues_found:
            cue_audio = self._get_micro_cue(cue_name)
            if cue_audio is not None and len(cue_audio) > 0:
                audio_parts.append(cue_audio)
                # 120ms pause after cue before speech
                audio_parts.append(np.zeros(int(0.12 * 24000), dtype=np.float32))

        if speech_text:
            pipeline = self._get_pipeline()
            voice_obj = self._resolve_voice(voice)
            generator = pipeline(speech_text, voice=voice_obj, speed=speed)
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    if hasattr(audio, 'detach'):
                        audio = audio.detach().cpu().numpy()
                    audio_parts.append(np.asarray(audio, dtype=np.float32))

        if not audio_parts:
            return b""

        full_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]

        buf = io.BytesIO()
        sf.write(buf, full_audio, 24000, format="WAV")
        wav_bytes = buf.getvalue()

        # Save to disk cache
        try:
            with open(cache_file, "wb") as f:
                f.write(wav_bytes)
        except Exception as e:
            logger.warning(f"[TTS] Failed to write cache: {e}")

        elapsed = time.time() - t0
        audio_sec = len(full_audio) / 24000.0
        logger.info(f"[TTS] Synthesized {audio_sec:.1f}s of audio in {elapsed:.2f}s (RTF={elapsed/audio_sec:.2f})")
        return wav_bytes

    def synthesize(self, raw_markdown: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """
        Synthesizes speech for the provided markdown text.
        Returns WAV audio bytes (16-bit PCM, 24kHz).
        """
        voice = voice or self.default_voice
        speed = speed or self.default_speed

        clean_text = normalizer.normalize(raw_markdown)
        if not clean_text:
            return b""

        return self._synthesize_audio(clean_text, voice, speed)

    def synthesize_sentence(self, sentence: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """
        Synthesizes a single sentence for streaming playback.
        """
        voice = voice or self.default_voice
        speed = speed or self.default_speed
        clean_text = normalizer.normalize(sentence)
        if not clean_text:
            return b""

        return self._synthesize_audio(clean_text, voice, speed)


tts_service = TTSService.get_instance()
