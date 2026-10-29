"""
Text to speech for the scratchpad: the draft read aloud in a chosen voice.

The text goes to OpenRouter's speech endpoint (ElevenLabs Eleven v4 Turbo by
default) with the key stored by 'hyprwhspr config llm', and the MP3 it
returns is played with the first available player. A long draft is split at
paragraph and sentence breaks; the next part is synthesized while the
current one plays, so reading starts after the first part.

Stdlib only (urllib), like llm_refiner, so core stays importable.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

try:
    from .paths import RUNTIME_DIR
except ImportError:
    from paths import RUNTIME_DIR


ENDPOINT = 'https://openrouter.ai/api/v1/audio/speech'
DEFAULT_MODEL = 'elevenlabs/eleven-v4-turbo'
DEFAULT_VOICE = 'sarah'
DEFAULT_TIMEOUT = 30
# The OpenRouter key stored by 'hyprwhspr config llm'
CREDENTIAL = 'openrouter'

VOICES = (
    'george', 'sarah', 'adam', 'alice', 'bella', 'bill', 'brian', 'callum', 'charlie', 'chris',
    'daniel', 'eric', 'harry', 'jessica', 'laura', 'liam', 'lily', 'matilda', 'river', 'roger', 'will',
)

# Characters per request. The endpoint takes far more; a short first part
# starts reading sooner.
CHUNK_CHARS = 1500

# Players that read MP3, best first: (binary, arguments before the file)
PLAYERS = (
    ('mpv', ['--no-video', '--no-terminal', '--really-quiet']),
    ('ffplay', ['-nodisp', '-autoexit', '-loglevel', 'quiet']),
    ('pw-play', []),
    ('paplay', []),
)


class SpeechError(Exception):
    """The draft could not be synthesized or played."""


def normalize_voice(value) -> Optional[str]:
    """The voice name in lower case, or None if it isn't one of VOICES."""
    voice = value.strip().lower() if isinstance(value, str) else ''
    return voice if voice in VOICES else None


def configured_voice(get_setting: Callable) -> str:
    return normalize_voice(get_setting('scratchpad_voice', DEFAULT_VOICE)) or DEFAULT_VOICE


def speakable_text(text: str) -> str:
    """The draft without Markdown marks a voice would read out or stumble on."""
    lines = []
    for line in (text or '').splitlines():
        line = re.sub(r'^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+(\[[ xX]\]\s+)?)', '', line)
        line = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', line)  # [label](url) -> label
        line = re.sub(r'(\*\*|__|`+)', '', line)
        lines.append(line.rstrip())
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def split_text(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Parts of at most limit characters, cut at paragraph, sentence or word breaks."""
    pieces = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units = [paragraph] if len(paragraph) <= limit else re.split(r'(?<=[.!?…])\s+', paragraph)
        for unit in units:
            while len(unit) > limit:
                cut = unit.rfind(' ', 0, limit + 1)
                cut = cut if cut > 0 else limit
                pieces.append(unit[:cut].strip())
                unit = unit[cut:].strip()
            if unit:
                pieces.append(unit)
    parts = []
    for piece in pieces:
        if parts and len(parts[-1]) + 2 + len(piece) <= limit:
            parts[-1] = f'{parts[-1]}\n\n{piece}'
        else:
            parts.append(piece)
    return parts


def _timeout(get_setting: Callable) -> float:
    value = get_setting('scratchpad_speech_timeout', DEFAULT_TIMEOUT)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 300:
        return float(value)
    return float(DEFAULT_TIMEOUT)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode('utf-8', 'replace'))
        message = body.get('error', {}).get('message') if isinstance(body, dict) else None
    except Exception:
        message = None
    if isinstance(message, str) and message.strip():
        return f'HTTP {exc.code}: {message.strip()[:200]}'
    return f'HTTP {exc.code}'


def synthesize(text: str, get_setting: Callable, voice: Optional[str] = None,
               credential_lookup: Optional[Callable] = None,
               opener: Callable = urllib.request.urlopen) -> bytes:
    """MP3 audio of text in voice (default: scratchpad_voice), or raise SpeechError."""
    if credential_lookup is None:
        try:
            from .credential_manager import get_credential as credential_lookup
        except ImportError:
            from credential_manager import get_credential as credential_lookup
    api_key = credential_lookup(CREDENTIAL)
    api_key = api_key.strip() if isinstance(api_key, str) else ''
    if not api_key:
        raise SpeechError("no OpenRouter API key stored; run 'hyprwhspr config llm'")

    model = get_setting('scratchpad_speech_model', DEFAULT_MODEL) or DEFAULT_MODEL
    body = {
        'model': model,
        'input': text,
        'voice': normalize_voice(voice) or configured_voice(get_setting),
        'response_format': 'mp3',
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode('utf-8'), method='POST', headers={
            'Content-Type': 'application/json',
            'User-Agent': 'hyprwhspr',
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://hyprwhspr.com',
            'X-Title': 'hyprwhspr',
        },
    )
    try:
        with opener(request, timeout=_timeout(get_setting)) as response:
            audio = response.read()
    except urllib.error.HTTPError as e:
        raise SpeechError(_error_detail(e)) from e
    except urllib.error.URLError as e:
        raise SpeechError(f'cannot reach OpenRouter: {e.reason}') from e
    except TimeoutError as e:
        raise SpeechError('request timed out') from e
    except (OSError, ValueError) as e:
        raise SpeechError(f'request failed: {type(e).__name__}') from e
    if not audio:
        raise SpeechError('OpenRouter returned no audio')
    return audio


def find_player() -> Optional[List[str]]:
    """Command prefix of the first installed MP3 player, or None."""
    for binary, args in PLAYERS:
        if shutil.which(binary):
            return [binary] + args
    return None


class SpeechPlayer:
    """Reads one text at a time aloud on a worker thread; stop() cuts it off.

    synthesize(part, voice) returns MP3 bytes for one part. on_error(message)
    is called from the worker when reading fails; nothing is called when it
    finishes or is stopped.
    """

    def __init__(self, synthesize: Callable[[str, Optional[str]], bytes],
                 on_error: Optional[Callable[[str], None]] = None,
                 player: Optional[Callable[[], Optional[List[str]]]] = None,
                 audio_dir: Optional[Path] = None):
        self._synthesize = synthesize
        self._on_error = on_error
        self._find_player = player or find_player
        self._audio_dir = Path(audio_dir) if audio_dir is not None else RUNTIME_DIR
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        """Start reading text, replacing anything being read; raise SpeechError up front."""
        parts = split_text(speakable_text(text))
        if not parts:
            raise SpeechError('nothing to read')
        command = self._find_player()
        if command is None:
            raise SpeechError('no audio player found; install mpv, ffmpeg (ffplay) or pipewire (pw-play)')
        self.stop()
        with self._lock:
            self._stop = stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(parts, voice, command, stop),
                                            daemon=True, name="ScratchpadSpeech")
            self._thread.start()

    def stop(self) -> bool:
        """Stop reading; False if nothing was being read."""
        with self._lock:
            thread, process = self._thread, self._process
            self._stop.set()
            if process is not None and process.poll() is None:
                process.terminate()
        if thread is None or not thread.is_alive():
            return False
        if thread is not threading.current_thread():
            thread.join(timeout=2)
        return True

    def _run(self, parts, voice, command, stop):
        # Not a with block: a stop must not wait for a request in flight
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ScratchpadSpeechSynth")
        try:
            upcoming = pool.submit(self._synthesize, parts[0], voice)
            for index in range(len(parts)):
                audio = upcoming.result()
                if stop.is_set():
                    return
                if index + 1 < len(parts):
                    upcoming = pool.submit(self._synthesize, parts[index + 1], voice)
                self._play(audio, command, stop)
                if stop.is_set():
                    return
        except Exception as e:
            if not stop.is_set():
                stop.set()
                if self._on_error is not None:
                    self._on_error(str(e) if isinstance(e, SpeechError) else type(e).__name__)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _play(self, audio, command, stop):
        self._audio_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, path = tempfile.mkstemp(dir=self._audio_dir, prefix='.scratchpad-speech-', suffix='.mp3')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(audio)
            with self._lock:
                if stop.is_set():
                    return
                try:
                    self._process = subprocess.Popen(
                        command + [path], stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError as e:
                    raise SpeechError(f'cannot start {command[0]}: {e.strerror or e}') from e
            code = self._process.wait()
            if code != 0 and not stop.is_set():
                raise SpeechError(f'{command[0]} exited with status {code}')
        finally:
            with self._lock:
                self._process = None
            try:
                os.unlink(path)
            except OSError:
                pass
