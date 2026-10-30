"""Always-on wake word listener for voice control (openWakeWord).

One input stream, separate from the recorder's, is scored for two phrases: a
start word that opens a voice command and an end word that closes it (or ends
any running dictation). The same stream doubles as the quiet command
recorder: begin_capture()/end_capture() keep the 16 kHz audio between the two
words, so no recording stream has to open and the first word is never lost.

openWakeWord, onnxruntime and sounddevice are imported lazily, only when the
listener starts, so core stays importable without them. Their ONNX models are
downloaded once into WAKEWORD_MODELS_DIR.
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from .paths import WAKEWORD_MODELS_DIR
    from .service_log import log
except ImportError:
    from paths import WAKEWORD_MODELS_DIR
    from service_log import log


SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # openWakeWord scores 80 ms frames
# Single-score openWakeWord models; 'timer' and 'weather' are multi-class.
PRETRAINED_WORDS = ('alexa', 'hey_jarvis', 'hey_mycroft', 'hey_rhasspy')

# Detection: a word fires once per utterance. It re-arms after its score falls
# below half the threshold, and nothing fires within MIN_GAP_S of a detection.
REFIRE_COOLDOWN_S = 1.5
MIN_GAP_S = 0.5
# No audio for this long means the stream died (suspend, server restart).
STALL_TIMEOUT_S = 3.0
REOPEN_BACKOFF_S = (1.0, 2.0, 5.0, 10.0)


class WakeWordError(RuntimeError):
    """The listener cannot run with this configuration or installation."""


def normalize_word(word) -> str:
    """'Hey Jarvis' -> 'hey_jarvis'; paths are returned expanded."""
    if not isinstance(word, str) or not word.strip():
        raise WakeWordError('wake word must be a non-empty string')
    word = word.strip()
    if word.endswith(('.onnx', '.tflite')) or os.sep in word:
        return str(Path(word).expanduser())
    return word.lower().replace(' ', '_').replace('-', '_')


def resolve_model_path(word, model_dir: Path, download: Optional[Callable] = None) -> Path:
    """The ONNX file for a pretrained word name or a custom model path.

    Pretrained models (and openWakeWord's two shared feature models) are
    downloaded into model_dir on first use.
    """
    word = normalize_word(word)
    if word.endswith('.tflite'):
        raise WakeWordError(f'{word}: use the .onnx export; the tflite runtime is not supported')
    if word.endswith('.onnx') or os.sep in word:
        path = Path(word)
        if not path.is_file():
            raise WakeWordError(f'wake word model not found: {path}')
        return path
    if word not in PRETRAINED_WORDS:
        raise WakeWordError(
            f'unknown wake word {word!r}; use one of {", ".join(PRETRAINED_WORDS)} '
            'or a path to a custom .onnx model')
    import openwakeword
    url = openwakeword.MODELS[word]['download_url'].replace('.tflite', '.onnx')
    return _ensure_downloaded(url, model_dir, download)


def feature_model_paths(model_dir: Path, download: Optional[Callable] = None):
    """(melspectrogram, embedding) ONNX paths, downloaded on first use."""
    import openwakeword
    return tuple(
        _ensure_downloaded(openwakeword.FEATURE_MODELS[name]['download_url'].replace('.tflite', '.onnx'),
                           model_dir, download)
        for name in ('melspectrogram', 'embedding')
    )


def _ensure_downloaded(url: str, model_dir: Path, download: Optional[Callable]) -> Path:
    path = Path(model_dir) / url.rsplit('/', 1)[-1]
    if path.is_file() and path.stat().st_size > 0:
        return path
    if download is None:
        from openwakeword.utils import download_file as download
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    log(f"[VOICE] Downloading wake word model {path.name}")
    download(url, str(model_dir))
    if not (path.is_file() and path.stat().st_size > 0):
        raise WakeWordError(f'download of {path.name} failed')
    return path


class OpenWakeWordDetector:
    """Scores 80 ms int16 frames for the start and end word (end_word None: start only)."""

    def __init__(self, start_word, end_word, model_dir: Path = WAKEWORD_MODELS_DIR):
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordError(
                f'openWakeWord is not installed ({e.name or e}); see the voice control docs') from e
        start_path = resolve_model_path(start_word, model_dir)
        end_path = resolve_model_path(end_word, model_dir) if end_word is not None else None
        if end_path is not None and start_path.resolve() == end_path.resolve():
            raise WakeWordError('the start and end word must be different models')
        melspec, embedding = feature_model_paths(model_dir)
        self._model = Model(
            wakeword_models=[str(path) for path in (start_path, end_path) if path is not None],
            inference_framework='onnx',
            melspec_model_path=str(melspec),
            embedding_model_path=str(embedding),
        )
        # openWakeWord keys each model by its file name without extension
        self._start_key = start_path.stem
        self._end_key = end_path.stem if end_path is not None else None

    def score(self, frame: np.ndarray):
        scores = self._model.predict(frame)
        end = float(scores.get(self._end_key, 0.0)) if self._end_key else 0.0
        return float(scores.get(self._start_key, 0.0)), end


def _default_stream_factory(device, samplerate, blocksize, callback):
    import sounddevice as sd
    try:
        return sd.InputStream(device=device, samplerate=samplerate, channels=1,
                              dtype='float32', blocksize=blocksize, callback=callback)
    except Exception:
        # Some devices refuse 16 kHz; take their native rate and resample.
        native = int(sd.query_devices(device, 'input')['default_samplerate'])
        if native == samplerate:
            raise
        return sd.InputStream(device=device, samplerate=native, channels=1,
                              dtype='float32', blocksize=int(native * 0.08), callback=callback)


class _Resampler:
    """Streaming conversion of one stream's blocks to 16 kHz."""

    def __init__(self, source_rate: int):
        self.source_rate = int(source_rate)
        self._stream = None
        if self.source_rate != SAMPLE_RATE:
            import soxr
            self._stream = soxr.ResampleStream(self.source_rate, SAMPLE_RATE, 1, dtype='float32')

    def __call__(self, block: np.ndarray) -> np.ndarray:
        if self._stream is None:
            return block
        return np.asarray(self._stream.resample_chunk(block), dtype=np.float32).reshape(-1)


class WakeWordListener:
    """Run the wake word stream on a worker thread and report detections.

    end_word may be None for a listener with a single word, which fires 'start'.

    on_event(kind, detected_at) runs on the worker thread with kind 'start',
    'end' or 'capture_limit' and a time.monotonic() stamp. It must return
    quickly: audio queues up while it runs.
    """

    def __init__(self, start_word, end_word, on_event: Callable, *, threshold: float = 0.5,
                 device=None, max_capture_seconds: float = 15.0,
                 detector_factory: Optional[Callable] = None,
                 stream_factory: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._start_word = start_word
        self._end_word = end_word
        self._on_event = on_event
        self.threshold = float(threshold)
        self._device = device
        self._max_capture_samples = int(max(1.0, float(max_capture_seconds)) * SAMPLE_RATE)
        self._detector_factory = detector_factory or (
            lambda: OpenWakeWordDetector(start_word, end_word))
        self._stream_factory = stream_factory or _default_stream_factory
        self._clock = clock

        self._stop = threading.Event()
        self._thread = None
        self._blocks = queue.Queue(maxsize=256)
        self._pending = np.zeros(0, dtype=np.float32)

        self._armed = {'start': True, 'end': True}
        self._last_fire = {'start': float('-inf'), 'end': float('-inf')}
        self._last_any_fire = float('-inf')

        self._capture_lock = threading.Lock()
        self._capture = None  # list of 16 kHz float32 blocks while capturing
        self._capture_samples = 0
        self._capture_limit_hit = False
        self.ready = threading.Event()  # set once the models are loaded

    # ------------------------ Lifecycle ------------------------

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="WakeWordListener")
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def _run(self):
        try:
            detector = self._detector_factory()
        except Exception as e:
            log(f"[VOICE] Wake word listener disabled: {e}")
            return
        self.ready.set()
        words = self._start_word if self._end_word is None else f"{self._start_word} / {self._end_word}"
        log(f"[VOICE] Listening for wake words ({words})")
        attempt = 0
        while not self._stop.is_set():
            stream = None
            try:
                stream, resample = self._open_stream()
                attempt = 0
                self._pump(stream, resample, detector)
            except Exception as e:
                log(f"[VOICE] Wake word stream error: {e}")
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
            if self._stop.is_set():
                break
            delay = REOPEN_BACKOFF_S[min(attempt, len(REOPEN_BACKOFF_S) - 1)]
            attempt += 1
            self._stop.wait(delay)

    def _open_stream(self):
        while True:  # drop blocks left over from a dead stream
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                break
        self._pending = np.zeros(0, dtype=np.float32)

        def callback(indata, frames, time_info, status):
            try:
                self._blocks.put_nowait(np.array(indata[:, 0], dtype=np.float32))
            except queue.Full:
                pass  # the worker fell behind; detection can afford the gap

        stream = self._stream_factory(self._device, SAMPLE_RATE, FRAME_SAMPLES, callback)
        stream.start()
        return stream, _Resampler(getattr(stream, 'samplerate', SAMPLE_RATE))

    def _pump(self, stream, resample, detector):
        last_audio = self._clock()
        while not self._stop.is_set():
            try:
                block = self._blocks.get(timeout=0.5)
            except queue.Empty:
                if self._clock() - last_audio >= STALL_TIMEOUT_S:
                    log("[VOICE] Wake word stream stalled - reopening")
                    return
                continue
            last_audio = self._clock()
            self.process_block(resample(block), detector)

    # ------------------------ Detection ------------------------

    def process_block(self, block: np.ndarray, detector):
        """Feed 16 kHz float32 audio: capture it, score whole frames, fire events."""
        self._append_capture(block)
        self._pending = np.concatenate([self._pending, block]) if self._pending.size else block
        while self._pending.size >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            start_score, end_score = detector.score(pcm)
            now = self._clock()
            for kind, score in (('start', start_score), ('end', end_score)):
                if self._should_fire(kind, score, now):
                    self._fire(kind, now)

    def _should_fire(self, kind, score, now) -> bool:
        if score < self.threshold * 0.5:
            self._armed[kind] = True
            return False
        if score < self.threshold or not self._armed[kind]:
            return False
        if now - self._last_fire[kind] < REFIRE_COOLDOWN_S or now - self._last_any_fire < MIN_GAP_S:
            return False
        return True

    def _fire(self, kind, now):
        self._armed[kind] = False
        self._last_fire[kind] = now
        self._last_any_fire = now
        log(f"[VOICE] Heard {'start' if kind == 'start' else 'end'} word")
        try:
            self._on_event(kind, now)
        except Exception as e:
            log(f"[VOICE] Wake word handler failed: {e}")

    # ------------------------ Command capture ------------------------

    def begin_capture(self):
        """Keep the audio from now on until end_capture() or cancel_capture()."""
        with self._capture_lock:
            self._capture = []
            self._capture_samples = 0
            self._capture_limit_hit = False

    def cancel_capture(self):
        with self._capture_lock:
            self._capture = None
            self._capture_samples = 0

    def capturing(self) -> bool:
        with self._capture_lock:
            return self._capture is not None

    def end_capture(self, trim_seconds: float = 0.0) -> Optional[np.ndarray]:
        """The captured 16 kHz audio minus its last trim_seconds; None if not capturing."""
        with self._capture_lock:
            blocks, self._capture = self._capture, None
            self._capture_samples = 0
        if blocks is None:
            return None
        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        trim = int(max(0.0, trim_seconds) * SAMPLE_RATE)
        return audio[:-trim] if trim else audio

    def _append_capture(self, block):
        fire = False
        with self._capture_lock:
            if self._capture is None or self._capture_limit_hit:
                return
            room = self._max_capture_samples - self._capture_samples
            if block.size >= room:
                block = block[:room]
                self._capture_limit_hit = fire = True
            self._capture.append(block)
            self._capture_samples += block.size
        if fire:
            try:
                self._on_event('capture_limit', self._clock())
            except Exception as e:
                log(f"[VOICE] Wake word handler failed: {e}")
