"""Always-on wake word listener for voice control.

Three engines score the stream. openWakeWord runs one trained model per word
(pretrained names or custom .onnx files). sherpa-onnx runs a small open
vocabulary keyword spotter: any English phrase works as a word, spelled into
the model's tokens at start, with no training. Vosk runs a small recognizer
for one language, restricted to the two phrases; it reports a phrase only at
the pause after it, so its hits carry how long ago the phrase ended and the
listener keeps a few seconds of audio to backfill a command from.

One input stream, separate from the recorder's, is scored for two phrases: a
start word that opens a voice command and an end word that closes it (or ends
any running dictation). The same stream doubles as the quiet command
recorder: begin_capture()/end_capture() keep the 16 kHz audio between the two
words, so no recording stream has to open and the first word is never lost.

The engines and sounddevice are imported lazily, only when the listener
starts, so core stays importable without them. Their ONNX models are
downloaded once into WAKEWORD_MODELS_DIR.
"""

import collections
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
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

ENGINES = ('openwakeword', 'sherpa', 'vosk')
# sherpa-onnx's English keyword spotter (zipformer transducer, 3.3M parameters,
# GigaSpeech). Only the int8 encoder/joiner and the fp32 decoder are kept.
SHERPA_MODEL = 'sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01'
SHERPA_MODEL_URL = f'https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{SHERPA_MODEL}.tar.bz2'
SHERPA_MODEL_SHA256 = 'f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a'
SHERPA_MODEL_MAX_BYTES = 64 * 1024 * 1024
SHERPA_MODEL_FILES = {
    'tokens': 'tokens.txt',
    'bpe_model': 'bpe.model',
    'encoder': 'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
    'decoder': 'decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    'joiner': 'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
}

# Vosk's small models by language: (name, MD5 published in its model list).
# Each downloads from VOSK_MODEL_BASE_URL/<name>.zip.
VOSK_MODEL_BASE_URL = 'https://alphacephei.com/vosk/models'
VOSK_MODEL_MAX_BYTES = 400 * 1024 * 1024
VOSK_MODELS = {
    'ar': ('vosk-model-small-ar-0.3', '0eb578c0779f85ef039b7ee5bc1b64ed'),
    'ar-tn': ('vosk-model-small-ar-tn-0.1-linto', '67bbe16beaa17e6cf14f8c295ee5ddd1'),
    'ca': ('vosk-model-small-ca-0.4', '5dc7901815fef3dc8b784b31f309385c'),
    'cn': ('vosk-model-small-cn-0.22', 'd1a8e82933dcc7632667c036b0bb3dbb'),
    'cs': ('vosk-model-small-cs-0.4-rhasspy', 'df5ddac5ee8632f6b46a6fc751953286'),
    'de': ('vosk-model-small-de-0.15', '4f21f92c0897b48287ef8839420608eb'),
    'en-gb': ('vosk-model-small-en-gb-0.15', '6afd611b04b2b47c129c3615dc502383'),
    'en-in': ('vosk-model-small-en-in-0.4', '62fd085f33c8c5fc01a235030e2a1fe3'),
    'en-us': ('vosk-model-small-en-us-0.15', '09ab50ccd62b674cbaa231b825f9c1cb'),
    'eo': ('vosk-model-small-eo-0.42', '3e9319ca789fa06fd3efa68d51c85d6b'),
    'es': ('vosk-model-small-es-0.42', '2d5c94f9859a84881a0ef744738ebd31'),
    'fa': ('vosk-model-small-fa-0.42', 'cc2b18af256ffab2c44055f6a02ecb3d'),
    'fr': ('vosk-model-small-fr-0.22', '8873b1234503f6edd55f54bfff31cf3e'),
    'gu': ('vosk-model-small-gu-0.42', '4595a6f0cc0c88fee6eec7d80496fc10'),
    'hi': ('vosk-model-small-hi-0.22', '80f1265262c9a8a515f2707498e1b485'),
    'it': ('vosk-model-small-it-0.22', 'fbd8f9c72cbb8c3dfa3e4581bd3585f4'),
    'ja': ('vosk-model-small-ja-0.22', '0e3163dd62dfb0d823353718ac3cbf79'),
    'ka': ('vosk-model-small-ka-0.42', '7e95b6a69b60a5f1c45e29d1cd2ac353'),
    'ko': ('vosk-model-small-ko-0.22', 'fa8029a173787a159e0e72fe6135f890'),
    'ky': ('vosk-model-small-ky-0.42', '541eb45589c2b6f8350905086a76f072'),
    'kz': ('vosk-model-small-kz-0.42', '1c7dc487ac5e0c24acfa7566c106ba2d'),
    'nl': ('vosk-model-small-nl-0.22', '4582e1f20d6849099da08511f9797017'),
    'pl': ('vosk-model-small-pl-0.22', '91cbbd6231320467da672be31827b6ac'),
    'pt': ('vosk-model-small-pt-0.3', '458c69371c5a0b9ab6ee8fa417bf89da'),
    'ru': ('vosk-model-small-ru-0.22', 'd1759dc83eb8fd87850129afbd9f4b7b'),
    'sv': ('vosk-model-small-sv-rhasspy-0.15', '5ae431c65fe8636692118792daa22ecc'),
    'te': ('vosk-model-small-te-0.42', 'ae3c093fd7dd883d462983b15900ca61'),
    'tg': ('vosk-model-small-tg-0.22', 'fca4e40de915a5bd2f30e32d60af45c0'),
    'tr': ('vosk-model-small-tr-0.3', '198511631860597639a0ebf779263fcf'),
    'ua': ('vosk-model-small-uk-v3-small', '3188e41b4a38369d8b9527d0149616ab'),
    'uz': ('vosk-model-small-uz-0.22', '4d75acfef76fe919c8fb68cd1179c182'),
    'vn': ('vosk-model-small-vn-0.4', 'b31b474a1ef75488c5fa575f8e2a1269'),
}
# ISO codes for the languages Vosk lists under another name
VOSK_ALIASES = {'en': 'en-us', 'zh': 'cn', 'uk': 'ua', 'vi': 'vn', 'kk': 'kz'}

# Detection: a word fires once per utterance. It re-arms after its score falls
# below half the threshold, and nothing fires within MIN_GAP_S of a detection.
REFIRE_COOLDOWN_S = 1.5
MIN_GAP_S = 0.5
# Recent audio kept to backfill a command whose start word was reported late.
BACKFILL_SECONDS = 10.0
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
    """Scores 80 ms int16 frames for the start and end word."""

    def __init__(self, start_word, end_word, model_dir: Path = WAKEWORD_MODELS_DIR):
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordError(
                f'openWakeWord is not installed ({e.name or e}); see the voice control docs') from e
        start_path = resolve_model_path(start_word, model_dir)
        end_path = resolve_model_path(end_word, model_dir)
        if start_path.resolve() == end_path.resolve():
            raise WakeWordError('the start and end word must be different models')
        melspec, embedding = feature_model_paths(model_dir)
        self._model = Model(
            wakeword_models=[str(start_path), str(end_path)],
            inference_framework='onnx',
            melspec_model_path=str(melspec),
            embedding_model_path=str(embedding),
        )
        # openWakeWord keys each model by its file name without extension
        self._start_key = start_path.stem
        self._end_key = end_path.stem

    def score(self, frame: np.ndarray):
        scores = self._model.predict(frame)
        return float(scores.get(self._start_key, 0.0)), float(scores.get(self._end_key, 0.0))


def keyword_tokens(word, encode: Callable, vocab) -> str:
    """The keyword spotter's token line for a phrase: 'hey_jarvis' -> '▁HE Y ▁JA R VI S'.

    encode turns upper-case text into BPE pieces; every piece must be in the
    model's vocabulary, which only spells English letters and apostrophes.
    """
    word = normalize_word(word)
    if word.endswith('.onnx') or os.sep in word:
        raise WakeWordError(f"{word}: custom .onnx models need voice_control_engine 'openwakeword'")
    text = ' '.join(re.split(r'[_\s]+', word)).strip().upper()
    if not re.fullmatch(r"[A-Z']+( [A-Z']+)*", text):
        raise WakeWordError(f"wake word {word!r}: the keyword model only spells English letters and apostrophes")
    tokens = list(encode(text))
    if not tokens or any(token not in vocab for token in tokens):
        raise WakeWordError(f"wake word {word!r} can't be spelled with the keyword model's tokens")
    return ' '.join(tokens)


def sherpa_model_files(model_dir: Path, download: Optional[Callable] = None) -> dict:
    """Paths of the keyword spotter's files, downloaded and verified on first use.

    download(url, destination) writes the release archive to destination.
    """
    target = Path(model_dir) / SHERPA_MODEL
    files = {key: target / name for key, name in SHERPA_MODEL_FILES.items()}
    if all(path.is_file() and path.stat().st_size > 0 for path in files.values()):
        return files
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    log(f"[VOICE] Downloading keyword spotter model {SHERPA_MODEL}")
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        archive = Path(tmp) / f'{SHERPA_MODEL}.tar.bz2'
        try:
            (download or _download_sherpa)(SHERPA_MODEL_URL, archive)
        except WakeWordError:
            raise
        except Exception as e:
            raise WakeWordError(f'download of {SHERPA_MODEL} failed: {e}') from e
        if not archive.is_file():
            raise WakeWordError(f'download of {SHERPA_MODEL} failed')
        with archive.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != SHERPA_MODEL_SHA256:
            raise WakeWordError(f'{SHERPA_MODEL} does not match its pinned checksum')
        staged = Path(tmp) / SHERPA_MODEL
        _extract_sherpa_files(archive, staged)
        if target.exists():
            shutil.rmtree(target)
        staged.rename(target)
    return files


def _download_sherpa(url: str, destination: Path):
    _download_bounded(url, destination, SHERPA_MODEL_MAX_BYTES)


def _download_bounded(url: str, destination: Path, maximum_bytes: int):
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, destination.open('wb') as output:
        while True:
            block = response.read(64 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise WakeWordError(f'download of {url.rsplit("/", 1)[-1]} exceeded {maximum_bytes} bytes')
            output.write(block)


def _extract_sherpa_files(archive: Path, destination: Path):
    """Copy only the files the spotter needs; nothing else in the archive is written."""
    wanted = {f'{SHERPA_MODEL}/{name}' for name in SHERPA_MODEL_FILES.values()}
    destination.mkdir(parents=True)
    with tarfile.open(archive, 'r:bz2') as bundle:
        members = {m.name.lstrip('./'): m for m in bundle.getmembers() if m.name.lstrip('./') in wanted}
        if set(members) != wanted or not all(m.isfile() for m in members.values()):
            raise WakeWordError(f'{SHERPA_MODEL} archive is missing model files')
        for name, member in members.items():
            source = bundle.extractfile(member)
            with source, (destination / Path(name).name).open('wb') as output:
                shutil.copyfileobj(source, output)


class SherpaKeywordDetector:
    """Spots the start and end phrase with sherpa-onnx; a hit scores 1.0 for one frame."""

    def __init__(self, start_word, end_word, model_dir: Path = WAKEWORD_MODELS_DIR, *,
                 threshold: float = 0.25, boost: float = 1.0, download: Optional[Callable] = None):
        try:
            import sentencepiece
            import sherpa_onnx
        except ImportError as e:
            raise WakeWordError(
                f'sherpa-onnx keyword spotting is not installed ({e.name or e}); see the voice control docs') from e
        files = sherpa_model_files(model_dir, download)
        pieces = sentencepiece.SentencePieceProcessor(model_file=str(files['bpe_model']))
        with open(files['tokens'], encoding='utf-8') as f:
            vocab = {line.split()[0] for line in f if line.strip()}

        def encode(text):
            return pieces.encode(text, out_type=str)

        start = keyword_tokens(start_word, encode, vocab)
        end = keyword_tokens(end_word, encode, vocab)
        start_words, end_words = start.split(), end.split()
        shorter = min(len(start_words), len(end_words))
        if start_words[:shorter] == end_words[:shorter]:
            raise WakeWordError('the start and end word must differ, and neither may begin with the other')
        with tempfile.TemporaryDirectory() as tmp:
            keywords = Path(tmp) / 'keywords.txt'
            keywords.write_text(f'{start} @start\n{end} @end\n', encoding='utf-8')
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(files['tokens']), encoder=str(files['encoder']),
                decoder=str(files['decoder']), joiner=str(files['joiner']),
                keywords_file=str(keywords), num_threads=1, provider='cpu',
                keywords_score=float(boost), keywords_threshold=float(threshold),
            )
        self._stream = self._spotter.create_stream()

    def score(self, frame: np.ndarray):
        self._stream.accept_waveform(SAMPLE_RATE, frame.astype(np.float32) / 32768.0)
        hits = set()
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            hit = self._spotter.get_result(self._stream)
            if hit:
                hits.add(hit)
                self._spotter.reset_stream(self._stream)
        return (1.0 if 'start' in hits else 0.0), (1.0 if 'end' in hits else 0.0)


def vosk_phrase(word) -> str:
    """The words Vosk listens for: 'Koniec_Nagrania' -> 'koniec nagrania'."""
    word = normalize_word(word)
    if word.endswith('.onnx') or os.sep in word:
        raise WakeWordError(f"{word}: custom .onnx models need voice_control_engine 'openwakeword'")
    phrase = ' '.join(re.split(r'[_\s]+', word)).strip()
    if not phrase or not all(re.fullmatch(r"[^\W\d_]+(['-][^\W\d_]+)*", w) for w in phrase.split()):
        raise WakeWordError(f"wake word {word!r}: use letters only (write numbers out)")
    return phrase


def vosk_model_dir(model, model_dir: Path, download: Optional[Callable] = None) -> Path:
    """A Vosk model directory: a language code from VOSK_MODELS, downloaded on
    first use and checked against its pinned MD5, or a path to an unpacked model.

    download(url, destination) writes the model's zip file to destination.
    """
    if not isinstance(model, str) or not model.strip():
        raise WakeWordError('voice_control_vosk_model must be a language code or a model directory')
    model = model.strip()
    if os.sep in model or model.startswith('~'):
        path = Path(model).expanduser()
        if not (path / 'am').is_dir() and not (path / 'conf').is_dir():
            raise WakeWordError(f'not a Vosk model directory: {path}')
        return path
    code = model.lower().replace('_', '-')
    code = VOSK_ALIASES.get(code, code)
    if code not in VOSK_MODELS:
        raise WakeWordError(f"no Vosk model for {model!r}; use one of {', '.join(sorted(VOSK_MODELS))} "
                            'or a path to an unpacked model')
    name, md5 = VOSK_MODELS[code]
    root = Path(model_dir) / 'vosk'
    target = root / name
    if (target / 'am').is_dir() or (target / 'conf').is_dir():
        return target
    root.mkdir(parents=True, exist_ok=True)
    log(f"[VOICE] Downloading Vosk model {name}")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / f'{name}.zip'
        try:
            (download or _download_vosk)(f'{VOSK_MODEL_BASE_URL}/{name}.zip', archive)
        except WakeWordError:
            raise
        except Exception as e:
            raise WakeWordError(f'download of {name} failed: {e}') from e
        if not archive.is_file():
            raise WakeWordError(f'download of {name} failed')
        with archive.open('rb') as f:
            digest = hashlib.file_digest(f, 'md5').hexdigest()
        if digest != md5:
            raise WakeWordError(f'{name} does not match its published checksum')
        staged = Path(tmp) / 'unpacked'
        _extract_vosk_model(archive, staged, name)
        if target.exists():
            shutil.rmtree(target)
        (staged / name).rename(target)
    return target


def _download_vosk(url: str, destination: Path):
    _download_bounded(url, destination, VOSK_MODEL_MAX_BYTES)


def _extract_vosk_model(archive: Path, destination: Path, name: str):
    """Unpack the model's own folder; any entry outside it is an error."""
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = [entry for entry in bundle.infolist() if not entry.is_dir()]
        if not entries:
            raise WakeWordError(f'{name} archive is empty')
        for entry in entries:
            target = (destination / entry.filename).resolve()
            if not entry.filename.startswith(f'{name}/') or not target.is_relative_to(root / name):
                raise WakeWordError(f'{name} archive contains an unexpected path: {entry.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def _vosk_missing_words(vosk, model, words):
    """Words outside the model's vocabulary.

    Vosk drops them from a grammar and says so only in its native log, so
    stderr is captured while one throwaway recognizer is built.
    """
    with tempfile.TemporaryFile() as captured:
        sys.stderr.flush()
        saved = os.dup(2)
        try:
            os.dup2(captured.fileno(), 2)
            vosk.SetLogLevel(0)
            vosk.KaldiRecognizer(model, SAMPLE_RATE, json.dumps(words, ensure_ascii=False))
        finally:
            vosk.SetLogLevel(-1)
            os.dup2(saved, 2)
            os.close(saved)
        captured.seek(0)
        text = captured.read().decode('utf-8', 'replace')
    return [w for w in words if f"missing in vocabulary: '{w}'" in text]


class VoskKeywordDetector:
    """Recognizes only the two phrases with a small Vosk model.

    A phrase counts when a finished utterance contains all its words, each
    at least min_confidence. Utterances finish at the next pause, so
    hit_lag says how long ago the phrase ended.
    """

    def __init__(self, start_word, end_word, model_dir: Path = WAKEWORD_MODELS_DIR, *,
                 model='en-us', min_confidence: float = 0.9, download: Optional[Callable] = None):
        try:
            import vosk
        except ImportError as e:
            raise WakeWordError(f'Vosk is not installed ({e.name or e}); see the voice control docs') from e
        self._phrases = {'start': vosk_phrase(start_word).split(), 'end': vosk_phrase(end_word).split()}
        start, end = self._phrases['start'], self._phrases['end']
        shorter = min(len(start), len(end))
        if start[:shorter] == end[:shorter]:
            raise WakeWordError('the start and end word must differ, and neither may begin with the other')
        path = vosk_model_dir(model, model_dir, download)
        vosk.SetLogLevel(-1)
        self._model = vosk.Model(str(path))
        missing = _vosk_missing_words(vosk, self._model, sorted(set(start + end)))
        if missing:
            raise WakeWordError(f"not in the {path.name} vocabulary: {', '.join(missing)}; "
                                'pick other words or spellings')
        grammar = [' '.join(start), ' '.join(end), '[unk]']
        self._recognizer = vosk.KaldiRecognizer(self._model, SAMPLE_RATE, json.dumps(grammar, ensure_ascii=False))
        self._recognizer.SetWords(True)
        self._min_confidence = float(min_confidence)
        self._samples = 0
        self.hit_lag = {}

    def score(self, frame: np.ndarray):
        self._samples += frame.size
        self.hit_lag = {}
        if not self._recognizer.AcceptWaveform(frame.astype(np.int16).tobytes()):
            return 0.0, 0.0
        words = json.loads(self._recognizer.Result()).get('result', [])
        now = self._samples / SAMPLE_RATE
        for kind, phrase in self._phrases.items():
            for i in range(len(words) - len(phrase) + 1):
                window = words[i:i + len(phrase)]
                if ([w.get('word') for w in window] == phrase
                        and min(float(w.get('conf', 0.0)) for w in window) >= self._min_confidence):
                    self.hit_lag[kind] = max(0.0, now - float(window[-1].get('end', now)))
        return (1.0 if 'start' in self.hit_lag else 0.0), (1.0 if 'end' in self.hit_lag else 0.0)


def make_detector(engine, start_word, end_word, *, keyword_threshold: float = 0.25,
                  keyword_boost: float = 1.0, vosk_model='en-us', vosk_confidence: float = 0.9):
    if engine == 'openwakeword':
        return OpenWakeWordDetector(start_word, end_word)
    if engine == 'sherpa':
        return SherpaKeywordDetector(start_word, end_word, threshold=keyword_threshold, boost=keyword_boost)
    if engine == 'vosk':
        return VoskKeywordDetector(start_word, end_word, model=vosk_model, min_confidence=vosk_confidence)
    raise WakeWordError(f"unknown wake word engine {engine!r}; use {' or '.join(ENGINES)}")


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

    on_event(kind, detected_at) runs on the worker thread with kind 'start',
    'end' or 'capture_limit' and a time.monotonic() stamp of when the word
    ended. It must return quickly: audio queues up while it runs.
    """

    def __init__(self, start_word, end_word, on_event: Callable, *, threshold: float = 0.5,
                 device=None, max_capture_seconds: float = 15.0,
                 engine: str = 'openwakeword', keyword_threshold: float = 0.25,
                 keyword_boost: float = 1.0, vosk_model='en-us', vosk_confidence: float = 0.9,
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
            lambda: make_detector(engine, start_word, end_word, keyword_threshold=keyword_threshold,
                                  keyword_boost=keyword_boost, vosk_model=vosk_model,
                                  vosk_confidence=vosk_confidence))
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
        self._recent = collections.deque()  # the last BACKFILL_SECONDS of 16 kHz blocks
        self._recent_samples = 0
        self._recent_until = float('-inf')  # clock when the newest block arrived
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
        log(f"[VOICE] Listening for wake words ({self._start_word} / {self._end_word})")
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
        self._remember(block)
        self._pending = np.concatenate([self._pending, block]) if self._pending.size else block
        while self._pending.size >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            start_score, end_score = detector.score(pcm)
            now = self._clock()
            lag = getattr(detector, 'hit_lag', None) or {}
            for kind, score in (('start', start_score), ('end', end_score)):
                if self._should_fire(kind, score, now):
                    self._fire(kind, now, now - float(lag.get(kind, 0.0)))

    def _should_fire(self, kind, score, now) -> bool:
        if score < self.threshold * 0.5:
            self._armed[kind] = True
            return False
        if score < self.threshold or not self._armed[kind]:
            return False
        if now - self._last_fire[kind] < REFIRE_COOLDOWN_S or now - self._last_any_fire < MIN_GAP_S:
            return False
        return True

    def _fire(self, kind, now, detected_at=None):
        self._armed[kind] = False
        self._last_fire[kind] = now
        self._last_any_fire = now
        log(f"[VOICE] Heard {'start' if kind == 'start' else 'end'} word")
        try:
            self._on_event(kind, now if detected_at is None else detected_at)
        except Exception as e:
            log(f"[VOICE] Wake word handler failed: {e}")

    # ------------------------ Command capture ------------------------

    def begin_capture(self, since: Optional[float] = None):
        """Keep the audio from now on until end_capture() or cancel_capture().

        since (a clock stamp, such as a start word's detected_at) also keeps
        the recent audio after it, up to BACKFILL_SECONDS.
        """
        with self._capture_lock:
            self._capture = []
            self._capture_samples = 0
            self._capture_limit_hit = False
            if since is not None and self._recent_samples:
                wanted = int(min(BACKFILL_SECONDS, self._recent_until - since) * SAMPLE_RATE)
                if wanted > 0:
                    recent = np.concatenate(self._recent)[-wanted:]
                    recent = recent[:self._max_capture_samples]
                    self._capture.append(recent)
                    self._capture_samples = recent.size

    def _remember(self, block):
        with self._capture_lock:
            self._recent.append(block)
            self._recent_samples += block.size
            self._recent_until = self._clock()
            limit = int(BACKFILL_SECONDS * SAMPLE_RATE)
            while self._recent and self._recent_samples - self._recent[0].size >= limit:
                self._recent_samples -= self._recent.popleft().size

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
