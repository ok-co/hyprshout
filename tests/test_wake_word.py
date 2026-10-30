import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import wake_word  # noqa: E402
from wake_word import FRAME_SAMPLES, SAMPLE_RATE, WakeWordError, WakeWordListener  # noqa: E402


class ScriptedDetector:
    """Returns (start, end) scores from a script, one pair per 80 ms frame."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.frames = []

    def score(self, frame):
        self.frames.append(frame)
        return self.scores.pop(0) if self.scores else (0.0, 0.0)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def frame(value=0.1):
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


class WordResolutionTests(unittest.TestCase):
    def test_names_normalize_and_paths_expand(self):
        self.assertEqual(wake_word.normalize_word(" Hey Jarvis "), "hey_jarvis")
        self.assertEqual(wake_word.normalize_word("hey-mycroft"), "hey_mycroft")
        self.assertEqual(wake_word.normalize_word("~/w/x.onnx"), str(Path("~/w/x.onnx").expanduser()))
        for bad in ("", "   ", None, 3):
            with self.assertRaises(WakeWordError):
                wake_word.normalize_word(bad)

    def test_custom_models_must_be_existing_onnx_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "computer.onnx"
            model.write_bytes(b"onnx")
            self.assertEqual(wake_word.resolve_model_path(str(model), Path(tmp)), model)
            with self.assertRaisesRegex(WakeWordError, "not found"):
                wake_word.resolve_model_path(str(Path(tmp) / "missing.onnx"), Path(tmp))
            with self.assertRaisesRegex(WakeWordError, "onnx export"):
                wake_word.resolve_model_path(str(Path(tmp) / "computer.tflite"), Path(tmp))
            with self.assertRaisesRegex(WakeWordError, "unknown wake word"):
                wake_word.resolve_model_path("timer", Path(tmp))

    def test_pretrained_models_download_once_as_onnx(self):
        fake = types.SimpleNamespace(MODELS={"hey_jarvis": {
            "download_url": "https://example.invalid/hey_jarvis_v0.1.tflite"}})
        calls = []

        def download(url, directory):
            calls.append(url)
            (Path(directory) / url.rsplit("/", 1)[-1]).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"openwakeword": fake}):
            directory = Path(tmp) / "wakeword"
            path = wake_word.resolve_model_path("hey_jarvis", directory, download)
            again = wake_word.resolve_model_path("hey_jarvis", directory, download)
        self.assertEqual(path.name, "hey_jarvis_v0.1.onnx")
        self.assertEqual(again, path)
        self.assertEqual(calls, ["https://example.invalid/hey_jarvis_v0.1.onnx"])

    def test_failed_download_is_an_error(self):
        fake = types.SimpleNamespace(MODELS={"alexa": {"download_url": "https://example.invalid/alexa.tflite"}})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"openwakeword": fake}):
            with self.assertRaisesRegex(WakeWordError, "download"):
                wake_word.resolve_model_path("alexa", Path(tmp), lambda url, directory: None)


class ListenerDetectionTests(unittest.TestCase):
    def listener(self, **kwargs):
        self.events = []
        self.clock = Clock()
        return WakeWordListener("hey_jarvis", "hey_mycroft",
                                lambda kind, at: self.events.append((kind, at)),
                                clock=self.clock, **kwargs)

    def feed(self, listener, detector, frames=1, step=0.08):
        for _ in range(frames):
            listener.process_block(frame(), detector)
            self.clock.now += step

    def test_a_word_fires_once_per_utterance(self):
        listener = self.listener()
        detector = ScriptedDetector([(0.9, 0.0)] * 5 + [(0.1, 0.0)] * 30 + [(0.9, 0.0)])
        self.feed(listener, detector, frames=36)
        self.assertEqual([kind for kind, _ in self.events], ["start", "start"])

    def test_frames_reach_the_detector_as_int16_pcm(self):
        listener = self.listener()
        detector = ScriptedDetector([])
        listener.process_block(np.concatenate([frame(0.5), frame(0.5)[:100]]), detector)
        self.assertEqual(len(detector.frames), 1)  # the remainder waits for a whole frame
        self.assertEqual(detector.frames[0].dtype, np.int16)
        self.assertEqual(int(detector.frames[0][0]), int(0.5 * 32767))

    def test_the_other_word_is_ignored_right_after_a_detection(self):
        listener = self.listener()
        detector = ScriptedDetector([(0.9, 0.9), (0.0, 0.9)] + [(0.0, 0.0)] * 10 + [(0.0, 0.9)])
        self.feed(listener, detector, frames=13)
        self.assertEqual([kind for kind, _ in self.events], ["start", "end"])
        self.assertGreaterEqual(self.events[1][1] - self.events[0][1], wake_word.MIN_GAP_S)

    def test_capture_keeps_audio_between_the_words_minus_the_trim(self):
        listener = self.listener()
        detector = ScriptedDetector([])
        self.feed(listener, detector)            # before the command: not kept
        listener.begin_capture()
        self.feed(listener, detector, frames=25)  # 2 s
        audio = listener.end_capture(trim_seconds=0.5)
        self.assertEqual(len(audio), 25 * FRAME_SAMPLES - SAMPLE_RATE // 2)
        self.assertIsNone(listener.end_capture())
        self.assertFalse(listener.capturing())

    def test_cancelled_capture_returns_nothing(self):
        listener = self.listener()
        listener.begin_capture()
        self.feed(listener, ScriptedDetector([]))
        listener.cancel_capture()
        self.assertIsNone(listener.end_capture())

    def test_capture_limit_fires_once_and_bounds_memory(self):
        listener = self.listener(max_capture_seconds=1.0)
        listener.begin_capture()
        self.feed(listener, ScriptedDetector([]), frames=30)
        self.assertEqual([kind for kind, _ in self.events], ["capture_limit"])
        self.assertEqual(len(listener.end_capture()), SAMPLE_RATE)

    def test_handler_errors_do_not_stop_detection(self):
        clock = Clock()
        listener = WakeWordListener("a", "b", mock.Mock(side_effect=RuntimeError("boom")), clock=clock)
        listener.process_block(frame(), ScriptedDetector([(0.9, 0.0)]))
        self.assertFalse(listener._armed["start"])


class ListenerStreamTests(unittest.TestCase):
    def test_worker_scores_stream_audio_and_stops(self):
        events = []
        streams = []

        class FakeStream:
            samplerate = SAMPLE_RATE

            def __init__(self, device, rate, blocksize, callback):
                self.callback = callback
                streams.append(self)

            def start(self):
                for _ in range(3):
                    self.callback(np.full((FRAME_SAMPLES, 1), 0.2, dtype=np.float32), FRAME_SAMPLES, None, None)

            def stop(self):
                pass

            def close(self):
                pass

        detector = ScriptedDetector([(0.0, 0.0), (0.9, 0.0)])
        listener = WakeWordListener(
            "a", "b", lambda kind, at: (events.append(kind), listener._stop.set()),
            detector_factory=lambda: detector, stream_factory=FakeStream)
        self.assertTrue(listener.start())
        listener._thread.join(timeout=5)
        self.assertFalse(listener._thread.is_alive())
        self.assertTrue(listener.ready.is_set())
        self.assertEqual(events, ["start"])
        self.assertEqual(len(streams), 1)

    def test_detector_failure_disables_the_listener(self):
        listener = WakeWordListener("a", "b", mock.Mock(),
                                    detector_factory=mock.Mock(side_effect=WakeWordError("not installed")),
                                    stream_factory=mock.Mock())
        with mock.patch.object(wake_word, "log") as log:
            listener.start()
            listener._thread.join(timeout=5)
        self.assertFalse(listener.ready.is_set())
        self.assertIn("not installed", log.call_args[0][0])

    def test_missing_openwakeword_is_reported_as_wake_word_error(self):
        with mock.patch.dict(sys.modules, {"openwakeword": None, "openwakeword.model": None}):
            with self.assertRaisesRegex(WakeWordError, "openWakeWord is not installed"):
                wake_word.OpenWakeWordDetector("hey_jarvis", "hey_mycroft")


FAKE_VOCAB = {"▁HE", "Y", "▁COM", "PUT", "ER", "▁OVER", "▁AND", "▁OUT", "▁STOP", "'", "S"}


def fake_encode(text):
    pieces = {"HEY": ["▁HE", "Y"], "COMPUTER": ["▁COM", "PUT", "ER"], "OVER": ["▁OVER"], "AND": ["▁AND"],
              "OUT": ["▁OUT"], "STOP": ["▁STOP"], "WHAT'S": ["▁WHAT", "'", "S"]}
    return [piece for word in text.split() for piece in pieces.get(word, ["▁" + word])]


class KeywordTokenTests(unittest.TestCase):
    def test_phrases_in_any_spelling_become_token_lines(self):
        for word in ("hey computer", "Hey_Computer", " HEY-computer ", "hey  computer"):
            self.assertEqual(wake_word.keyword_tokens(word, fake_encode, FAKE_VOCAB), "▁HE Y ▁COM PUT ER")

    def test_unspellable_phrases_are_rejected(self):
        with self.assertRaisesRegex(WakeWordError, "English letters"):
            wake_word.keyword_tokens("hey zürich", fake_encode, FAKE_VOCAB)
        with self.assertRaisesRegex(WakeWordError, "English letters"):
            wake_word.keyword_tokens("agent 47", fake_encode, FAKE_VOCAB)
        with self.assertRaisesRegex(WakeWordError, "can't be spelled"):
            wake_word.keyword_tokens("what's up", fake_encode, FAKE_VOCAB)  # ▁WHAT is not in the vocab

    def test_onnx_models_need_the_openwakeword_engine(self):
        with self.assertRaisesRegex(WakeWordError, "openwakeword"):
            wake_word.keyword_tokens("~/models/hey_computer.onnx", fake_encode, FAKE_VOCAB)


def sherpa_archive(path, extra=None, skip=()):
    """A release-shaped archive with stub model files; returns its SHA-256."""
    with tarfile.open(path, "w:bz2") as bundle:
        names = [n for n in wake_word.SHERPA_MODEL_FILES.values() if n not in skip]
        entries = [(f"{wake_word.SHERPA_MODEL}/{n}", n.encode()) for n in names] + list(extra or [])
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SherpaModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.models = self.root / "wakeword"
        self.calls = []

    def download_from(self, archive):
        def download(url, destination):
            self.calls.append(url)
            Path(destination).write_bytes(Path(archive).read_bytes())
        return download

    def test_model_downloads_once_and_keeps_only_the_needed_files(self):
        archive = self.root / "release.tar.bz2"
        digest = sherpa_archive(archive, extra=[(f"{wake_word.SHERPA_MODEL}/test_wavs/0.wav", b"wav"),
                                                ("../escape.txt", b"no")])
        with mock.patch.object(wake_word, "SHERPA_MODEL_SHA256", digest):
            files = wake_word.sherpa_model_files(self.models, self.download_from(archive))
            again = wake_word.sherpa_model_files(self.models, self.download_from(archive))
        self.assertEqual(again, files)
        self.assertEqual(self.calls, [wake_word.SHERPA_MODEL_URL])
        self.assertEqual(files["encoder"].read_bytes(), wake_word.SHERPA_MODEL_FILES["encoder"].encode())
        self.assertEqual(sorted(p.name for p in (self.models / wake_word.SHERPA_MODEL).iterdir()),
                         sorted(wake_word.SHERPA_MODEL_FILES.values()))
        self.assertEqual([p.name for p in self.models.iterdir()], [wake_word.SHERPA_MODEL])
        self.assertFalse((self.root / "escape.txt").exists())

    def test_checksum_mismatch_installs_nothing(self):
        archive = self.root / "release.tar.bz2"
        sherpa_archive(archive)
        with mock.patch.object(wake_word, "SHERPA_MODEL_SHA256", "0" * 64):
            with self.assertRaisesRegex(WakeWordError, "checksum"):
                wake_word.sherpa_model_files(self.models, self.download_from(archive))
        self.assertEqual(list(self.models.iterdir()), [])

    def test_archive_missing_a_file_is_an_error(self):
        archive = self.root / "release.tar.bz2"
        digest = sherpa_archive(archive, skip=(wake_word.SHERPA_MODEL_FILES["joiner"],))
        with mock.patch.object(wake_word, "SHERPA_MODEL_SHA256", digest):
            with self.assertRaisesRegex(WakeWordError, "missing model files"):
                wake_word.sherpa_model_files(self.models, self.download_from(archive))
        self.assertEqual(list(self.models.iterdir()), [])

    def test_network_errors_become_wake_word_errors(self):
        def download(url, destination):
            raise OSError("offline")
        with self.assertRaisesRegex(WakeWordError, "offline"):
            wake_word.sherpa_model_files(self.models, download)


class FakeSpotter:
    """sherpa_onnx.KeywordSpotter stand-in: emits scripted results, one per decode."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.keywords = Path(kwargs["keywords_file"]).read_text(encoding="utf-8")
        self.results = []
        self.received = []
        self.resets = 0
        FakeSpotter.instances.append(self)

    def create_stream(self):
        spotter = self

        class Stream:
            def accept_waveform(self, rate, samples):
                spotter.received.append((rate, samples))
                spotter.pending = 1

        return Stream()

    def is_ready(self, stream):
        return self.pending > 0

    def decode_stream(self, stream):
        self.pending -= 1

    def get_result(self, stream):
        return self.results.pop(0) if self.results else ""

    def reset_stream(self, stream):
        self.resets += 1


class SherpaDetectorTests(unittest.TestCase):
    def setUp(self):
        FakeSpotter.instances.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = Path(tmp.name) / wake_word.SHERPA_MODEL
        model.mkdir()
        for name in wake_word.SHERPA_MODEL_FILES.values():
            (model / name).write_text("stub", encoding="utf-8")
        (model / "tokens.txt").write_text("".join(f"{t} {i}\n" for i, t in enumerate(sorted(FAKE_VOCAB))),
                                          encoding="utf-8")
        self.model_dir = Path(tmp.name)
        processor = mock.Mock()
        processor.encode.side_effect = lambda text, out_type=str: fake_encode(text)
        modules = {
            "sherpa_onnx": types.SimpleNamespace(KeywordSpotter=FakeSpotter),
            "sentencepiece": types.SimpleNamespace(SentencePieceProcessor=mock.Mock(return_value=processor)),
        }
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def detector(self, start="hey computer", end="over and out", **kwargs):
        return wake_word.SherpaKeywordDetector(start, end, self.model_dir, **kwargs)

    def test_phrases_and_tuning_reach_the_spotter(self):
        self.detector(threshold=0.4, boost=2.0)
        spotter = FakeSpotter.instances[0]
        self.assertEqual(spotter.keywords, "▁HE Y ▁COM PUT ER @start\n▁OVER ▁AND ▁OUT @end\n")
        self.assertEqual((spotter.kwargs["keywords_threshold"], spotter.kwargs["keywords_score"]), (0.4, 2.0))
        self.assertEqual(spotter.kwargs["provider"], "cpu")
        self.assertFalse(Path(spotter.kwargs["keywords_file"]).exists())  # a temporary file

    def test_hits_score_one_frame_and_reset_the_stream(self):
        detector = self.detector()
        spotter = FakeSpotter.instances[0]
        spotter.results = ["", "start", "end"]
        pcm = np.full(FRAME_SAMPLES, 16384, dtype=np.int16)
        self.assertEqual(detector.score(pcm), (0.0, 0.0))
        self.assertEqual(detector.score(pcm), (1.0, 0.0))
        self.assertEqual(detector.score(pcm), (0.0, 1.0))
        self.assertEqual(spotter.resets, 2)
        rate, samples = spotter.received[0]
        self.assertEqual(rate, SAMPLE_RATE)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(float(samples[0]), 0.5)

    def test_listener_fires_once_per_hit(self):
        detector = self.detector()
        FakeSpotter.instances[0].results = ["start"]
        events = []
        clock = Clock()
        listener = WakeWordListener("hey computer", "over and out", lambda kind, at: events.append(kind), clock=clock)
        for _ in range(5):
            listener.process_block(frame(), detector)
            clock.now += 0.08
        self.assertEqual(events, ["start"])

    def test_words_must_differ_and_not_begin_with_each_other(self):
        for start, end in (("hey computer", "Hey_Computer"), ("hey computer", "hey computer stop"),
                           ("hey computer stop", "hey computer")):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(WakeWordError, "begin with"):
                self.detector(start, end)
        self.detector("hey computer", "hey stop")  # a shared first word is fine

    def test_missing_packages_are_reported_as_wake_word_errors(self):
        with mock.patch.dict(sys.modules, {"sherpa_onnx": None}):
            with self.assertRaisesRegex(WakeWordError, "sherpa-onnx keyword spotting is not installed"):
                self.detector()

    def test_engines_are_chosen_by_name(self):
        with mock.patch.object(wake_word, "SherpaKeywordDetector") as sherpa:
            wake_word.make_detector("sherpa", "a", "b", keyword_threshold=0.3, keyword_boost=1.5)
        sherpa.assert_called_once_with("a", "b", threshold=0.3, boost=1.5)
        with mock.patch.object(wake_word, "VoskKeywordDetector") as vosk:
            wake_word.make_detector("vosk", "a", "b", vosk_model="pl", vosk_confidence=0.8)
        vosk.assert_called_once_with("a", "b", model="pl", min_confidence=0.8)
        with self.assertRaisesRegex(WakeWordError, "unknown wake word engine"):
            wake_word.make_detector("porcupine", "a", "b")


class VoskPhraseTests(unittest.TestCase):
    def test_phrases_keep_their_letters_in_any_language(self):
        self.assertEqual(wake_word.vosk_phrase("Hej Komputer"), "hej komputer")
        self.assertEqual(wake_word.vosk_phrase("koniec_nagrania"), "koniec nagrania")
        self.assertEqual(wake_word.vosk_phrase("Zażółć gęślą"), "zażółć gęślą")
        self.assertEqual(wake_word.vosk_phrase("aujourd'hui"), "aujourd'hui")

    def test_digits_and_model_paths_are_rejected(self):
        with self.assertRaisesRegex(WakeWordError, "letters only"):
            wake_word.vosk_phrase("agent 47")
        with self.assertRaisesRegex(WakeWordError, "openwakeword"):
            wake_word.vosk_phrase("~/models/x.onnx")


def vosk_zip(path, name, extra=()):
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(f"{name}/", "")
        bundle.writestr(f"{name}/am/final.mdl", "am")
        bundle.writestr(f"{name}/conf/model.conf", "conf")
        for entry, data in extra:
            bundle.writestr(entry, data)
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


class VoskModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.models = self.root / "wakeword"
        self.calls = []

    def download_from(self, archive):
        def download(url, destination):
            self.calls.append(url)
            Path(destination).write_bytes(Path(archive).read_bytes())
        return download

    def test_language_models_download_once_and_verify(self):
        archive = self.root / "pl.zip"
        md5 = vosk_zip(archive, "vosk-model-small-pl-test")
        with mock.patch.dict(wake_word.VOSK_MODELS, {"pl": ("vosk-model-small-pl-test", md5)}):
            path = wake_word.vosk_model_dir("PL", self.models, self.download_from(archive))
            again = wake_word.vosk_model_dir("pl", self.models, self.download_from(archive))
        self.assertEqual(path, again)
        self.assertEqual(path, self.models / "vosk" / "vosk-model-small-pl-test")
        self.assertEqual((path / "am" / "final.mdl").read_text(), "am")
        self.assertEqual(self.calls, [f"{wake_word.VOSK_MODEL_BASE_URL}/vosk-model-small-pl-test.zip"])
        self.assertEqual([p.name for p in (self.models / "vosk").iterdir()], ["vosk-model-small-pl-test"])

    def test_iso_codes_resolve_to_vosk_names(self):
        archive = self.root / "en.zip"
        md5 = vosk_zip(archive, "vosk-model-small-en-us-test")
        with mock.patch.dict(wake_word.VOSK_MODELS, {"en-us": ("vosk-model-small-en-us-test", md5)}):
            path = wake_word.vosk_model_dir("en", self.models, self.download_from(archive))
        self.assertEqual(path.name, "vosk-model-small-en-us-test")

    def test_checksum_mismatch_and_unsafe_archives_install_nothing(self):
        cases = (
            ("0" * 32, (), "checksum"),
            (None, (("../evil.txt", "x"),), "unexpected path"),
            (None, (("other/file.txt", "x"),), "unexpected path"),
        )
        for md5, extra, error in cases:
            with self.subTest(error=error, extra=extra):
                archive = self.root / "bad.zip"
                digest = vosk_zip(archive, "vosk-model-small-pl-test", extra)
                with mock.patch.dict(wake_word.VOSK_MODELS, {"pl": ("vosk-model-small-pl-test", md5 or digest)}):
                    with self.assertRaisesRegex(WakeWordError, error):
                        wake_word.vosk_model_dir("pl", self.models, self.download_from(archive))
                self.assertEqual(list((self.models / "vosk").iterdir()), [])
                self.assertFalse((self.root / "evil.txt").exists())

    def test_unknown_languages_and_paths(self):
        with self.assertRaisesRegex(WakeWordError, "no Vosk model for 'klingon'"):
            wake_word.vosk_model_dir("klingon", self.models)
        with self.assertRaisesRegex(WakeWordError, "not a Vosk model directory"):
            wake_word.vosk_model_dir(str(self.root / "missing"), self.models)
        own = self.root / "my-model"
        (own / "am").mkdir(parents=True)
        self.assertEqual(wake_word.vosk_model_dir(str(own), self.models), own)

    def test_missing_words_are_read_from_the_native_log(self):
        class Recognizer:
            def __init__(self, model, rate, grammar):
                for word in json.loads(grammar):
                    if word.startswith("x"):
                        os.write(2, f"WARNING Ignoring word missing in vocabulary: '{word}'\n".encode())

        fake = types.SimpleNamespace(KaldiRecognizer=Recognizer, SetLogLevel=mock.Mock())
        self.assertEqual(wake_word._vosk_missing_words(fake, None, ["hej", "xżółw", "koniec"]), ["xżółw"])
        self.assertEqual(fake.SetLogLevel.call_args_list[-1], mock.call(-1))


class FakeRecognizer:
    """vosk.KaldiRecognizer stand-in: finishes an utterance on scripted frames."""

    instances = []

    def __init__(self, model, rate, grammar=None):
        self.grammar = json.loads(grammar) if grammar else None
        self.results = {}  # frame number -> list of words
        self.frames = 0
        FakeRecognizer.instances.append(self)

    def SetWords(self, enabled):
        self.words = enabled

    def AcceptWaveform(self, data):
        self.frames += 1
        return self.frames in self.results

    def Result(self):
        return json.dumps({"result": self.results[self.frames]})


class VoskDetectorTests(unittest.TestCase):
    def setUp(self):
        FakeRecognizer.instances.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model = Path(tmp.name) / "model"
        (self.model / "am").mkdir(parents=True)
        self.vosk = types.SimpleNamespace(Model=mock.Mock(), KaldiRecognizer=FakeRecognizer, SetLogLevel=mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"vosk": self.vosk})
        patcher.start()
        self.addCleanup(patcher.stop)
        missing = mock.patch.object(wake_word, "_vosk_missing_words", return_value=[])
        self.missing = missing.start()
        self.addCleanup(missing.stop)

    def detector(self, start="Hej Komputer", end="koniec nagrania", **kwargs):
        return wake_word.VoskKeywordDetector(start, end, model=str(self.model), **kwargs)

    @staticmethod
    def words(*items):
        return [{"word": w, "conf": c, "end": e} for w, c, e in items]

    def test_grammar_holds_only_the_two_phrases(self):
        self.detector()
        self.assertEqual(FakeRecognizer.instances[-1].grammar, ["hej komputer", "koniec nagrania", "[unk]"])
        self.missing.assert_called_once_with(self.vosk, self.vosk.Model.return_value, ["hej", "komputer", "koniec", "nagrania"])

    def test_confident_phrases_fire_with_their_lag(self):
        detector = self.detector()
        recognizer = FakeRecognizer.instances[-1]
        recognizer.results = {
            10: self.words(("hej", 1.0, 0.3), ("komputer", 0.95, 0.56)),        # 0.8 s fed
            20: self.words(("hej", 0.6, 1.2), ("komputer", 1.0, 1.4)),          # too unsure
            30: self.words(("koniec", 1.0, 1.9), ("nagrania", 1.0, 2.2)),
        }
        pcm = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        scores = [detector.score(pcm) for _ in range(30)]
        self.assertEqual(scores[9], (1.0, 0.0))
        self.assertEqual(scores[19], (0.0, 0.0))
        self.assertEqual(scores[29], (0.0, 1.0))
        self.assertAlmostEqual(detector.hit_lag["end"], 2.4 - 2.2)
        self.assertEqual(sum(1 for s in scores if any(s)), 2)

    def test_confidence_is_configurable(self):
        detector = self.detector(min_confidence=0.5)
        FakeRecognizer.instances[-1].results = {1: self.words(("hej", 0.6, 0.0), ("komputer", 0.6, 0.08))}
        self.assertEqual(detector.score(np.zeros(FRAME_SAMPLES, dtype=np.int16)), (1.0, 0.0))

    def test_unknown_words_and_overlapping_phrases_are_errors(self):
        self.missing.return_value = ["hejże"]
        with self.assertRaisesRegex(WakeWordError, "not in the model vocabulary: hejże"):
            self.detector("hejże komputer")
        self.missing.return_value = []
        with self.assertRaisesRegex(WakeWordError, "begin with"):
            self.detector("hej", "hej komputer")

    def test_missing_vosk_is_reported_as_wake_word_error(self):
        with mock.patch.dict(sys.modules, {"vosk": None}):
            with self.assertRaisesRegex(WakeWordError, "Vosk is not installed"):
                self.detector()


class ListenerBackfillTests(unittest.TestCase):
    def test_late_hits_are_dated_when_the_word_ended_and_backfill_the_capture(self):
        clock = Clock()
        events = []

        class LateDetector(ScriptedDetector):
            hit_lag = {}

            def score(self, frame):
                result = super().score(frame)
                self.hit_lag = {"start": 0.4} if result[0] else {}
                return result

        def on_event(kind, at):
            events.append((kind, at))
            listener.begin_capture(since=at)

        listener = WakeWordListener("a", "b", on_event, clock=clock)
        detector = LateDetector([(0.0, 0.0)] * 9 + [(1.0, 0.0)])
        for _ in range(10):
            clock.now += 0.08
            listener.process_block(frame(), detector)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][1], clock.now - 0.4)
        listener.process_block(frame(), detector)
        audio = listener.end_capture()
        self.assertEqual(len(audio), int(0.4 * SAMPLE_RATE) + FRAME_SAMPLES)

    def test_backfill_is_bounded_by_what_was_kept(self):
        clock = Clock()
        listener = WakeWordListener("a", "b", mock.Mock(), clock=clock, max_capture_seconds=60)
        detector = ScriptedDetector([])
        for _ in range(int(wake_word.BACKFILL_SECONDS / 0.08) + 50):
            clock.now += 0.08
            listener.process_block(frame(), detector)
        listener.begin_capture(since=clock.now - 100)
        self.assertLessEqual(len(listener.end_capture()), wake_word.BACKFILL_SECONDS * SAMPLE_RATE)
        listener.begin_capture()
        self.assertEqual(len(listener.end_capture()), 0)


if __name__ == "__main__":
    unittest.main()
