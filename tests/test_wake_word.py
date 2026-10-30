import sys
import tempfile
import types
import unittest
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

    def test_a_single_word_detector_scores_only_its_word(self):
        model = mock.Mock()
        model.predict.return_value = {"hey_jarvis_v0.1": 0.9}
        fake = types.ModuleType("openwakeword.model")
        fake.Model = mock.Mock(return_value=model)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"openwakeword": types.ModuleType("openwakeword"),
                                              "openwakeword.model": fake}), \
                mock.patch.object(wake_word, "resolve_model_path",
                                  side_effect=lambda word, model_dir: Path(tmp) / f"{word}_v0.1.onnx"), \
                mock.patch.object(wake_word, "feature_model_paths",
                                  return_value=(Path(tmp) / "mel.onnx", Path(tmp) / "emb.onnx")):
            detector = wake_word.OpenWakeWordDetector("hey_jarvis", None)
        self.assertEqual(fake.Model.call_args.kwargs["wakeword_models"],
                         [str(Path(tmp) / "hey_jarvis_v0.1.onnx")])
        self.assertEqual(detector.score(np.zeros(FRAME_SAMPLES, dtype=np.int16)), (0.9, 0.0))

    def test_missing_openwakeword_is_reported_as_wake_word_error(self):
        with mock.patch.dict(sys.modules, {"openwakeword": None, "openwakeword.model": None}):
            with self.assertRaisesRegex(WakeWordError, "openWakeWord is not installed"):
                wake_word.OpenWakeWordDetector("hey_jarvis", "hey_mycroft")


if __name__ == "__main__":
    unittest.main()
