"""Long-form auto-pause on silence and the pause word."""

import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import longform_controller  # noqa: E402
from longform_controller import AUTO_PAUSE_PREROLL_S, LongFormController, SilenceTracker  # noqa: E402


def load_audio_capture():
    """audio_capture without a sound stack; the gate never touches sounddevice."""
    if 'audio_capture' in sys.modules:
        return sys.modules['audio_capture']
    with mock.patch.dict(sys.modules, {'sounddevice': sys.modules.get('sounddevice')
                                       or types.ModuleType('sounddevice')}):
        import audio_capture
    return audio_capture


class SilenceTrackerTests(unittest.TestCase):
    def feed(self, tracker, levels, held=False):
        return [tracker.step(level, held) for level in levels]

    def warm_up(self, tracker, level=0.0):
        self.feed(tracker, [level] * SilenceTracker.WARMUP_POLLS)

    def test_warm_up_is_ignored(self):
        tracker = SilenceTracker(0.1)
        polls = SilenceTracker.WARMUP_POLLS
        self.assertEqual(self.feed(tracker, [0.0] * polls), [None] * polls)
        self.assertIsNone(tracker.floor)

    def test_threshold_follows_the_quietest_recent_level(self):
        tracker = SilenceTracker(10.0)
        self.warm_up(tracker, 0.5)  # the start beep
        self.feed(tracker, [0.01, 0.002, 0.03])
        self.assertEqual(tracker.floor, 0.002)
        self.assertAlmostEqual(tracker.threshold, 0.002 * SilenceTracker.THRESHOLD_FACTOR)

    def test_a_too_quiet_start_ages_out(self):
        # The stream starts near-silent; the room is louder than 3x that
        tracker = SilenceTracker(2.0)
        self.warm_up(tracker)
        self.feed(tracker, [1e-4] * 5)
        actions = self.feed(tracker, [0.001] * 80)
        self.assertIn('hold', actions)
        self.assertLessEqual(actions.index('hold'), 50 + 20)
        self.assertEqual(tracker.floor, 0.001)

    def test_the_last_floor_seeds_a_resume(self):
        tracker = SilenceTracker(1.0, floor=0.003)
        self.assertAlmostEqual(tracker.threshold, 0.009)

    def test_digital_silence_says_nothing_about_the_room(self):
        tracker = SilenceTracker(1.0)
        self.warm_up(tracker)
        self.feed(tracker, [0.003, 0.0, 0.0])
        self.assertAlmostEqual(tracker.floor, 0.003)

    def test_threshold_has_a_minimum(self):
        tracker = SilenceTracker(1.0)
        self.assertEqual(tracker.threshold, SilenceTracker.MIN_THRESHOLD)

    def test_holds_after_the_quiet_and_releases_on_speech_after_the_ping(self):
        tracker = SilenceTracker(0.3, threshold=0.01)
        self.warm_up(tracker)
        self.assertIsNone(tracker.step(0.001, held=False))
        self.assertIsNone(tracker.step(0.05, held=False))  # speech resets the count
        self.assertEqual(self.feed(tracker, [0.001] * 3), [None, None, 'hold'])
        # The pause ping is loud but it is not speech
        self.assertEqual(self.feed(tracker, [0.05] * SilenceTracker.HOLD_GRACE_POLLS, held=True),
                         [None] * SilenceTracker.HOLD_GRACE_POLLS)
        self.assertIsNone(tracker.step(0.001, held=True))
        self.assertEqual(tracker.step(0.02, held=True), 'release')
        self.assertIsNone(tracker.step(0.02, held=False))


class AudioGateTests(unittest.TestCase):
    def capture(self, chunks):
        module = load_audio_capture()
        capture = module.AudioCapture.__new__(module.AudioCapture)
        capture.lock = threading.Lock()
        capture.sample_rate = 10
        capture.is_recording = True
        capture.audio_data = [np.full(n, float(i), dtype=np.float32) for i, n in enumerate(chunks)]
        capture._buffered_samples = sum(chunks)
        capture._reset_gate_locked()
        return capture

    def test_hold_trims_the_silence_and_release_restores_the_preroll(self):
        capture = self.capture([10, 10, 10])
        self.assertTrue(capture.hold_audio(trim_tail_seconds=1.5, preroll_seconds=0.5))
        self.assertTrue(capture.gate_held)
        self.assertEqual([len(c) for c in capture.audio_data], [10, 5])
        self.assertEqual(capture._buffered_samples, 15)

        with capture.lock:
            for value in range(6):  # 18 held samples; the newest chunks covering 0.5 s stay
                capture._gate_chunk_locked(np.full(3, 100.0 + value, dtype=np.float32))
        self.assertAlmostEqual(capture.release_audio(), 0.6)
        self.assertFalse(capture.gate_held)
        self.assertEqual([c[0] for c in capture.audio_data[2:]], [104.0, 105.0])
        self.assertEqual(capture._buffered_samples, 21)
        self.assertEqual(capture.release_audio(), 0.0)

    def test_hold_needs_a_running_recording(self):
        capture = self.capture([10])
        capture.is_recording = False
        self.assertFalse(capture.hold_audio(1.0))
        self.assertFalse(capture.gate_held)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ControllerTests(unittest.TestCase):
    def controller(self, **settings):
        settings.setdefault('long_form_auto_pause_seconds', 2.0)
        settings.setdefault('long_form_auto_save_interval', 0)
        self.threads = []
        self.clock = Clock()
        audio_capture = SimpleNamespace(
            sample_rate=16000,
            rolling_avg_level=0.0,
            start_recording=mock.Mock(return_value=True),
            resume_recording=mock.Mock(return_value=True),
            pause_recording=mock.Mock(return_value=np.ones(32000, dtype=np.float32)),
            stop_recording=mock.Mock(),
            hold_audio=mock.Mock(return_value=True),
            release_audio=mock.Mock(return_value=AUTO_PAUSE_PREROLL_S),
        )
        segment_manager = SimpleNamespace(
            start_session=mock.Mock(), save_segment=mock.Mock(return_value=Path('seg.wav')),
            clear_session=mock.Mock(), get_total_size=mock.Mock(return_value=0), max_size_bytes=1,
        )

        def thread_factory(**kwargs):
            thread = mock.Mock()
            self.threads.append(kwargs)
            return thread

        controller = LongFormController(
            config=SimpleNamespace(get_setting=lambda key, default=None: settings.get(key, default)),
            audio_capture=audio_capture,
            audio_manager=SimpleNamespace(play_start_sound=mock.Mock(), play_stop_sound=mock.Mock(),
                                          play_error_sound=mock.Mock(), play_cue_sound=mock.Mock()),
            whisper_manager=SimpleNamespace(),
            inject_text=mock.Mock(), notify_capture=mock.Mock(), set_visualizer_state=mock.Mock(),
            show_mic_osd=mock.Mock(), hide_mic_osd=mock.Mock(), show_result_and_hide=mock.Mock(),
            write_recording_status=mock.Mock(), set_processing=mock.Mock(), hallucination_markers=set(),
            segment_manager_factory=lambda max_size_mb: segment_manager,
            thread_factory=thread_factory, clock=self.clock,
            state_file=Path('/nonexistent/hyprwhspr-test/longform_state'),
        )
        controller.write_state = mock.Mock()
        return controller

    def started(self, **settings):
        controller = self.controller(**settings)
        controller.primary_shortcut()
        self.assertEqual(controller.state, 'RECORDING')
        return controller

    def step(self, controller, action, seconds=2.0):
        generation = self.threads[-1]['args'][0]
        tracker = mock.Mock(threshold=0.01, level=0.001, floor=0.003)
        tracker.step.return_value = action
        return controller.auto_pause_step(generation, tracker, seconds)

    def test_off_by_default(self):
        controller = self.started(long_form_auto_pause_seconds=0)
        self.assertEqual(self.threads, [])
        controller.audio_capture.hold_audio.assert_not_called()

    def test_silence_holds_with_indicator_and_speech_resumes(self):
        controller = self.started()
        self.assertEqual(len(self.threads), 1)

        self.assertTrue(self.step(controller, 'hold'))
        controller.audio_capture.hold_audio.assert_called_once_with(
            trim_tail_seconds=1.5, preroll_seconds=AUTO_PAUSE_PREROLL_S)
        self.assertTrue(controller.auto_paused)
        self.assertEqual(controller.state, 'RECORDING')  # the session itself stays open
        controller.set_visualizer_state.assert_called_with('paused')
        controller.audio_manager.play_cue_sound.assert_called_with('stop')

        self.assertTrue(self.step(controller, 'release'))
        controller.audio_capture.release_audio.assert_called_once_with()
        self.assertFalse(controller.auto_paused)
        controller.set_visualizer_state.assert_called_with('recording')
        controller.audio_manager.play_cue_sound.assert_called_with('start')

    def test_a_failed_hold_changes_nothing(self):
        controller = self.started()
        controller.audio_capture.hold_audio.return_value = False
        self.step(controller, 'hold')
        self.assertFalse(controller.auto_paused)
        controller.audio_manager.play_cue_sound.assert_not_called()

    def test_a_manual_pause_retires_the_monitor_and_resume_starts_a_new_one(self):
        controller = self.started()
        self.step(controller, 'hold')
        controller.primary_shortcut()  # pause
        self.assertEqual(controller.state, 'PAUSED')
        self.assertFalse(controller.auto_paused)
        self.assertFalse(self.step(controller, 'release'))
        controller.audio_capture.release_audio.assert_not_called()

        controller.primary_shortcut()  # resume
        self.assertEqual(len(self.threads), 2)
        self.assertTrue(self.step(controller, None))

    def test_the_noise_floor_carries_over_to_a_resume(self):
        controller = self.started()
        self.step(controller, None)
        controller.primary_shortcut()
        controller.primary_shortcut()
        generation, stop_event, seconds = self.threads[-1]['args']
        stop_event.set()
        with mock.patch.object(longform_controller, 'SilenceTracker') as tracker:
            controller._auto_pause_loop(generation, stop_event, seconds)
        tracker.assert_called_once_with(2.0, 0.0, floor=0.003)

    def test_monitor_retired_on_submit_and_cancel(self):
        for finish in ('cancel_shortcut', 'submit_shortcut'):
            with self.subTest(finish=finish):
                controller = self.started()
                controller.whisper_manager.transcribe_audio = mock.Mock(return_value='')
                controller.segment_manager.concatenate_all = mock.Mock(return_value=None)
                controller.segment_manager.has_segments = mock.Mock(return_value=False)
                getattr(controller, finish)()
                self.assertFalse(self.step(controller, 'hold'))
                controller.audio_capture.hold_audio.assert_not_called()

    # ------------------------ pause word ------------------------

    def saved_samples(self, controller):
        return len(controller.segment_manager.save_segment.call_args.args[0])

    def test_pause_word_pauses_without_its_own_audio_and_resumes(self):
        controller = self.started()
        self.clock.now = 100.25
        controller.pause_word(100.0, trim_seconds=1.0)
        self.assertEqual(controller.state, 'PAUSED')
        self.assertEqual(self.saved_samples(controller), 32000 - int(1.25 * 16000))
        controller.audio_manager.play_stop_sound.assert_called_once_with()

        controller.pause_word(103.0)
        self.assertEqual(controller.state, 'RECORDING')
        controller.audio_capture.resume_recording.assert_called_once_with()

    def test_pause_word_during_auto_pause_keeps_the_buffer_whole(self):
        controller = self.started()
        self.step(controller, 'hold')
        controller.pause_word(self.clock.now)
        self.assertEqual(self.saved_samples(controller), 32000)

    def test_pause_word_that_resumed_an_auto_pause_is_cut_with_its_preroll(self):
        controller = self.started()
        self.step(controller, 'hold')
        self.clock.now = 200.0
        self.step(controller, 'release')  # the word's first syllable
        self.clock.now = 200.9
        controller.pause_word(200.8, trim_seconds=1.0)
        trimmed = int((AUTO_PAUSE_PREROLL_S + 0.9) * 16000)
        self.assertEqual(self.saved_samples(controller), 32000 - trimmed)

    def test_pause_word_long_after_a_resume_cuts_only_the_word(self):
        controller = self.started()
        self.step(controller, 'hold')
        self.clock.now = 200.0
        self.step(controller, 'release')
        self.clock.now = 210.0
        controller.pause_word(210.0, trim_seconds=1.0)
        self.assertEqual(self.saved_samples(controller), 16000)

    def test_pause_word_does_not_start_a_session(self):
        controller = self.controller()
        controller.pause_word(100.0)
        self.assertEqual(controller.state, 'IDLE')
        controller.audio_capture.start_recording.assert_not_called()


if __name__ == "__main__":
    unittest.main()
