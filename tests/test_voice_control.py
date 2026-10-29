"""Voice control in the daemon: wake word events, command decisions, voice dictations."""

import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_suspend_resume_recovery import _import_main_isolated, app_global, patch_app_global


class FakeConfig:
    def __init__(self, **values):
        self.values = values

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def get_hallucination_markers(self):
        return ['blank audio']


class ImmediateThread:
    """threading.Thread stand-in that runs the target on start()."""

    def __init__(self, target=None, args=(), kwargs=None, **_ignored):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


class SyncThreading:
    Thread = ImmediateThread

    def __getattr__(self, name):
        return getattr(threading, name)


class VoiceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()
        cls.App = cls.main.hyprwhsprApp
        cls.outcome = app_global(cls.App, 'InjectionOutcome')
        plan_module = cls.App._voice_execute.__globals__
        cls.DictationPlan = plan_module['voice_command'].DictationPlan
        cls.VoiceDecision = plan_module['voice_command'].VoiceDecision
        cls.DecisionError = plan_module['DecisionError']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = patch_app_global(self.App, 'MODEL_UNLOADED_FILE', Path(self.tmp.name) / 'unloaded')
        patch.start()
        self.addCleanup(patch.stop)

    def app(self, **settings):
        app = self.App.__new__(self.App)
        app.config = FakeConfig(**settings)
        app._recording_lock = threading.Lock()
        app._voice_lock = threading.Lock()
        app._voice_state = 'idle'
        app._voice_generation = 0
        app._voice_plan = None
        app._voice_start_phrase = 'hey jarvis'
        app._voice_end_phrase = 'hey mycroft'
        app._voice_listener = mock.Mock()
        app._voice_listener.end_capture.return_value = np.full(16000, 0.1, dtype=np.float32)
        app.is_recording = app.is_processing = app._recording_starting = False
        app._longform_active = app._model_operation_active = False
        app._model_initializing = app._backend_init_failed = False
        app.audio_manager = mock.Mock()
        app._notify_user = mock.Mock()
        app.whisper_manager = mock.Mock(_model_manually_unloaded=False)
        app.whisper_manager.is_ready.return_value = True
        app._start_recording = mock.Mock()
        app._handle_recovery = mock.Mock(return_value=(True, 'ok'))
        return app

    def decide_with(self, decision=None, error=None):
        fake = types.SimpleNamespace(decide=mock.Mock(return_value=decision, side_effect=error))
        return patch_app_global(self.App, 'voice_command', fake), fake

    # ------------------------ wake word events ------------------------

    def test_start_word_opens_a_quiet_command_with_a_cue(self):
        app = self.app()
        app._on_wake_event('start', 1.0)
        self.assertEqual(app._voice_state, 'command')
        app._voice_listener.begin_capture.assert_called_once()
        app.audio_manager.play_cue_sound.assert_called_once_with('start')
        app._start_recording.assert_not_called()

    def test_start_word_is_ignored_while_recording_or_processing(self):
        for busy in ('is_recording', 'is_processing', '_longform_active'):
            app = self.app()
            setattr(app, busy, True)
            with self.subTest(busy=busy):
                app._on_wake_event('start', 1.0)
                self.assertEqual(app._voice_state, 'idle')
                app._voice_listener.begin_capture.assert_not_called()

    def test_cues_can_be_turned_off(self):
        app = self.app(voice_control_cues=False)
        app._on_wake_event('start', 1.0)
        app.audio_manager.play_cue_sound.assert_not_called()

    def test_end_word_closes_the_command_and_decides_in_the_background(self):
        app = self.app(voice_control_end_trim_seconds=0.8)
        app._on_wake_event('start', 1.0)
        app._run_voice_command = mock.Mock()
        with patch_app_global(self.App, 'threading', SyncThreading()), \
                mock.patch.object(self.App._voice_trim_seconds.__globals__['time'], 'monotonic', return_value=10.1):
            app._on_wake_event('end', 10.0)
        trim = app._voice_listener.end_capture.call_args.args[0]
        self.assertAlmostEqual(trim, 0.9)
        self.assertEqual(app._voice_state, 'deciding')
        app._run_voice_command.assert_called_once()
        app.audio_manager.play_cue_sound.assert_called_with('stop')

    def test_end_word_stops_any_running_recording_with_a_trim(self):
        app = self.app()
        app.is_recording = True
        app._stop_recording = mock.Mock()
        with patch_app_global(self.App, 'threading', SyncThreading()):
            app._on_wake_event('end', 0.0)
        self.assertGreaterEqual(app._stop_recording.call_args.kwargs['trim_tail_seconds'], 1.0)

    def test_end_word_in_continuous_mode_waits_for_flushes_first(self):
        app = self.app(recording_mode='continuous')
        app.is_recording = True
        calls = []
        app._continuous_stop_and_wait = lambda: calls.append('wait')
        app._stop_recording = lambda **kwargs: calls.append('stop')
        app._voice_stop_recording(0.0)
        self.assertEqual(calls, ['wait', 'stop'])

    def test_overlong_command_is_dropped(self):
        app = self.app()
        app._on_wake_event('start', 1.0)
        app._on_wake_event('capture_limit', 16.0)
        self.assertEqual(app._voice_state, 'idle')
        app._voice_listener.cancel_capture.assert_called_once()
        app.audio_manager.play_error_sound.assert_called_once()

    # ------------------------ manual control wins ------------------------

    def test_manual_start_and_cancel_drop_a_pending_command(self):
        for trigger in ('preempt', 'cancel'):
            app = self.app()
            app._on_wake_event('start', 1.0)
            with self.subTest(trigger=trigger):
                if trigger == 'preempt':
                    app._voice_preempt()
                else:
                    self.App._cancel_recording(app)
                self.assertEqual(app._voice_state, 'idle')
                self.assertEqual(app._voice_generation, 2)
                app._voice_listener.cancel_capture.assert_called_once()

    def test_record_cancel_drops_a_pending_command(self):
        app = self.app()
        app._on_wake_event('start', 1.0)
        app._handle_control_command('cancel')
        self.assertEqual(app._voice_state, 'idle')
        app._voice_listener.cancel_capture.assert_called_once()

    def test_only_manual_starts_preempt(self):
        app = self.app()
        app._start_recording = types.MethodType(self.App._start_recording, app)
        app._voice_preempt = mock.Mock()
        app.text_injector = mock.Mock()
        app.text_injector.stream_busy.return_value = False
        app.whisper_manager.realtime_client_missing.return_value = False
        app.whisper_manager.realtime_busy.return_value = False
        app._file_transcription_active = False
        app._start_settled = threading.Event()
        app._release_blocked_capture = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock(side_effect=RuntimeError('stop after gate'))
        for attr in ('_hide_mic_osd', '_stop_audio_level_monitoring', '_cancel_live_typing',
                     '_write_recording_status', '_restore_recording_playback'):
            setattr(app, attr, mock.Mock())
        plan = self.DictationPlan(output='scratchpad')
        app._start_recording(voice_plan=plan)
        app._voice_preempt.assert_not_called()
        self.assertIs(app._voice_plan, plan)

        app.is_recording = False
        app._start_recording()
        app._voice_preempt.assert_called_once()
        self.assertIsNone(app._voice_plan)

    def test_a_decision_overtaken_by_a_manual_start_does_nothing(self):
        app = self.app()
        app._on_wake_event('start', 1.0)
        with patch_app_global(self.App, 'threading', SyncThreading()):
            app._voice_finish_command(2.0)

        def transcribe(*args, **kwargs):
            app._voice_preempt()  # hotkey pressed while transcribing
            return 'dictate'
        app._voice_state = 'deciding'
        app.whisper_manager.transcribe_audio.side_effect = transcribe
        patch, fake = self.decide_with(self.VoiceDecision('dictate', 0.9))
        with patch:
            app._run_voice_command(app._voice_generation, np.zeros(10, dtype=np.float32))
        fake.decide.assert_not_called()
        app._start_recording.assert_not_called()

    # ------------------------ decisions ------------------------

    def run_command(self, app, transcript, decision=None, error=None):
        app._voice_state = 'deciding'
        app.whisper_manager.transcribe_audio.return_value = transcript
        patch, fake = self.decide_with(decision, error)
        real_plan_for = self.App._voice_execute.__globals__['plan_for']
        fake.plan_for = real_plan_for
        with patch:
            app._run_voice_command(app._voice_generation, np.zeros(16000, dtype=np.float32))
        self.assertEqual(app._voice_state, 'idle')
        return fake

    def test_dictation_decisions_start_a_recording_with_their_plan(self):
        app = self.app(voice_control_translation_language='fr')
        fake = self.run_command(app, 'Hey Jarvis, translate what I say. Hey Mycroft.',
                                self.VoiceDecision('translate', 0.9, spoken_language='de'))
        self.assertEqual(fake.decide.call_args.args[0], 'translate what I say.')
        app._start_recording.assert_called_once_with(
            language_override='de',
            voice_plan=self.DictationPlan(refine='translate', target_language='fr'))

    def test_start_word_is_ignored_while_the_decided_dictation_starts(self):
        app = self.app()
        seen = []

        def starting(**kwargs):
            seen.append(app._voice_state)
            app._on_wake_event('start', 5.0)  # said again before the recording is up
        app._start_recording.side_effect = starting
        self.run_command(app, 'dictate', self.VoiceDecision('dictate', 0.9))
        self.assertEqual(seen, ['acting'])
        app._voice_listener.begin_capture.assert_not_called()

    def test_recovery_decisions_paste_or_copy_the_last_dictation(self):
        app = self.app()
        app._handle_recovery.return_value = (False, 'No saved dictation in this daemon session')
        self.run_command(app, 'paste that again', self.VoiceDecision('paste_last', 0.95))
        app._handle_recovery.assert_called_once_with('paste_last')
        self.assertIn('No saved dictation', app._notify_user.call_args.args[1])
        app._start_recording.assert_not_called()

    def test_unclear_commands_are_reported_and_cancels_are_quiet(self):
        app = self.app()
        self.run_command(app, 'banana', self.VoiceDecision('nothing', 0.2, uncertain=True))
        self.assertIn("Didn't understand: banana", app._notify_user.call_args.args[1])
        app = self.app()
        self.run_command(app, 'never mind', self.VoiceDecision('nothing', 0.9))
        app._notify_user.assert_not_called()
        app._start_recording.assert_not_called()

    def test_empty_or_phantom_commands_never_reach_the_model(self):
        for transcript in ('', 'Hey Mycroft.', '[BLANK AUDIO]'):
            app = self.app()
            with self.subTest(transcript=transcript):
                fake = self.run_command(app, transcript, self.VoiceDecision('dictate', 0.9))
                fake.decide.assert_not_called()
                app.audio_manager.play_error_sound.assert_called_once()

    def test_decision_failures_notify_and_return_to_idle(self):
        app = self.app()
        self.run_command(app, 'dictate', error=self.DecisionError("no OpenRouter API key stored"))
        self.assertIn('API key', app._notify_user.call_args.args[1])
        app._start_recording.assert_not_called()

    def test_commands_wait_for_a_loaded_model(self):
        app = self.app()
        app.whisper_manager.is_ready.return_value = False
        fake = self.run_command(app, 'dictate', self.VoiceDecision('dictate', 0.9))
        app.whisper_manager.transcribe_audio.assert_not_called()
        fake.decide.assert_not_called()
        self.assertIn('not loaded', app._notify_user.call_args.args[1])

    # ------------------------ voice dictation delivery ------------------------

    def delivery_app(self, **settings):
        app = self.app(**settings)
        app._recording_control_server = mock.Mock()
        app._recording_control_server.has_capture_subscriber.return_value = False
        app.text_injector = mock.Mock()
        app.text_injector.inject_text.return_value = self.outcome.INJECTED
        app._mark_delivery_healthy = mock.Mock()
        return app

    def test_plain_dictations_keep_the_configured_refine_step(self):
        app = self.delivery_app()
        app._inject_text('hello', voice_plan=self.DictationPlan())
        app.text_injector.inject_text.assert_called_once_with('hello')

    def test_raw_dictations_skip_refinement(self):
        app = self.delivery_app()
        app._inject_text('um hello', voice_plan=self.DictationPlan(refine='off'))
        refine = app.text_injector.inject_text.call_args.kwargs['refine']
        self.assertEqual(refine('um hello'), 'um hello')

    def test_translated_dictations_translate_or_keep_the_original(self):
        app = self.delivery_app()
        app._inject_text('hello', voice_plan=self.DictationPlan(refine='translate', target_language='de'))
        refine = app.text_injector.inject_text.call_args.kwargs['refine']
        translate = mock.Mock(return_value='Hallo')
        with patch_app_global(self.App, 'voice_command', types.SimpleNamespace(translate_text=translate)):
            self.assertEqual(refine('hello'), 'Hallo')
            self.assertEqual(translate.call_args.args[:2], ('hello', 'de'))
            translate.side_effect = RuntimeError('HTTP 500')
            self.assertEqual(refine('hello'), 'hello')
        self.assertIn('Translation failed', app._notify_user.call_args.args[1])

    def test_scratchpad_dictations_are_saved_not_pasted(self):
        path = Path(self.tmp.name) / 'pad.md'
        app = self.delivery_app(voice_control_scratchpad_file=str(path))
        app.text_injector.prepare_text.return_value = 'Buy milk.'
        outcome = app._inject_text('buy milk', voice_plan=self.DictationPlan(output='scratchpad'))
        self.assertEqual(outcome, self.outcome.CONSUMED)
        app.text_injector.inject_text.assert_not_called()
        self.assertIn('Buy milk.', path.read_text(encoding='utf-8'))

    def test_scratchpad_write_failure_is_a_failed_delivery(self):
        app = self.delivery_app(voice_control_scratchpad_file=self.tmp.name)  # a directory
        app.text_injector.prepare_text.return_value = 'note'
        outcome = app._inject_text('note', voice_plan=self.DictationPlan(output='scratchpad'))
        self.assertEqual(outcome, self.outcome.FAILED)
        self.assertIn('scratchpad', app._notify_user.call_args.args[1])

    def test_capture_clients_still_get_the_raw_text(self):
        app = self.delivery_app()
        app._recording_control_server.has_capture_subscriber.return_value = True
        app._notify_capture = mock.Mock()
        app._inject_text('hello', voice_plan=self.DictationPlan(output='scratchpad'))
        app._notify_capture.assert_called_once_with('hello', final=True)

    def test_stop_hands_the_plan_and_trimmed_audio_to_processing(self):
        app = self.delivery_app()
        plan = self.DictationPlan(refine='off')
        app.is_recording = True
        app._voice_plan = plan
        app._recording_session = object()
        app._recording_finalizing = threading.Event()
        app._start_owner = threading.get_ident()
        app._ptt_press_time, app._ptt_locked = None, False
        app.audio_capture = mock.Mock(sample_rate=16000)
        app.audio_capture.stop_recording.return_value = np.full(32000, 0.1, dtype=np.float32)
        app._process_audio = mock.Mock()
        app._is_zero_volume = mock.Mock(return_value=False)
        for attr in ('_autostop_stop_silence_monitor', '_clear_mic_osd_preview_text', '_set_visualizer_state',
                     '_stop_audio_level_monitoring', '_write_recording_status', '_restore_recording_playback',
                     '_save_debug_recording', '_cancel_live_typing'):
            setattr(app, attr, mock.Mock())
        self.App._stop_recording(app, trim_tail_seconds=1.0)
        audio = app._process_audio.call_args.args[0]
        self.assertEqual(len(audio), 16000)
        self.assertEqual(app._process_audio.call_args.kwargs,
                         {'voice_plan': plan, 'end_phrase': 'hey mycroft'})
        self.assertIsNone(app._voice_plan)

    def test_trim_keeps_recordings_that_would_vanish(self):
        app = self.app()
        app.audio_capture = types.SimpleNamespace(sample_rate=16000)
        audio = np.zeros(16000 + 1000, dtype=np.float32)
        self.assertEqual(len(app._trim_audio_tail(audio, 1.0)), len(audio))
        self.assertEqual(len(app._trim_audio_tail(np.zeros(48000), 1.0)), 32000)
        self.assertIsNone(app._trim_audio_tail(None, 1.0))

    def test_processing_strips_the_end_word_from_the_transcript(self):
        app = self.delivery_app()
        app.audio_capture = types.SimpleNamespace(sample_rate=16000)
        app._current_language_override = None
        app.whisper_manager.transcribe_audio.return_value = 'Note this down. Hey Mycroft.'
        app._finish_live_typing = mock.Mock(return_value=None)
        app._inject_text = mock.Mock(return_value=self.outcome.INJECTED)
        for attr in ('_notify_capture', '_clear_mic_osd_preview_text', '_show_result_and_hide'):
            setattr(app, attr, mock.Mock())
        plan = self.DictationPlan()
        self.App._process_audio(app, np.zeros(10), voice_plan=plan, end_phrase='hey mycroft')
        app._inject_text.assert_called_once_with('Note this down.', voice_plan=plan)

    def test_voice_dictations_never_type_live(self):
        app = self.delivery_app(realtime_live_typing=True)
        app.whisper_manager.realtime_live_typing_supported.return_value = True
        self.assertTrue(app._live_typing_eligible())
        app._voice_plan = self.DictationPlan()
        self.assertFalse(app._live_typing_eligible())

    # ------------------------ startup ------------------------

    def test_listener_starts_only_where_voice_control_can_work(self):
        cases = (
            ({}, True, None),
            ({'recording_mode': 'long_form'}, True, 'long_form'),
            ({'transcription_backend': 'realtime-ws'}, True, 'realtime-ws'),
            ({}, False, 'raw ALSA'),
            ({'voice_control_end_word': 'Hey Jarvis'}, True, 'must differ'),
        )
        for extra, shared, reason in cases:
            app = self.app(voice_control_enabled=True, **extra)
            app._voice_listener = None
            app.audio_capture = mock.Mock(device_id=7)
            app.audio_capture.input_is_shared.return_value = shared
            listener = mock.Mock()
            with self.subTest(extra=extra, shared=shared), \
                    patch_app_global(self.App, 'WakeWordListener', listener):
                started = app._start_voice_control()
                if reason is None:
                    self.assertTrue(started)
                    args, kwargs = listener.call_args
                    self.assertEqual(args[:2], ('hey_jarvis', 'hey_mycroft'))
                    self.assertEqual(kwargs['device'], 7)
                else:
                    self.assertFalse(started)
                    listener.assert_not_called()
                    self.assertIn(reason, app._notify_user.call_args.args[1])

    def test_disabled_voice_control_starts_nothing(self):
        app = self.app()
        app._voice_listener = None
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            self.assertFalse(app._start_voice_control())
        listener.assert_not_called()
        app._stop_voice_control()  # no listener: a no-op


if __name__ == "__main__":
    unittest.main()
