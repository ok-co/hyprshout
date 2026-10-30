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


class VoiceAppTestCase(unittest.TestCase):
    """An app with voice control state and mocked collaborators."""

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

    def delivery_app(self, **settings):
        app = self.app(**settings)
        app._recording_control_server = mock.Mock()
        app._recording_control_server.has_capture_subscriber.return_value = False
        app.text_injector = mock.Mock()
        app.text_injector.inject_text.return_value = self.outcome.INJECTED
        app._mark_delivery_healthy = mock.Mock()
        return app


class VoiceControlTests(VoiceAppTestCase):
    # ------------------------ wake word events ------------------------

    def test_start_word_opens_a_quiet_command_with_a_cue(self):
        app = self.app()
        app._on_wake_event('start', 1.0)
        self.assertEqual(app._voice_state, 'command')
        app._voice_listener.begin_capture.assert_called_once()
        app.audio_manager.play_cue_sound.assert_called_once_with('start')
        app._start_recording.assert_not_called()

    def test_start_word_is_ignored_while_processing_or_in_long_form(self):
        for busy in ('is_processing', '_longform_active'):
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
        app._stop_scratchpad_speech = mock.Mock()
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
        plan = self.DictationPlan(route='scratchpad')
        app._start_recording(route='scratchpad', voice_plan=plan)
        app._voice_preempt.assert_not_called()
        self.assertIs(app._voice_plan, plan)
        self.assertEqual(app._current_route, 'scratchpad')

        app.is_recording = False
        app._start_recording()
        app._voice_preempt.assert_called_once()
        self.assertIsNone(app._voice_plan)
        # Every start stops the draft being read aloud: the microphone would hear it
        self.assertEqual(app._stop_scratchpad_speech.call_count, 2)

    def test_every_start_silences_the_assistant(self):
        app = self.app()
        app._start_recording = types.MethodType(self.App._start_recording, app)
        app._voice_preempt = mock.Mock()
        app._stop_scratchpad_speech = mock.Mock()
        app._stop_assistant_speech = mock.Mock()
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
        app._start_recording(route='assistant', voice_plan=self.DictationPlan(route='assistant'))
        app._stop_assistant_speech.assert_called_once_with()
        self.assertEqual(app._current_route, 'assistant')

    def test_the_start_word_stops_the_draft_being_read_aloud(self):
        app = self.app()
        app._scratchpad_speech = mock.Mock()
        app._on_wake_event('start', 1.0)
        app._scratchpad_speech.stop.assert_called_once_with()

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

    def test_dictation_decisions_start_a_recording_on_their_route(self):
        app = self.app()
        fake = self.run_command(app, 'Hey Jarvis, translate what I say into French. Hey Mycroft.',
                                self.VoiceDecision('translate', 0.9, spoken_language='de', target_language='fr'))
        self.assertEqual(fake.decide.call_args.args[0], 'translate what I say into French.')
        app._start_recording.assert_called_once_with(
            language_override='de', route='translate',
            voice_plan=self.DictationPlan(route='translate', translate_language='French'))

        app = self.app()
        self.run_command(app, 'take a note', self.VoiceDecision('scratchpad', 0.9))
        app._start_recording.assert_called_once_with(
            language_override=None, route='scratchpad', voice_plan=self.DictationPlan(route='scratchpad'))

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

    def test_scratchpad_decisions_run_the_draft_requests(self):
        for action in ('scratchpad_paste', 'scratchpad_copy', 'scratchpad_undo', 'scratchpad_clear'):
            app = self.app()
            app._handle_scratchpad_request = mock.Mock(return_value=(True, 'done'))
            with self.subTest(action=action):
                self.run_command(app, 'do it', self.VoiceDecision(action, 0.9))
                app._handle_scratchpad_request.assert_called_once_with(action)
                app._notify_user.assert_not_called()
                app._start_recording.assert_not_called()
        app = self.app()
        app._handle_scratchpad_request = mock.Mock(return_value=(False, 'The draft is empty'))
        self.run_command(app, 'paste the draft', self.VoiceDecision('scratchpad_paste', 0.9))
        self.assertIn('The draft is empty', app._notify_user.call_args.args[1])

    def test_unclear_commands_are_reported_and_cancels_are_quiet(self):
        app = self.app()
        self.run_command(app, 'banana', self.VoiceDecision('nothing', 0.2, uncertain=True))
        self.assertIn("Didn't understand: banana", app._notify_user.call_args.args[1])
        app = self.app()
        self.run_command(app, 'hmm, actually leave it', self.VoiceDecision('nothing', 0.9))
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
        self.run_command(app, 'start an email to Anna', error=self.DecisionError("no OpenRouter API key stored"))
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

    def test_manual_dictations_pass_only_their_route(self):
        app = self.delivery_app()
        app._inject_text('hello', route='translate')
        app.text_injector.inject_text.assert_called_once_with('hello', translate=True)

    def test_voice_plans_add_raw_and_the_translation_language(self):
        cases = (
            ('paste', self.DictationPlan(), {'translate': False}),
            ('paste', self.DictationPlan(raw=True), {'translate': False, 'raw': True}),
            ('translate', self.DictationPlan(route='translate', translate_language='German'),
             {'translate': True, 'translate_language': 'German'}),
        )
        for route, plan, expected in cases:
            app = self.delivery_app()
            with self.subTest(plan=plan):
                app._inject_text('hello', route=route, voice_plan=plan)
                app.text_injector.inject_text.assert_called_once_with('hello', **expected)

    def test_scratchpad_dictations_edit_the_draft(self):
        app = self.delivery_app()
        app._scratchpad_apply = mock.Mock(return_value=self.outcome.CONSUMED)
        outcome = app._inject_text('buy milk', route='scratchpad', voice_plan=self.DictationPlan(route='scratchpad'))
        self.assertEqual(outcome, self.outcome.CONSUMED)
        app._scratchpad_apply.assert_called_once_with('buy milk')
        app.text_injector.inject_text.assert_not_called()

    def test_capture_clients_still_get_the_raw_text(self):
        app = self.delivery_app()
        app._recording_control_server.has_capture_subscriber.return_value = True
        app._notify_capture = mock.Mock()
        app._inject_text('hello', route='scratchpad', voice_plan=self.DictationPlan(route='scratchpad'))
        app._notify_capture.assert_called_once_with('hello', final=True)

    def test_stop_hands_the_plan_and_trimmed_audio_to_processing(self):
        app = self.delivery_app()
        plan = self.DictationPlan(raw=True)
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
        app._current_route = 'paste'
        app.whisper_manager.transcribe_audio.return_value = 'Note this down. Hey Mycroft.'
        app._finish_live_typing = mock.Mock(return_value=None)
        app._inject_text = mock.Mock(return_value=self.outcome.INJECTED)
        for attr in ('_notify_capture', '_clear_mic_osd_preview_text', '_show_result_and_hide'):
            setattr(app, attr, mock.Mock())
        plan = self.DictationPlan()
        self.App._process_audio(app, np.zeros(10), voice_plan=plan, end_phrase='hey mycroft')
        app._inject_text.assert_called_once_with('Note this down.', route='paste', voice_plan=plan)

    def test_voice_dictations_never_type_live(self):
        app = self.delivery_app(realtime_live_typing=True)
        app._current_route = 'paste'
        app.whisper_manager.realtime_live_typing_supported.return_value = True
        self.assertTrue(app._live_typing_eligible())
        app._voice_plan = self.DictationPlan()
        self.assertFalse(app._live_typing_eligible())

    # ------------------------ voice assistant ------------------------

    def test_ask_decisions_answer_the_command_aloud(self):
        app = self.app()
        seen = []

        def respond(text, still_wanted):
            seen.append((text, app._voice_state, still_wanted()))
            app._voice_cancel_command("cancel requested")  # cancel while the LLM runs
            seen.append(still_wanted())
            return True
        app._assistant_respond = mock.Mock(side_effect=respond)
        self.run_command(app, 'Hey Jarvis, what is the capital of Peru?',
                         self.VoiceDecision('ask', 0.9))
        self.assertEqual(seen, [('what is the capital of Peru?', 'acting', True), False])
        app._start_recording.assert_not_called()
        app._notify_user.assert_not_called()

    def test_assistant_decisions_record_the_question(self):
        app = self.app()
        self.run_command(app, 'Jarvis, I need you', self.VoiceDecision('assistant', 0.9))
        app._start_recording.assert_called_once_with(
            language_override=None, route='assistant', voice_plan=self.DictationPlan(route='assistant'))

    def test_assistant_dictations_go_to_the_assistant(self):
        for answered, outcome in ((True, self.outcome.CONSUMED), (False, self.outcome.FAILED)):
            app = self.delivery_app()
            app._assistant_respond = mock.Mock(return_value=answered)
            with self.subTest(answered=answered):
                result = app._inject_text('tell me a joke', route='assistant',
                                          voice_plan=self.DictationPlan(route='assistant'))
                self.assertEqual(result, outcome)
                app._assistant_respond.assert_called_once_with('tell me a joke')
                app.text_injector.inject_text.assert_not_called()

    def test_the_start_word_and_cancel_silence_the_assistant(self):
        app = self.app()
        app._assistant_speech = mock.Mock()
        app._on_wake_event('start', 1.0)
        app._assistant_speech.stop.assert_called_once_with()

        app = self.app()
        app._assistant_speech = mock.Mock()
        self.App._cancel_recording(app)
        app._assistant_speech.stop.assert_called_once_with()

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

    def test_engine_settings_reach_the_listener(self):
        app = self.app(voice_control_enabled=True, voice_control_engine=' Sherpa ',
                       voice_control_start_word='hey computer', voice_control_end_word='over and out',
                       voice_control_keyword_threshold=0.4, voice_control_keyword_boost=2.0)
        app._voice_listener = None
        app.audio_capture = mock.Mock(device_id=None)
        listener = mock.Mock()
        with patch_app_global(self.App, 'WakeWordListener', listener):
            self.assertTrue(app._start_voice_control())
        args, kwargs = listener.call_args
        self.assertEqual(args[:2], ('hey_computer', 'over_and_out'))
        self.assertEqual((kwargs['engine'], kwargs['keyword_threshold'], kwargs['keyword_boost']),
                         ('sherpa', 0.4, 2.0))
        self.assertEqual((kwargs['vosk_model'], kwargs['vosk_confidence']), ('en-us', 0.9))
        self.assertEqual((app._voice_start_phrase, app._voice_end_phrase), ('hey computer', 'over and out'))

    def test_unknown_engine_disables_voice_control(self):
        app = self.app(voice_control_enabled=True, voice_control_engine='porcupine')
        app._voice_listener = None
        app.audio_capture = mock.Mock(device_id=None)
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            self.assertFalse(app._start_voice_control())
        listener.assert_not_called()
        self.assertIn('porcupine', app._notify_user.call_args.args[1])

    def test_long_form_mode_listens_for_the_pause_word_only(self):
        for enabled in (False, True):
            app = self.app(recording_mode='long_form', long_form_pause_word='Hey Rhasspy',
                           voice_control_enabled=enabled)
            app._voice_listener = None
            app.audio_capture = mock.Mock(device_id=7)
            app.audio_capture.input_is_shared.return_value = True
            with self.subTest(voice_control_enabled=enabled), \
                    patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
                self.assertTrue(app._start_voice_control())
                args, kwargs = listener.call_args
                self.assertEqual(args[:2], ('hey_rhasspy', None))
                self.assertEqual(args[2], app._on_pause_word)
                self.assertEqual(kwargs['device'], 7)

    def test_pause_word_needs_a_word_and_a_shared_microphone(self):
        for word, shared in (('', True), ('alexa', False)):
            app = self.app(recording_mode='long_form', long_form_pause_word=word)
            app._voice_listener = None
            app.audio_capture = mock.Mock(device_id=7)
            app.audio_capture.input_is_shared.return_value = shared
            with self.subTest(word=word, shared=shared), \
                    patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
                self.assertFalse(app._start_voice_control())
                listener.assert_not_called()

    def test_pause_word_toggles_the_long_form_session(self):
        app = self.app(voice_control_end_trim_seconds=0.8)
        app._longform = mock.Mock()
        with patch_app_global(self.App, 'threading', SyncThreading()):
            app._on_pause_word('start', 5.0)
            app._on_pause_word('end', 6.0)
        app._longform.pause_word.assert_called_once_with(5.0, 0.8)

    def test_pause_word_uses_the_configured_engine(self):
        app = self.app(recording_mode='long_form', long_form_pause_word='przerwa',
                       voice_control_engine='vosk', voice_control_vosk_model='pl')
        app._voice_listener = None
        app.audio_capture = mock.Mock(device_id=None)
        app.audio_capture.input_is_shared.return_value = True
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            self.assertTrue(app._start_voice_control())
        kwargs = listener.call_args.kwargs
        self.assertEqual((kwargs['engine'], kwargs['vosk_model']), ('vosk', 'pl'))
        app = self.app(recording_mode='long_form', long_form_pause_word='przerwa', voice_control_engine='nope')
        app._voice_listener = None
        app.audio_capture = mock.Mock(device_id=None)
        app.audio_capture.input_is_shared.return_value = True
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            self.assertFalse(app._start_voice_control())
        listener.assert_not_called()

    def test_disabled_voice_control_starts_nothing(self):
        app = self.app()
        app._voice_listener = None
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            self.assertFalse(app._start_voice_control())
        listener.assert_not_called()
        app._stop_voice_control()  # no listener: a no-op


class VoicePhraseAndDictationTests(VoiceAppTestCase):
    """Spoken command phrases, pause-ended commands and the start word inside a dictation."""

    def phrase_app(self, **settings):
        app = self.app(**settings)
        app._handle_scratchpad_request = mock.Mock(return_value=(True, 'Pasted'))
        app._scratchpad = mock.Mock()
        app._scratchpad.read.return_value = 'Fix the parser.'
        app._scratchpad_apply = mock.Mock(return_value=self.outcome.CONSUMED)
        app._stop_scratchpad_speech = mock.Mock()
        app._stop_assistant_speech = mock.Mock()
        return app

    def say(self, app, *events):
        with patch_app_global(self.App, 'threading', SyncThreading()):
            for kind in events:
                app._on_wake_event(kind, 1.0)

    # ------------------------ phrases after the start word ------------------------

    def test_a_spotted_phrase_runs_at_once_without_transcription(self):
        app = self.phrase_app()
        self.say(app, 'start', 'phrase:send the draft')
        app._handle_scratchpad_request.assert_called_once_with('scratchpad_paste')
        app.whisper_manager.transcribe_audio.assert_not_called()
        app._voice_listener.cancel_capture.assert_called_once()
        self.assertEqual(app._voice_state, 'idle')

    def test_phrases_count_only_inside_a_command(self):
        app = self.phrase_app()
        self.say(app, 'phrase:send the draft', 'phrase:cancel')
        app._handle_scratchpad_request.assert_not_called()
        app._notify_user.assert_not_called()

    def test_cancel_and_help_need_no_model(self):
        app = self.phrase_app()
        self.say(app, 'start', 'phrase:never mind')
        app._notify_user.assert_called_once_with('hyprwhspr voice', 'Cancelled', urgency='low')
        app = self.phrase_app(voice_control_announce=False)
        self.say(app, 'start', 'phrase:never mind')
        app._notify_user.assert_not_called()
        app = self.phrase_app()
        self.say(app, 'start', 'phrase:what can i say')
        self.assertIn('send the draft', app._notify_user.call_args.args[1])

    def test_draft_phrases_edit_the_draft(self):
        app = self.phrase_app(llm_translate_language='Polish', voice_control_languages=['de'])
        self.say(app, 'start', 'phrase:translate the draft')
        app._scratchpad_apply.assert_called_once_with('Translate the whole draft into Polish.')
        app = self.phrase_app(voice_control_languages=['de'])
        self.say(app, 'start', 'phrase:translate the draft into german')
        app._scratchpad_apply.assert_called_once_with('Translate the whole draft into German.')
        app = self.phrase_app()
        app._scratchpad.read.return_value = '  '
        self.say(app, 'start', 'phrase:polish the draft')
        app._scratchpad_apply.assert_not_called()
        self.assertIn('empty', app._notify_user.call_args.args[1])

    def test_dictation_phrases_start_the_dictation_and_say_how_to_finish(self):
        app = self.phrase_app(voice_control_languages=['fr'])
        self.say(app, 'start', 'phrase:translate into french')
        app._start_recording.assert_called_once_with(
            language_override=None, route='translate',
            voice_plan=self.DictationPlan(route='translate', translate_language='French'))
        message = app._notify_user.call_args.args[1]
        self.assertIn('translated into French', message)
        self.assertIn('say "hey jarvis" when you\'re done', message)

    def test_a_transcribed_phrase_skips_the_decision_model(self):
        app = self.phrase_app()
        fake = self.run_command(app, 'Undo that.', self.VoiceDecision('dictate', 0.9))
        fake.decide.assert_not_called()
        app._handle_scratchpad_request.assert_called_once_with('scratchpad_undo')

    def test_draft_instructions_from_the_decision_model_edit_the_draft(self):
        app = self.phrase_app()
        self.run_command(app, 'turn the draft into three bullet points', self.VoiceDecision('scratchpad_edit', 0.8))
        app._scratchpad_apply.assert_called_once_with('turn the draft into three bullet points')
        app._start_recording.assert_not_called()

    # ------------------------ how a command ends ------------------------

    def test_commands_watch_for_their_pause_unless_turned_off(self):
        app = self.phrase_app(voice_control_pause_seconds=2.0, voice_control_silence_seconds=6)
        self.say(app, 'start')
        self.assertEqual(app._voice_listener.begin_capture.call_args.kwargs,
                         {'since': 1.0, 'silence_seconds': 6.0, 'pause_seconds': 2.0})
        app = self.phrase_app(voice_control_end_on_pause=False)
        self.say(app, 'start')
        self.assertNotIn('pause_seconds', app._voice_listener.begin_capture.call_args.kwargs)

    def test_the_pause_ends_a_command_without_a_trim(self):
        app = self.phrase_app()
        app._run_voice_command = mock.Mock()
        self.say(app, 'start', 'pause')
        app._voice_listener.end_capture.assert_called_once_with(0.0)
        app._run_voice_command.assert_called_once()
        self.assertEqual(app._voice_state, 'deciding')

    def test_silence_or_the_start_word_again_cancels(self):
        for second in ('silence', 'start'):
            app = self.phrase_app()
            with self.subTest(second=second):
                self.say(app, 'start', second)
                self.assertEqual(app._voice_state, 'idle')
                app._voice_listener.cancel_capture.assert_called_once()
                app.audio_manager.play_cue_sound.assert_called_with('stop')
                app.whisper_manager.transcribe_audio.assert_not_called()

    # ------------------------ who transcribes ------------------------

    def test_vosk_transcribes_when_the_backend_cannot(self):
        app = self.phrase_app()
        app._voice_listener.can_transcribe.return_value = True
        app._voice_listener.transcribe.return_value = 'send the draft'
        app.whisper_manager.is_ready.return_value = False
        self.run_command(app, 'unused')
        app.whisper_manager.transcribe_audio.assert_not_called()
        app._handle_scratchpad_request.assert_called_once_with('scratchpad_paste')

        app = self.phrase_app(voice_control_command_transcriber='backend')
        app._voice_listener.can_transcribe.return_value = True
        app.whisper_manager.is_ready.return_value = False
        self.run_command(app, 'unused')
        app._voice_listener.transcribe.assert_not_called()
        self.assertIn('not loaded', app._notify_user.call_args.args[1])

        app = self.phrase_app(voice_control_command_transcriber='vosk')
        app._voice_listener.can_transcribe.return_value = True
        app._voice_listener.transcribe.return_value = 'undo'
        self.run_command(app, 'unused')
        app.whisper_manager.transcribe_audio.assert_not_called()

    def test_realtime_ws_works_with_vosk_commands(self):
        app = self.app(transcription_backend='realtime-ws', voice_control_engine='vosk')
        app.audio_capture = mock.Mock()
        self.assertIsNone(app._voice_control_unsupported())
        app = self.app(transcription_backend='realtime-ws', voice_control_engine='vosk',
                       voice_control_command_transcriber='backend')
        app.audio_capture = mock.Mock()
        self.assertIn('realtime-ws', app._voice_control_unsupported())

    def test_vosk_listens_for_the_phrases_of_its_language(self):
        app = self.app(voice_control_enabled=True, voice_control_engine='vosk', voice_control_vosk_model='pl',
                       voice_control_start_word='hej komputer', voice_control_end_word='koniec nagrania')
        app._voice_listener = None
        app.audio_capture = mock.Mock(device_id=None)
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            self.assertTrue(app._start_voice_control())
        phrases = listener.call_args.kwargs['phrases']
        self.assertIn('wyślij szkic', phrases)
        self.assertNotIn('send the draft', phrases)
        app = self.app(voice_control_enabled=True, voice_control_engine='sherpa',
                       voice_control_start_word='hey computer', voice_control_end_word='over and out')
        app._voice_listener = None
        app.audio_capture = mock.Mock(device_id=None)
        with patch_app_global(self.App, 'WakeWordListener', mock.Mock()) as listener:
            app._start_voice_control()
        self.assertNotIn('phrases', listener.call_args.kwargs)
        self.assertEqual(app._voice_phrases().match_command('abbrechen'), ('cancel', None))

    # ------------------------ the start word inside a dictation ------------------------

    def recording_app(self, **settings):
        app = self.phrase_app(**settings)
        app.is_recording = True
        app._current_route = 'paste'
        app._stop_recording = mock.Mock()
        app._cancel_recording = mock.Mock()
        return app

    def test_the_start_word_in_a_dictation_opens_a_window(self):
        app = self.recording_app(voice_control_pause_seconds=1.2)
        self.say(app, 'start')
        self.assertEqual(app._voice_state, 'dictation')
        self.assertEqual(app._voice_listener.begin_capture.call_args.kwargs,
                         {'since': 1.0, 'pause_seconds': 1.2, 'silence_seconds': 1.2})
        app._stop_scratchpad_speech.assert_not_called()

    def test_no_window_where_the_dictation_cannot_be_steered(self):
        for settings, extra in (({'voice_control_in_dictation': False}, {}),
                                ({'recording_mode': 'continuous'}, {}),
                                ({}, {'_live_typing': object()})):
            app = self.recording_app(**settings)
            for name, value in extra.items():
                setattr(app, name, value)
            with self.subTest(settings=settings, extra=extra):
                self.say(app, 'start')
                self.assertEqual(app._voice_state, 'idle')
                app._voice_listener.begin_capture.assert_not_called()

    def test_saying_nothing_or_the_start_word_again_finishes(self):
        for second in ('silence', 'start', 'end', 'phrase:done', 'capture_limit'):
            app = self.recording_app()
            with self.subTest(second=second):
                self.say(app, 'start', second)
                self.assertEqual(app._voice_state, 'idle')
                kwargs = app._stop_recording.call_args.kwargs
                self.assertEqual(kwargs['tail_phrase'], 'hey jarvis')
                self.assertGreaterEqual(kwargs['trim_tail_seconds'], 1.0)
                self.assertEqual(app._current_route, 'paste')

    def test_cancel_drops_the_dictation(self):
        app = self.recording_app()
        self.say(app, 'start', 'phrase:scrap that')
        app._cancel_recording.assert_called_once_with()
        app._stop_recording.assert_not_called()
        self.assertEqual(app._notify_user.call_args.args[1], 'Dictation discarded')

    def test_routing_phrases_send_the_dictation_elsewhere(self):
        for phrase, route in (('into the draft', 'scratchpad'), ('translate it', 'translate'), ('ask', 'assistant')):
            app = self.recording_app()
            with self.subTest(phrase=phrase):
                self.say(app, 'start', f'phrase:{phrase}')
                self.assertEqual(app._current_route, route)
                app._stop_recording.assert_called_once()

    def test_raw_and_submit_change_the_plan(self):
        app = self.recording_app()
        self.say(app, 'start', 'phrase:submit')
        self.assertEqual(app._voice_plan, self.DictationPlan(submit=True))
        app = self.recording_app()
        app._voice_plan = self.DictationPlan(route='translate', translate_language='German')
        self.say(app, 'start', 'phrase:raw')
        self.assertEqual(app._voice_plan,
                         self.DictationPlan(route='translate', translate_language='German', raw=True))

    def test_keep_going_records_on_and_marks_the_words_for_removal(self):
        app = self.recording_app()
        self.say(app, 'start', 'phrase:keep going')
        app._stop_recording.assert_not_called()
        self.assertTrue(app.is_recording)
        self.assertEqual(app._voice_strip_phrases, ['hey jarvis keep going', 'hey jarvis'])
        self.assertEqual(app._voice_state, 'idle')

    def test_speech_after_the_start_word_is_transcribed_and_matched(self):
        app = self.recording_app()
        app.whisper_manager.transcribe_audio.return_value = 'Hey Jarvis, cancel that.'
        self.say(app, 'start', 'pause')
        app._cancel_recording.assert_called_once_with()

        app = self.recording_app()
        app.whisper_manager.transcribe_audio.return_value = 'and the logs too'
        self.say(app, 'start', 'pause')
        app._stop_recording.assert_not_called()
        app._cancel_recording.assert_not_called()
        self.assertEqual(app._voice_strip_phrases, ['hey jarvis and the logs too', 'hey jarvis'])
        app._notify_user.assert_not_called()

    def test_vosk_transcribes_the_window_and_realtime_never_shares_its_stream(self):
        app = self.recording_app()
        app._voice_listener.can_transcribe.return_value = True
        app._voice_listener.transcribe.return_value = 'done'
        self.say(app, 'start', 'pause')
        app.whisper_manager.transcribe_audio.assert_not_called()
        app._stop_recording.assert_called_once()

        app = self.recording_app(transcription_backend='realtime-ws')
        self.say(app, 'start', 'pause')
        app.whisper_manager.transcribe_audio.assert_not_called()
        app._stop_recording.assert_not_called()

    def test_a_stop_by_other_means_closes_the_window_and_drops_the_start_word(self):
        app = self.recording_app()
        self.say(app, 'start')
        app._voice_recording_ended()
        self.assertEqual(app._voice_state, 'idle')
        self.assertEqual(app._voice_strip_phrases, ['hey jarvis'])
        app._voice_recording_ended()  # nothing open: no change
        self.assertEqual(app._voice_strip_phrases, ['hey jarvis'])

    def test_processing_removes_the_marked_words(self):
        app = self.delivery_app()
        app.audio_capture = types.SimpleNamespace(sample_rate=16000)
        app._current_language_override = None
        app._current_route = 'paste'
        app.whisper_manager.transcribe_audio.return_value = 'Fix the parser. Hey Jarvis, keep going. Add a test.'
        app._finish_live_typing = mock.Mock(return_value=None)
        app._inject_text = mock.Mock(return_value=self.outcome.INJECTED)
        app._notify_capture = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app._process_audio(np.zeros(16000, dtype=np.float32),
                           strip_phrases=['hey jarvis keep going', 'hey jarvis'])
        app._inject_text.assert_called_once_with('Fix the parser. Add a test.', route='paste')

    def test_submit_presses_enter_after_the_paste(self):
        app = self.delivery_app()
        app._inject_text('hello', voice_plan=self.DictationPlan(submit=True))
        app.text_injector.send_enter.assert_called_once_with()
        app = self.delivery_app(auto_submit=True)
        app._inject_text('hello', voice_plan=self.DictationPlan(submit=True))
        app.text_injector.send_enter.assert_not_called()  # auto_submit already pressed it
        app = self.delivery_app()
        app.text_injector.inject_text.return_value = self.outcome.FAILED
        app._inject_text('hello', voice_plan=self.DictationPlan(submit=True))
        app.text_injector.send_enter.assert_not_called()

    def test_cancelling_a_recording_closes_its_window(self):
        app = self.recording_app()
        app._cancel_recording = types.MethodType(self.App._cancel_recording, app)
        app._recording_session = object()
        app._ptt_reset = mock.Mock()
        app._cleanup_recording_state = mock.Mock()
        app._wait_for_start_settled = mock.Mock()
        app.audio_capture = mock.Mock()
        self.say(app, 'start')
        app._voice_strip_phrases = ['hey jarvis']
        app._cancel_recording()
        self.assertEqual(app._voice_state, 'idle')
        self.assertEqual(app._voice_strip_phrases, [])


if __name__ == "__main__":
    unittest.main()
