"""Translate triggers: the translate shortcut and `record start --translate`
start a recording whose dictation the LLM translates; other triggers don't."""

import threading
import types
import unittest
from unittest import mock

import numpy as np

from tests.test_suspend_resume_recovery import FakeConfig, _import_main_isolated, patch_app_global


class TranslateTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, **settings):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig({'recording_mode': 'toggle', **settings})
        app.is_recording = False
        app._start_recording = mock.Mock()
        app._autostop_start_silence_monitor = mock.Mock()
        app._continuous_start_silence_monitor = mock.Mock()
        app._longform = mock.Mock()
        return app

    def test_control_translate_starts_a_translated_recording(self):
        app = self._app()
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            app._handle_control_command('translate', 'it')
            app._handle_control_command('start')
        self.assertEqual(app._start_recording.call_args_list, [
            mock.call(language_override='it', route='translate'),
            mock.call(language_override=None, route='paste'),
        ])

    def test_control_translate_in_long_form_starts_a_translated_session(self):
        app = self._app(recording_mode='long_form')
        app._handle_control_command('translate')
        app._longform.request_start.assert_called_once_with(language_override=None, route='translate')

    def test_translate_shortcut_starts_translated_and_primary_does_not(self):
        app = self._app()
        app._on_translate_shortcut_triggered()
        app._on_shortcut_triggered()
        self.assertEqual(app._start_recording.call_args_list, [
            mock.call(language_override=None, route='translate'),
            mock.call(language_override=None, route='paste'),
        ])

    def test_translate_shortcut_is_registered_per_recording_mode(self):
        expected = {
            'toggle': ('_on_translate_shortcut_triggered', None),
            'push_to_talk': ('_on_translate_shortcut_triggered', '_on_shortcut_released'),
            'auto': ('_on_translate_shortcut_triggered', '_on_shortcut_released'),
            'long_form': ('_on_translate_shortcut_long_form', None),
        }
        for mode, (press, release) in expected.items():
            with self.subTest(mode=mode):
                app = self._app(recording_mode=mode, translate_shortcut='SUPER+ALT+T')
                app.secondary_shortcuts = app._translate_shortcuts = app._cancel_shortcuts = None
                handlers = {}

                def factory(key, on_press, on_release, **kwargs):
                    handler = mock.Mock()
                    handlers[key] = (on_press, on_release, handler)
                    return handler

                with patch_app_global(self.main.hyprwhsprApp, 'GlobalShortcuts', factory), \
                        patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
                    app._setup_global_shortcuts()
                on_press, on_release, handler = handlers['SUPER+ALT+T']
                self.assertEqual(on_press, getattr(app, press))
                self.assertEqual(on_release, getattr(app, release) if release else None)
                self.assertIs(app._translate_shortcuts, handler)
                handler.start.assert_called_once_with()

    def test_no_translate_shortcut_without_the_setting(self):
        app = self._app()
        app.secondary_shortcuts = app._translate_shortcuts = app._cancel_shortcuts = None
        with patch_app_global(self.main.hyprwhsprApp, 'GlobalShortcuts', mock.Mock()) as factory, \
                patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            app._setup_global_shortcuts()
        self.assertIsNone(app._translate_shortcuts)
        self.assertEqual(factory.call_count, 1)  # the primary shortcut only

    def test_continuous_flush_keeps_the_sessions_route(self):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._recording_lock = threading.Lock()
        app._recording_session = session = object()
        app._current_language_override = None
        app._current_route = 'translate'
        app.is_recording = True
        app._continuous_flush_lock = threading.Lock()
        app._continuous_transcription_done = threading.Event()
        app._continuous_cancelled = False
        app.audio_capture = mock.Mock(sample_rate=10)
        app.audio_capture.flush_buffer.return_value = np.full(10, 0.25, dtype=np.float32)
        app.config = types.SimpleNamespace(get_hallucination_markers=lambda: [])
        app.whisper_manager = mock.Mock()
        app.whisper_manager.transcribe_audio.return_value = 'ciao'
        app._is_zero_volume = mock.Mock(return_value=False)
        app._save_debug_recording = mock.Mock()
        app._inject_text = mock.Mock()
        app._notify_capture = mock.Mock()
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            app._continuous_flush_audio(session)
            self.assertTrue(app._continuous_transcription_done.wait(timeout=2))
        app._inject_text.assert_called_once_with('ciao', route='translate')


if __name__ == '__main__':
    unittest.main()
