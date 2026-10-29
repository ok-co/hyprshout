"""Scratchpad route: the scratchpad shortcut and `record start --scratchpad`
record a dictation the LLM applies to the draft instead of pasting it; the
draft is pasted on request."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import FakeConfig, _import_main_isolated, app_global, patch_app_global


class ScratchpadConfig(FakeConfig):
    def get_word_overrides(self):
        return {}

    def get_filter_filler_words(self):
        return False


class ScratchpadTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()
        cls.llm_refiner = app_global(cls.main.hyprwhsprApp, 'llm_refiner')
        cls.outcome = app_global(cls.main.hyprwhsprApp, 'InjectionOutcome')
        cls.Scratchpad = cls.main.Scratchpad

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        log = patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock())
        log.__enter__()
        self.addCleanup(log.__exit__, None, None, None)

    def _app(self, **settings):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = ScratchpadConfig({'recording_mode': 'toggle', **settings})
        app.is_recording = False
        app._start_recording = mock.Mock()
        app._autostop_start_silence_monitor = mock.Mock()
        app._continuous_start_silence_monitor = mock.Mock()
        app._longform = mock.Mock()
        app._scratchpad = self.Scratchpad(Path(self.tempdir.name) / 'scratchpad.md')
        app._scratchpad_notification_id = None
        app._show_scratchpad = mock.Mock()
        app._close_scratchpad_notification = mock.Mock()
        app._notify_user = mock.Mock()
        app._mark_delivery_healthy = mock.Mock()
        app.text_injector = mock.Mock()
        app.text_injector.deliver_prepared.return_value = (True, 'Paste dispatched')
        app._recording_control_server = mock.Mock()
        app._recording_control_server.has_capture_subscriber.return_value = False
        return app

    def _edit(self, result=None, error=None):
        edit = mock.Mock(return_value=result, side_effect=error)
        return mock.patch.object(self.llm_refiner, 'edit_draft', edit), edit

    # -- triggers --------------------------------------------------------

    def test_control_scratchpad_starts_a_scratchpad_recording(self):
        app = self._app()
        app._handle_control_command('scratchpad', 'de')
        app._start_recording.assert_called_once_with(language_override='de', route='scratchpad')

    def test_control_scratchpad_in_long_form_reaches_the_session(self):
        app = self._app(recording_mode='long_form')
        app._handle_control_command('scratchpad')
        app._longform.request_start.assert_called_once_with(language_override=None, route='scratchpad')

    def test_scratchpad_shortcut_starts_a_scratchpad_recording(self):
        app = self._app()
        app._on_scratchpad_shortcut_triggered()
        app._start_recording.assert_called_once_with(language_override=None, route='scratchpad')
        app = self._app(recording_mode='long_form')
        app._on_scratchpad_shortcut_long_form()
        app._longform.primary_shortcut.assert_called_once_with(route='scratchpad')

    def test_shortcuts_are_registered_per_recording_mode(self):
        expected = {
            'toggle': ('_on_scratchpad_shortcut_triggered', None),
            'push_to_talk': ('_on_scratchpad_shortcut_triggered', '_on_shortcut_released'),
            'auto': ('_on_scratchpad_shortcut_triggered', '_on_shortcut_released'),
            'long_form': ('_on_scratchpad_shortcut_long_form', None),
        }
        for mode, (press, release) in expected.items():
            with self.subTest(mode=mode):
                app = self._app(recording_mode=mode, scratchpad_shortcut='SUPER+ALT+S',
                                scratchpad_paste_shortcut='SUPER+ALT+P')
                app.secondary_shortcuts = app._translate_shortcuts = app._cancel_shortcuts = None
                app._scratchpad_shortcuts = app._scratchpad_paste_shortcuts = None
                handlers = {}

                def factory(key, on_press, on_release, **kwargs):
                    handler = mock.Mock()
                    handlers[key] = (on_press, on_release, handler)
                    return handler

                with patch_app_global(self.main.hyprwhsprApp, 'GlobalShortcuts', factory):
                    app._setup_global_shortcuts()
                on_press, on_release, handler = handlers['SUPER+ALT+S']
                self.assertEqual(on_press, getattr(app, press))
                self.assertEqual(on_release, getattr(app, release) if release else None)
                self.assertIs(app._scratchpad_shortcuts, handler)
                on_press, on_release, handler = handlers['SUPER+ALT+P']
                self.assertIsNone(on_press)  # pastes on release, modifiers up
                self.assertEqual(on_release, app._on_scratchpad_paste_shortcut)
                self.assertIs(app._scratchpad_paste_shortcuts, handler)
                handler.start.assert_called_once_with()

    def test_scratchpad_recording_is_not_typed_live(self):
        app = self._app(realtime_live_typing=True)
        app._current_route = 'scratchpad'
        self.assertFalse(app._live_typing_eligible())

    # -- applying dictations ---------------------------------------------

    def test_dictation_edits_the_draft_and_is_never_pasted(self):
        app = self._app()
        app._scratchpad.set('Fix the parser.')
        patcher, edit = self._edit('Fix the parser.\nAdd tests.')
        with patcher:
            outcome = app._inject_text('also add tests', route='scratchpad')
        self.assertEqual(outcome, self.outcome.CONSUMED)
        edit.assert_called_once_with('Fix the parser.', 'also add tests', app.config.get_setting)
        self.assertEqual(app._scratchpad.read(), 'Fix the parser.\nAdd tests.')
        app.text_injector.inject_text.assert_not_called()
        app._show_scratchpad.assert_called_once_with('Fix the parser.\nAdd tests.')
        app._mark_delivery_healthy.assert_called_once_with()

    def test_a_failed_edit_keeps_the_draft_and_the_dictation(self):
        app = self._app()
        app._scratchpad.set('Keep me.')
        patcher, _ = self._edit(error=self.llm_refiner.LLMRefineError('request timed out'))
        with patcher:
            outcome = app._inject_text('drop everything', route='scratchpad')
        self.assertEqual(outcome, self.outcome.FAILED)
        self.assertEqual(app._scratchpad.read(), 'Keep me.')
        app.text_injector.retain_text.assert_called_once_with('drop everything')
        self.assertIn('request timed out', app._notify_user.call_args[0][1])
        app.text_injector.inject_text.assert_not_called()
        app._show_scratchpad.assert_not_called()

    def test_a_capture_client_still_gets_the_text(self):
        app = self._app()
        app._recording_control_server.has_capture_subscriber.return_value = True
        app._notify_capture = mock.Mock()
        patcher, edit = self._edit('x')
        with patcher:
            app._inject_text('hello', route='scratchpad')
        edit.assert_not_called()
        app._notify_capture.assert_called_once_with('hello', final=True)

    # -- requests ----------------------------------------------------------

    def test_paste_delivers_the_draft_then_clears_it(self):
        app = self._app()
        app._scratchpad.set('Final prompt.')
        ok, message = app._handle_scratchpad_request('scratchpad_paste')
        self.assertTrue(ok)
        self.assertIn('cleared', message)
        app.text_injector.deliver_prepared.assert_called_once_with('Final prompt.', 'paste')
        self.assertEqual(app._scratchpad.read(), '')
        self.assertEqual(app._handle_scratchpad_request('scratchpad_undo'), (True, 'Undone'))
        self.assertEqual(app._scratchpad.read(), 'Final prompt.')

    def test_paste_can_keep_the_draft_and_copy_never_clears(self):
        app = self._app(scratchpad_clear_on_paste=False)
        app._scratchpad.set('Final prompt.')
        self.assertTrue(app._handle_scratchpad_request('scratchpad_paste')[0])
        self.assertTrue(app._handle_scratchpad_request('scratchpad_copy')[0])
        app.text_injector.deliver_prepared.assert_called_with('Final prompt.', 'copy')
        self.assertEqual(app._scratchpad.read(), 'Final prompt.')

    def test_failed_or_empty_paste_keeps_the_draft(self):
        app = self._app()
        self.assertEqual(app._handle_scratchpad_request('scratchpad_paste'), (False, 'The draft is empty'))
        app.text_injector.deliver_prepared.assert_not_called()
        app._scratchpad.set('Keep me.')
        app.text_injector.deliver_prepared.return_value = (False, 'Text delivery is busy')
        self.assertEqual(app._handle_scratchpad_request('scratchpad_paste'), (False, 'Text delivery is busy'))
        self.assertEqual(app._scratchpad.read(), 'Keep me.')

    def test_show_clear_and_undo(self):
        app = self._app()
        self.assertEqual(app._handle_scratchpad_request('scratchpad_undo'), (False, 'Nothing to undo'))
        app._scratchpad.set('line one\nline two')
        self.assertEqual(app._handle_scratchpad_request('scratchpad_show'), (True, 'line one\nline two'))
        self.assertTrue(app._handle_scratchpad_request('scratchpad_clear')[0])
        self.assertEqual(app._handle_scratchpad_request('scratchpad_show'), (True, ''))
        self.assertEqual(app._handle_scratchpad_request('scratchpad_clear'), (True, 'The draft is already empty'))
        self.assertFalse(app._handle_scratchpad_request('scratchpad_nope')[0])

    def test_paste_shortcut_reports_a_paste_that_did_not_happen(self):
        app = self._app()
        app._on_scratchpad_paste_shortcut()
        app._notify_user.assert_called_once_with('hyprwhspr scratchpad', 'The draft is empty')

    def test_notification_shows_the_end_of_the_draft_escaped(self):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = ScratchpadConfig()
        app._scratchpad_notification_id = 7
        send = mock.Mock(return_value=9)
        with mock.patch('desktop_notify.send_notification_with_id', send):
            app._show_scratchpad('use <b> & ' + 'word ' * 100)
        title, body = send.call_args[0]
        self.assertEqual(title, 'Scratchpad · 103 words')
        self.assertTrue(body.startswith('…'))
        self.assertLessEqual(len(body), 290)
        self.assertEqual(send.call_args.kwargs['replaces_id'], 7)
        self.assertEqual(app._scratchpad_notification_id, 9)
        with mock.patch('desktop_notify.send_notification_with_id', send):
            app._show_scratchpad('use <b> & more')
        self.assertEqual(send.call_args[0][1], 'use &lt;b&gt; &amp; more')


if __name__ == '__main__':
    unittest.main()
