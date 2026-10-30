"""The settings editor's key handling, with its dialogs scripted (no terminal needed)."""

import curses
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))

from cli import settings_tui  # noqa: E402
from tests.test_settings_model import document  # noqa: E402


class ScriptedTUI(settings_tui.SettingsTUI):
    """Dialog answers come from lists; what was asked is recorded."""

    def __init__(self, doc, lines=(), picks=(), confirms=(), edits=(), active=False, restarted=True):
        super().__init__(doc, service_active=lambda: active, restart=self._restart)
        self.lines, self.picks, self.confirms, self.edits = list(lines), list(picks), list(confirms), list(edits)
        self.asked, self.messages, self.edited, self.restarts = [], [], [], 0
        self._restarted = restarted

    def _restart(self):
        self.restarts += 1
        return self._restarted

    def ask_line(self, prompt, initial='', secret=False):
        self.asked.append((prompt, initial, secret))
        return self.lines.pop(0)

    def pick(self, title, options, current=0):
        self.asked.append((title, options, current))
        return self.picks.pop(0)

    def confirm(self, question, choices):
        self.asked.append((question, choices))
        return self.confirms.pop(0)

    def show_message(self, title, lines, wrap=True):
        self.messages.append((title, lines))

    def edit_external(self, text, suffix):
        self.edited.append((text, suffix))
        return self.edits.pop(0)

    def go(self, key):
        """Select a setting by key, in its section."""
        for index, section in enumerate(self.sections()):
            if key in section.keys and section.id != 'essentials':
                self.section_index, self.row, self.focus = index, section.keys.index(key), 'settings'
                return
        raise AssertionError(key)

    def press(self, *keys):
        for key in keys:
            self.handle_key(key)


class NavigationTests(unittest.TestCase):
    def test_arrows_move_between_sections_and_settings(self):
        tui = ScriptedTUI(document())
        tui.press(curses.KEY_DOWN, curses.KEY_DOWN)
        self.assertEqual(tui.section().id, 'transcription')
        tui.press(curses.KEY_RIGHT, curses.KEY_DOWN)
        self.assertEqual(tui.current_key(), 'model')
        tui.press(curses.KEY_END)
        self.assertEqual(tui.current_key(), tui.section().keys[-1])
        tui.press(curses.KEY_LEFT, curses.KEY_UP)
        self.assertEqual((tui.focus, tui.section().id, tui.row), ('sections', 'recording', 0))

    def test_search_lists_matches_and_esc_clears_it(self):
        tui = ScriptedTUI(document(), lines=['vosk'])
        tui.press('/')
        self.assertEqual(tui.section().id, 'search')
        self.assertIn('voice_control_vosk_model', tui.keys())
        self.assertEqual(tui.focus, 'settings')
        tui.press('\x1b')
        self.assertIsNone(tui.query)
        self.assertFalse(tui.done)

    def test_changed_view_lists_customized_settings(self):
        tui = ScriptedTUI(document(audio_volume=0.2))
        tui.press('c')
        self.assertEqual(tui.keys(), ['audio_volume'])
        tui.press('c')
        self.assertEqual(tui.section().id, 'essentials')

    def test_unknown_escape_sequences_are_decoded(self):
        class Window:
            def __init__(self, chars):
                self.chars = list(chars)

            def get_wch(self):
                if not self.chars:
                    raise curses.error
                return self.chars.pop(0)

            def nodelay(self, flag):
                pass

        self.assertEqual(settings_tui.read_key(Window('\x1b[B')), curses.KEY_DOWN)
        self.assertEqual(settings_tui.read_key(Window('\x1bOC')), curses.KEY_RIGHT)
        self.assertEqual(settings_tui.read_key(Window('\x1b[3~')), curses.KEY_DC)
        self.assertEqual(settings_tui.read_key(Window('\x1b')), '\x1b')
        self.assertEqual(settings_tui.read_key(Window('x')), 'x')


class EditingTests(unittest.TestCase):
    def test_space_toggles_and_cycles(self):
        tui = ScriptedTUI(document())
        tui.go('auto_submit')
        tui.press(' ')
        self.assertIs(tui.doc.get('auto_submit'), True)
        tui.go('recording_mode')
        tui.press(' ')
        self.assertEqual(tui.doc.get('recording_mode'), 'push_to_talk')
        tui.press(curses.KEY_BTAB)
        self.assertEqual(tui.doc.get('recording_mode'), 'toggle')

    def test_enter_on_a_choice_lists_it(self):
        tui = ScriptedTUI(document(), picks=[3])
        tui.go('recording_mode')
        tui.press('\n')
        title, options, current = tui.asked[0]
        self.assertEqual((options[3], current), ('continuous', 0))
        self.assertEqual(tui.doc.get('recording_mode'), 'continuous')

    def test_typed_values_are_checked_before_they_take(self):
        tui = ScriptedTUI(document(), lines=['99', '2'])
        tui.go('voice_control_pause_seconds')
        tui.press('\n')
        self.assertEqual(tui.asked[0][1], '1.5')
        self.assertEqual(tui.status_kind, 'error')
        self.assertIn('at most 10', tui.status)
        self.assertEqual(tui.doc.get('voice_control_pause_seconds'), 1.5)
        tui.press('\n')
        self.assertEqual(tui.doc.get('voice_control_pause_seconds'), 2)

    def test_custom_prompts_start_from_the_built_in_one(self):
        tui = ScriptedTUI(document(), edits=['Rewrite it as a terse prompt.\n'])
        tui.go('llm_refine_prompt')
        tui.press('\n')
        self.assertIn('transcript', tui.edited[0][0].lower())
        self.assertEqual(tui.edited[0][1], '.md')
        self.assertEqual(tui.doc.get('llm_refine_prompt'), 'Rewrite it as a terse prompt.')

    def test_an_untouched_built_in_prompt_stays_unset(self):
        tui = ScriptedTUI(document())
        tui.go('llm_refine_prompt')
        tui.edits = [settings_tui.builtin_text('llm_refine_prompt')]
        tui.press('\n')
        self.assertIsNone(tui.doc.get('llm_refine_prompt'))
        self.assertFalse(tui.doc.dirty)

    def test_emptied_prompts_go_back_to_the_built_in_one(self):
        tui = ScriptedTUI(document(assistant_prompt='Be grumpy.'), edits=['\n\n'])
        tui.go('assistant_prompt')
        tui.press('\n')
        self.assertIsNone(tui.doc.get('assistant_prompt'))

    def test_structured_settings_are_edited_as_json_and_can_be_retried(self):
        tui = ScriptedTUI(document(), edits=['{"hyper whisper": 5}', '{"jarvis": "Jarvis"}'], confirms=['y'])
        tui.go('word_overrides')
        tui.press('\n')
        self.assertEqual(tui.edited[0], ('{\n  "hyper whisper": "hyprwhspr"\n}', '.json'))
        self.assertEqual(tui.edited[1][0], '{"hyper whisper": 5}')
        self.assertEqual(tui.doc.get('word_overrides'), {'jarvis': 'Jarvis'})

    def test_e_edits_any_setting_in_the_editor(self):
        tui = ScriptedTUI(document(), edits=['0.25\n'])
        tui.go('audio_volume')
        tui.press('e')
        self.assertEqual(tui.doc.get('audio_volume'), 0.25)

    def test_backspace_restores_the_default(self):
        tui = ScriptedTUI(document(audio_volume=0.2))
        tui.go('audio_volume')
        tui.press(curses.KEY_BACKSPACE)
        self.assertEqual(tui.doc.get('audio_volume'), 0.5)
        self.assertIn('default', tui.status)

    def test_api_keys_are_typed_hidden_and_removed_on_request(self):
        doc = document()
        tui = ScriptedTUI(doc, lines=['', 'sk-or-secret99'], confirms=['n', 'y'])
        tui.go('credential:openrouter')
        tui.press('\n')
        self.assertNotIn('openrouter', doc.store.keys)  # empty keeps it
        tui.press('\n')
        self.assertTrue(all(secret for _, _, secret in tui.asked[:2]))
        self.assertEqual(doc.store.keys['openrouter'], 'sk-or-secret99')
        self.assertIn('set (', tui.value_text('credential:openrouter'))
        tui.press(curses.KEY_DC)
        self.assertIn('openrouter', doc.store.keys)
        tui.press(curses.KEY_DC)
        self.assertNotIn('openrouter', doc.store.keys)
        self.assertFalse(doc.dirty)


class SaveAndQuitTests(unittest.TestCase):
    def test_invalid_settings_block_the_save_and_are_listed(self):
        tui = ScriptedTUI(document(audio_volume=7))
        tui.go('auto_submit')
        tui.press(' ', 's')
        self.assertEqual(tui.doc.manager.saves, 0)
        title, lines = tui.messages[0]
        self.assertIn('Not saved', title)
        self.assertIn('audio_volume', lines[0])

    def test_saving_offers_a_restart_when_the_service_runs(self):
        tui = ScriptedTUI(document(), confirms=['y'], active=True)
        tui.go('auto_submit')
        tui.press(' ', 's')
        self.assertEqual(tui.doc.manager.saves, 1)
        self.assertEqual(tui.restarts, 1)
        self.assertEqual(tui.status_kind, 'ok')
        self.assertIn('restarted', tui.status)

        tui = ScriptedTUI(document(), active=False)
        tui.go('auto_submit')
        tui.press(' ', 's')
        self.assertEqual(tui.restarts, 0)
        self.assertIn('systemctl --user restart hyprwhspr', tui.status)

    def test_a_failed_restart_is_reported(self):
        tui = ScriptedTUI(document(), confirms=['y'], active=True, restarted=False)
        tui.go('auto_submit')
        tui.press(' ', 's')
        self.assertEqual(tui.status_kind, 'error')

    def test_nothing_to_save(self):
        tui = ScriptedTUI(document())
        tui.press('s')
        self.assertEqual(tui.doc.manager.saves, 0)
        self.assertEqual(tui.status, 'Nothing to save')

    def test_quitting_with_changes_asks(self):
        for answer, saves, done in (('c', 0, False), ('esc', 0, False), ('n', 0, True), ('y', 1, True)):
            tui = ScriptedTUI(document(), confirms=[answer])
            tui.go('auto_submit')
            with self.subTest(answer=answer):
                tui.press(' ', 'q')
                self.assertEqual((tui.doc.manager.saves, tui.done), (saves, done))

    def test_quitting_unchanged_needs_no_answer(self):
        tui = ScriptedTUI(document())
        tui.press('q')
        self.assertTrue(tui.done)


class DrawTests(unittest.TestCase):
    """Drawing every section must not raise, on a fake screen."""

    class Screen:
        def __init__(self, height=30, width=100):
            self.size = (height, width)
            self.text = {}

        def getmaxyx(self):
            return self.size

        def erase(self):
            self.text = {}

        def refresh(self):
            pass

        def addnstr(self, y, x, text, n, attr=0):
            self.text[(y, x)] = text[:n]

    def test_every_section_and_setting_draws(self):
        tui = ScriptedTUI(document(rest_api_provider='groq', my_old_setting=[1]))
        tui.screen = self.Screen()
        for index, section in enumerate(tui.sections()):
            tui.section_index, tui.focus = index, 'settings'
            for row in range(len(section.keys)):
                tui.row = row
                tui.draw()
        shown = ' '.join(tui.screen.text.values())
        self.assertIn('hyprwhspr settings', shown)

    def test_a_small_terminal_asks_for_room(self):
        tui = ScriptedTUI(document())
        tui.screen = self.Screen(10, 40)
        tui.draw()
        self.assertIn('at least', ' '.join(tui.screen.text.values()))


class CommandTests(unittest.TestCase):
    def test_without_a_terminal_it_explains_what_to_use(self):
        with mock.patch.object(sys.stdin, 'isatty', return_value=False), \
                mock.patch('sys.stderr') as stderr:
            self.assertFalse(settings_tui.settings_command())
        self.assertIn('config edit', ''.join(str(c.args[0]) for c in stderr.write.call_args_list))

    def test_the_editor_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, {'VISUAL': '', 'EDITOR': 'micro'}):
            self.assertEqual(settings_tui.find_editor(), 'micro')
        with mock.patch.dict(os.environ, {'VISUAL': 'code -w', 'EDITOR': 'micro'}):
            self.assertEqual(settings_tui.find_editor(), 'code -w')


if __name__ == "__main__":
    unittest.main()
