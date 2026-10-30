import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import settings_model  # noqa: E402
from config_manager import build_default_config  # noqa: E402
from settings_model import (  # noqa: E402
    ESSENTIALS, SECTIONS, UNSET, SettingsDocument, format_value, label_for, make_field, parse_text,
)


class FakeManager:
    def __init__(self, values=None):
        self.default_config = build_default_config()
        self.config = copy.deepcopy(self.default_config)
        self.config.update(values or {})
        self.saves = 0

    def save_config(self):
        self.saves += 1
        return True


class FakeStore:
    def __init__(self):
        self.keys = {}


def document(**values):
    store = FakeStore()
    credentials = {'get': store.keys.get, 'save': lambda i, k: store.keys.__setitem__(i, k) or True,
                   'delete': lambda i: store.keys.pop(i, None) is not None,
                   'mask': lambda k: k[:3] + '…' + k[-2:]}
    doc = SettingsDocument(FakeManager(values), credentials=credentials)
    doc.store = store
    return doc


class SchemaCoverageTests(unittest.TestCase):
    def test_every_setting_is_listed_once_outside_the_shortcuts(self):
        doc = document()
        sections = doc.sections()
        listed = [k for s in sections if s.id not in ('essentials', 'credentials') for k in s.keys]
        self.assertEqual(sorted(listed), sorted(doc.fields))
        self.assertEqual(len(listed), len(set(listed)))

    def test_sections_and_essentials_name_real_settings(self):
        properties = settings_model.load_schema()['properties']
        for _, _, _, keys, _ in SECTIONS:
            for key in keys:
                with self.subTest(key=key):
                    self.assertIn(key, properties)
        for key in ESSENTIALS:
            self.assertIn(key, properties)

    def test_nothing_falls_through_to_advanced_today(self):
        # A new setting lands in Advanced until SECTIONS places it
        self.assertEqual(document().sections()[-2].id, 'devices')

    def test_named_keys_lead_their_section(self):
        transcription = next(s for s in document().sections() if s.id == 'transcription')
        self.assertEqual(transcription.keys[:3], ['transcription_backend', 'model', 'language'])

    def test_api_keys_come_last(self):
        doc = document(rest_api_provider='groq')
        keys = doc.sections()[-1]
        self.assertEqual(keys.id, 'credentials')
        self.assertEqual(keys.keys, ['credential:openrouter', 'credential:llm-custom', 'credential:huggingface',
                                     'credential:groq'])


class FieldTests(unittest.TestCase):
    def test_kinds_follow_the_schema(self):
        doc = document()
        kinds = {key: doc.fields[key].kind for key in (
            'auto_submit', 'recording_mode', 'beam_size', 'audio_volume', 'primary_shortcut', 'language',
            'llm_refine_prompt', 'assistant_prompt', 'word_overrides', 'voice_control_languages')}
        self.assertEqual(kinds, {
            'auto_submit': 'bool', 'recording_mode': 'choice', 'beam_size': 'int', 'audio_volume': 'number',
            'primary_shortcut': 'text', 'language': 'text', 'llm_refine_prompt': 'prompt',
            'assistant_prompt': 'prompt', 'word_overrides': 'json', 'voice_control_languages': 'json'})

    def test_nullable_choices_offer_the_default(self):
        f = make_field('x', {'type': ['string', 'null'], 'enum': ['a', 'b']})
        self.assertEqual(f.choices, (None, 'a', 'b'))
        f = make_field('y', {'type': ['boolean', 'null']})
        self.assertEqual((f.kind, f.choices), ('choice', (None, True, False)))

    def test_labels_drop_only_redundant_prefixes(self):
        self.assertEqual(label_for('voice_control_pause_seconds', 'voice'), 'Pause seconds')
        self.assertEqual(label_for('llm_refine_endpoint_url', 'llm'), 'Refine endpoint URL')
        self.assertEqual(label_for('audio_device_name', 'devices'), 'Audio device name')
        self.assertEqual(label_for('mic_osd_enabled', 'feedback'), 'Mic OSD enabled')

    def test_values_render_on_one_short_line(self):
        f = make_field('k', {'type': 'string'})
        self.assertEqual(format_value(f, True), 'on')
        self.assertEqual(format_value(f, None), 'default')
        self.assertEqual(format_value(f, UNSET), 'auto')
        self.assertEqual(format_value(f, 'a\nb'), 'a ⏎ b')
        self.assertEqual(format_value(f, {'a': 1}), '1 entry')
        self.assertEqual(format_value(f, 'x' * 50, width=10), 'x' * 9 + '…')


class ParseTests(unittest.TestCase):
    def test_numbers_accept_a_comma_and_keep_whole_numbers_whole(self):
        number = make_field('n', {'type': 'number'})
        self.assertEqual(parse_text(number, '2,5'), 2.5)
        both = make_field('m', {'type': ['integer', 'number']})
        self.assertEqual(parse_text(make_field('i', {'type': 'integer'}), ' 7 '), 7)
        with self.assertRaisesRegex(ValueError, 'number'):
            parse_text(number, 'fast')
        with self.assertRaisesRegex(ValueError, 'whole number'):
            parse_text(make_field('i', {'type': 'integer'}), '1.5')
        self.assertIsInstance(parse_text(both, '3'), int)

    def test_mixed_types_are_read_by_what_was_typed(self):
        f = make_field('device', {'type': ['integer', 'string', 'null']})
        self.assertEqual(parse_text(f, '3'), 3)
        self.assertEqual(parse_text(f, 'USB mic'), 'USB mic')
        self.assertIsNone(parse_text(f, ''))

    def test_json_errors_say_where(self):
        with self.assertRaisesRegex(ValueError, 'line 1'):
            parse_text(make_field('j', {'type': 'object'}), '{"a": }')


class DocumentTests(unittest.TestCase):
    def test_typed_values_are_checked_against_the_schema(self):
        doc = document()
        self.assertIn('at most 10', doc.apply_text('voice_control_pause_seconds', '20'))
        self.assertEqual(doc.get('voice_control_pause_seconds'), 1.5)
        self.assertIsNone(doc.apply_text('voice_control_pause_seconds', '2.5'))
        self.assertEqual(doc.get('voice_control_pause_seconds'), 2.5)
        self.assertIn('choose one of', doc.apply_text('recording_mode', 'sometimes'))

    def test_empty_input_means_default_or_unset(self):
        doc = document(language='de', primary_shortcut='CTRL+X', threads=4)
        self.assertIsNone(doc.apply_text('language', ''))
        self.assertIsNone(doc.get('language'))
        self.assertIsNone(doc.apply_text('primary_shortcut', '  '))
        self.assertEqual(doc.get('primary_shortcut'), doc.defaults['primary_shortcut'])
        del doc.defaults['threads']  # a setting with no default goes back to unset
        self.assertIsNone(doc.apply_text('threads', ''))
        self.assertNotIn('threads', doc.values)
        self.assertIs(doc.get('threads'), UNSET)

    def test_reset_customized_and_changed(self):
        doc = document(audio_volume=0.3)
        self.assertTrue(doc.is_customized('audio_volume'))
        self.assertFalse(doc.dirty)
        doc.reset('audio_volume')
        self.assertFalse(doc.is_customized('audio_volume'))
        self.assertEqual(doc.changed_keys(), ['audio_volume'])
        self.assertEqual(doc.customized_section().keys, [])

    def test_cycling_choices_and_toggles(self):
        doc = document()
        doc.cycle('auto_submit')
        self.assertIs(doc.get('auto_submit'), True)
        doc.cycle('recording_mode')
        self.assertEqual(doc.get('recording_mode'), 'push_to_talk')
        doc.cycle('recording_mode', -1)
        doc.cycle('recording_mode', -1)
        self.assertEqual(doc.get('recording_mode'), 'long_form')

    def test_errors_cover_values_set_outside_the_editor(self):
        doc = document(audio_volume=7, recording_mode='sometimes')
        keys = sorted(key for key, _ in doc.errors())
        self.assertEqual(keys, ['audio_volume', 'recording_mode'])
        self.assertEqual(document().errors(), [])

    def test_save_writes_through_the_config_manager(self):
        doc = document()
        doc.cycle('auto_submit')
        self.assertTrue(doc.save())
        self.assertEqual(doc.manager.saves, 1)
        self.assertFalse(doc.dirty)

    def test_unknown_and_per_language_keys_are_kept_and_shown(self):
        doc = document(whisper_prompt_de='Grüß Gott', my_old_setting=1)
        self.assertEqual(doc.fields['whisper_prompt_de'].kind, 'prompt')
        self.assertTrue(doc.fields['whisper_prompt_de'].known)
        self.assertFalse(doc.fields['my_old_setting'].known)
        self.assertIsNone(doc.check('my_old_setting', 'anything'))
        self.assertIn('whisper_prompt_de', next(s for s in doc.sections() if s.id == 'transcription').keys)

    def test_search_matches_names_and_descriptions(self):
        doc = document()
        self.assertEqual(doc.search('pause sec')[:1], ['long_form_auto_pause_seconds'])
        self.assertIn('llm_refine_prompt', doc.search('prompt'))
        self.assertIn('voice_control_engine', doc.search('vosk'))
        self.assertEqual(doc.search('   '), [])

    def test_api_keys_are_stored_masked_and_removable(self):
        doc = document()
        cred = doc.credential('credential:openrouter')
        self.assertEqual(doc.credential_status(cred), 'not set')
        self.assertTrue(doc.set_credential(cred, '  sk-or-abcdef  '))
        self.assertEqual(doc.store.keys['openrouter'], 'sk-or-abcdef')
        self.assertEqual(doc.credential_status(cred), 'set (sk-…ef)')
        self.assertTrue(doc.delete_credential(cred))
        self.assertEqual(doc.credential_status(cred), 'not set')
        self.assertNotIn('openrouter', doc.values)

    def test_built_in_prompts_are_offered_as_starting_points(self):
        self.assertIn('transcript', settings_model.builtin_text('llm_refine_prompt').lower())
        self.assertIn('jarvis', settings_model.builtin_text('assistant_prompt').lower())
        self.assertIsNone(settings_model.builtin_text('whisper_prompt'))


if __name__ == "__main__":
    unittest.main()
