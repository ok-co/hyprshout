import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import voice_phrases  # noqa: E402
from voice_phrases import (  # noqa: E402
    COMMAND_ACTIONS, COMMAND_PHRASES, DICTATION_ACTIONS, DICTATION_PHRASES, DRAFT_INSTRUCTIONS,
    PhraseBook, normalize, remove_phrase, table_language,
)


def settings(**values):
    return lambda key, default=None: values.get(key, default)


class TableTests(unittest.TestCase):
    def test_every_language_covers_every_action_with_known_names(self):
        for language, table in COMMAND_PHRASES.items():
            with self.subTest(language=language):
                self.assertEqual(set(table), set(COMMAND_ACTIONS))
                self.assertEqual(set(DICTATION_PHRASES[language]), set(DICTATION_ACTIONS))

    def test_no_phrase_means_two_things_in_one_table_and_language(self):
        for tables in (COMMAND_PHRASES, DICTATION_PHRASES):
            for language, table in tables.items():
                seen = {}
                for action, phrases in table.items():
                    for phrase in phrases:
                        with self.subTest(language=language, phrase=phrase):
                            self.assertEqual(seen.setdefault(normalize(phrase), action), action)

    def test_draft_instructions_exist_for_every_draft_action(self):
        self.assertEqual(set(DRAFT_INSTRUCTIONS), {a for a in COMMAND_ACTIONS if a.startswith('draft_')})


class MatchTests(unittest.TestCase):
    def test_whole_commands_match_whatever_the_punctuation(self):
        book = PhraseBook(settings())
        self.assertEqual(book.match_command('Never mind.'), ('cancel', None))
        self.assertEqual(book.match_command('Send the draft!'), ('scratchpad_paste', None))
        self.assertEqual(book.match_command('Okay, undo that, please.'), ('scratchpad_undo', None))
        self.assertEqual(book.match_dictation("That's it."), ('finish', None))
        self.assertEqual(book.match_dictation('Keep going'), ('continue', None))

    def test_a_phrase_inside_a_longer_command_does_not_match(self):
        book = PhraseBook(settings())
        self.assertIsNone(book.match_command('please cancel my subscription'))
        self.assertIsNone(book.match_command('send the draft to Anna'))
        self.assertIsNone(book.match_dictation('done with the parser'))
        self.assertIsNone(book.match_command(''))
        self.assertIsNone(book.match_command(None))

    def test_every_language_is_matched_unless_one_is_chosen(self):
        self.assertEqual(PhraseBook(settings()).match_command('Anuluj'), ('cancel', None))
        self.assertEqual(PhraseBook(settings()).match_dictation('fertig'), ('finish', None))
        polish = PhraseBook(settings(), ['pl'])
        self.assertEqual(polish.match_command('wyślij szkic'), ('scratchpad_paste', None))
        self.assertIsNone(polish.match_command('cancel'))

    def test_languages_to_translate_into_get_their_own_phrases(self):
        book = PhraseBook(settings(voice_control_languages=['de', 'fr']))
        self.assertEqual(book.match_command('Translate the draft into German'), ('draft_translate', 'de'))
        self.assertEqual(book.match_command('translate into French'), ('translate', 'fr'))
        self.assertIsNone(book.match_command('translate into Polish'))
        self.assertIsNone(PhraseBook(settings(), ['de']).match_command('translate into German'))

    def test_overrides_replace_or_turn_off_an_actions_phrases(self):
        book = PhraseBook(settings(voice_control_phrases={
            'command': {'cancel': ['forget about it'], 'help': []},
            'dictation': {'finish': ['ship it']},
        }))
        self.assertEqual(book.match_command('forget about it'), ('cancel', None))
        self.assertIsNone(book.match_command('never mind'))
        self.assertIsNone(book.match_command('help'))
        self.assertEqual(book.match_dictation('Ship it.'), ('finish', None))
        self.assertIsNone(book.match_dictation('done'))
        self.assertEqual(book.match_dictation('cancel'), ('cancel', None))

    def test_malformed_overrides_are_ignored(self):
        book = PhraseBook(settings(voice_control_phrases={'command': ['cancel'], 'dictation': {'finish': 'done'}}))
        self.assertEqual(book.match_command('never mind'), ('cancel', None))
        self.assertIsNone(book.match_dictation('done'))

    def test_phrases_list_both_tables_once(self):
        phrases = PhraseBook(settings(), ['en']).phrases()
        self.assertEqual(len(phrases), len(set(phrases)))
        self.assertIn('send the draft', phrases)
        self.assertIn('keep going', phrases)
        self.assertIn("that's it", phrases)

    def test_help_names_a_phrase_for_each_kind_of_command(self):
        text = PhraseBook(settings(), ['en']).help_text('hey jarvis')
        self.assertIn('After "hey jarvis"', text)
        for phrase in ('send the draft', 'translate the draft', 'cancel', 'done', 'continue'):
            self.assertIn(phrase, text)


class LanguageTests(unittest.TestCase):
    def test_vosk_model_settings_map_to_tables(self):
        self.assertEqual(table_language('en-us'), 'en')
        self.assertEqual(table_language('EN_GB'), 'en')
        self.assertEqual(table_language('pl'), 'pl')
        self.assertIsNone(table_language('ja'))
        self.assertIsNone(table_language('~/models/vosk-model-pl'))
        self.assertIsNone(table_language(None))


class RemovePhraseTests(unittest.TestCase):
    def test_the_start_word_and_what_followed_it_leave_the_text(self):
        text = 'Fix the parser. Hey Jarvis, keep going. Then add a test.'
        self.assertEqual(remove_phrase(text, 'hey jarvis keep going'), 'Fix the parser. Then add a test.')

    def test_misheard_spellings_are_removed_too(self):
        self.assertEqual(remove_phrase('Fix the parser hey jervis and add a test', 'hey jarvis'),
                         'Fix the parser and add a test')

    def test_other_text_is_untouched(self):
        text = 'Hello there, how are you?'
        self.assertEqual(remove_phrase(text, 'hey jarvis'), text)
        self.assertEqual(remove_phrase('', 'hey jarvis'), '')
        self.assertEqual(remove_phrase(text, ''), text)

    def test_module_has_no_heavy_imports(self):
        self.assertNotIn('numpy', vars(voice_phrases))


if __name__ == "__main__":
    unittest.main()
