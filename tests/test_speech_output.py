"""Reading the scratchpad aloud: the OpenRouter speech request, text prep and playback."""
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))

import speech_output
from speech_output import SpeechError, SpeechPlayer


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def settings(**values):
    return lambda key, default=None: values.get(key, default)


class SynthesizeTests(unittest.TestCase):
    def test_request_uses_openrouter_eleven_v4_turbo_and_the_chosen_voice(self):
        seen = {}

        def opener(request, timeout):
            seen['request'], seen['timeout'] = request, timeout
            return FakeResponse(b'ID3audio')

        audio = speech_output.synthesize('Hello there.', settings(scratchpad_voice='George'),
                                         credential_lookup=lambda name: ' sk-or-key ', opener=opener)
        self.assertEqual(audio, b'ID3audio')
        request = seen['request']
        self.assertEqual(request.full_url, 'https://openrouter.ai/api/v1/audio/speech')
        self.assertEqual(request.get_header('Authorization'), 'Bearer sk-or-key')
        self.assertEqual(json.loads(request.data), {
            'model': 'elevenlabs/eleven-v4-turbo', 'input': 'Hello there.',
            'voice': 'george', 'response_format': 'mp3',
        })
        self.assertEqual(seen['timeout'], 30.0)

    def test_voice_argument_and_model_setting_win(self):
        seen = {}

        def opener(request, timeout):
            seen['body'] = json.loads(request.data)
            return FakeResponse(b'a')

        speech_output.synthesize('x', settings(scratchpad_voice='george', scratchpad_speech_model='m/tts',
                                               scratchpad_speech_timeout=5),
                                 voice='lily', credential_lookup=lambda name: 'k', opener=opener)
        self.assertEqual(seen['body']['voice'], 'lily')
        self.assertEqual(seen['body']['model'], 'm/tts')

    def test_unknown_configured_voice_falls_back_to_the_default(self):
        self.assertEqual(speech_output.configured_voice(settings(scratchpad_voice='nobody')), 'sarah')

    def test_missing_key_and_http_errors_raise_speech_error(self):
        with self.assertRaisesRegex(SpeechError, 'config llm'):
            speech_output.synthesize('x', settings(), credential_lookup=lambda name: None,
                                     opener=mock.Mock())

        def refuse(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 402, 'Payment Required', {},
                                         io.BytesIO(b'{"error":{"code":402,"message":"Insufficient credits"}}'))

        with self.assertRaisesRegex(SpeechError, 'HTTP 402: Insufficient credits'):
            speech_output.synthesize('x', settings(), credential_lookup=lambda name: 'k', opener=refuse)
        with self.assertRaisesRegex(SpeechError, 'no audio'):
            speech_output.synthesize('x', settings(), credential_lookup=lambda name: 'k',
                                     opener=lambda request, timeout: FakeResponse(b''))


class TextTests(unittest.TestCase):
    def test_markdown_marks_are_not_read_out(self):
        text = '# Plan\n\n- **Fix** the `parser`\n- [ ] see [the docs](https://x.y)\n> quoted'
        self.assertEqual(speech_output.speakable_text(text),
                         'Plan\n\nFix the parser\nsee the docs\nquoted')

    def test_long_text_is_split_at_breaks_within_the_limit(self):
        text = 'First paragraph.\n\n' + ' '.join(['A sentence here.'] * 10) + '\n\n' + 'x' * 45
        parts = speech_output.split_text(text, limit=40)
        self.assertTrue(all(len(part) <= 40 for part in parts), parts)
        self.assertEqual(parts[0], 'First paragraph.\n\nA sentence here.')
        self.assertEqual(''.join(parts).replace('\n', '').replace(' ', ''),
                         text.replace('\n', '').replace(' ', ''))

    def test_short_text_is_one_part(self):
        self.assertEqual(speech_output.split_text('one\n\ntwo'), ['one\n\ntwo'])


class SpeechPlayerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.audio_dir = Path(self.tempdir.name)

    def _player(self, command, synthesize=None, on_error=None):
        player = SpeechPlayer(synthesize or (lambda part, voice: b'mp3'), on_error=on_error,
                              player=lambda: command, audio_dir=self.audio_dir)
        self.addCleanup(player.stop)
        return player

    def test_parts_are_synthesized_in_order_and_played(self):
        calls = []
        first, second = 'a' * 1000 + '.', 'b' * 1000 + '.'

        def synthesize(part, voice):
            calls.append((part, voice))
            return b'mp3'

        player = self._player([sys.executable, '-c', 'pass'], synthesize)
        player.speak(f'{first}\n\n{second}', 'bella')
        player._thread.join(5)
        self.assertEqual(calls, [(first, 'bella'), (second, 'bella')])
        self.assertFalse(player.speaking)
        self.assertEqual(list(self.audio_dir.iterdir()), [])  # audio files removed

    def test_stop_cuts_playback_off(self):
        player = self._player([sys.executable, '-c', 'import time; time.sleep(30)'])
        player.speak('A long draft.')
        for _ in range(100):
            if player._process is not None:
                break
            threading.Event().wait(0.05)
        self.assertTrue(player.speaking)
        self.assertTrue(player.stop())
        self.assertFalse(player.speaking)
        self.assertFalse(player.stop())

    def test_failures_are_reported_once(self):
        errors = []

        def fail(part, voice):
            raise SpeechError('HTTP 401')

        player = self._player([sys.executable, '-c', 'pass'], fail, errors.append)
        player.speak('text')
        player._thread.join(5)
        self.assertEqual(errors, ['HTTP 401'])

    def test_nothing_to_read_or_no_player_fails_up_front(self):
        with self.assertRaisesRegex(SpeechError, 'nothing to read'):
            self._player(['true']).speak('  ')
        with self.assertRaisesRegex(SpeechError, 'no audio player'):
            self._player(None).speak('text')


if __name__ == '__main__':
    unittest.main()
