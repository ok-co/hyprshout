"""Voice assistant: the spoken-reply prompt, the chat request and the short conversation memory."""

import json
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest import mock

LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))

import voice_assistant
from voice_assistant import DEFAULT_PROMPT, AssistantError, Conversation, build_system_prompt

from tests.test_suspend_resume_recovery import _import_main_isolated, app_global, patch_app_global


KEY = "sk-or-v1-secretsecretsecret"
NOW = datetime(2026, 10, 29, 14, 5)


def settings(**values):
    return lambda key, default=None: values.get(key, default)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ChatOpener:
    """Answers every request with the next reply and records the request bodies."""

    def __init__(self, *replies, error=None):
        self.replies = list(replies)
        self.error = error
        self.bodies = []
        self.timeouts = []

    def __call__(self, request, timeout):
        self.bodies.append(json.loads(request.data))
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return FakeResponse({"choices": [{"finish_reason": "stop",
                                          "message": {"content": self.replies.pop(0)}}]})


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SystemPromptTests(unittest.TestCase):
    def test_default_prompt_is_a_witty_spoken_jarvis_with_the_time(self):
        prompt = build_system_prompt(settings(), now=NOW)
        self.assertTrue(prompt.startswith(DEFAULT_PROMPT))
        self.assertIn("Jarvis", prompt)
        self.assertIn("wit", prompt)
        self.assertIn("No Markdown", prompt)
        self.assertIn("Thursday, 29 October 2026, 14:05", prompt)

    def test_the_wake_words_must_never_be_said(self):
        prompt = build_system_prompt(settings(), avoid_phrases=("hey jarvis", None, "hey mycroft", "hey jarvis"),
                                     now=NOW)
        self.assertIn('Never say these phrases, because they control the app you run in: '
                      '"hey jarvis", "hey mycroft".', prompt)
        self.assertNotIn("Never say these phrases", build_system_prompt(settings(), now=NOW))

    def test_custom_prompt_and_prompt_file(self):
        prompt = build_system_prompt(settings(assistant_prompt="You are Marvin, gloomy."), now=NOW)
        self.assertTrue(prompt.startswith("You are Marvin, gloomy.\n\nIt is now"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text("  You are HAL.  \n", encoding="utf-8")
            prompt = build_system_prompt(settings(assistant_prompt="ignored", assistant_prompt_file=str(path)),
                                         now=NOW)
            self.assertTrue(prompt.startswith("You are HAL.\n\nIt is now"))
            path.write_text("   ", encoding="utf-8")
            with self.assertRaisesRegex(AssistantError, "empty"):
                build_system_prompt(settings(assistant_prompt_file=str(path)), now=NOW)
            with self.assertRaisesRegex(AssistantError, "could not read"):
                build_system_prompt(settings(assistant_prompt_file=str(Path(tmp) / "missing")), now=NOW)

    def test_voice_defaults_to_george_and_accepts_known_voices(self):
        self.assertEqual(voice_assistant.configured_voice(settings()), "george")
        self.assertEqual(voice_assistant.configured_voice(settings(assistant_voice="Lily")), "lily")
        self.assertEqual(voice_assistant.configured_voice(settings(assistant_voice="nobody")), "george")


class ConversationTests(unittest.TestCase):
    def ask(self, conversation, question, opener, **values):
        return conversation.ask(question, settings(**values), avoid_phrases=("hey jarvis",),
                                credential_lookup=lambda name: KEY, opener=opener)

    def test_question_goes_to_the_refine_provider_with_the_assistant_model(self):
        opener = ChatOpener("  Paris, last I checked.  ")
        reply = self.ask(Conversation(), " capital of France? ", opener,
                         llm_refine_model="base/model", assistant_model="chat/model", assistant_timeout=12)
        self.assertEqual(reply, "Paris, last I checked.")
        body = opener.bodies[0]
        self.assertEqual(body["model"], "chat/model")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertIn('"hey jarvis"', body["messages"][0]["content"])
        self.assertEqual(body["messages"][1]["content"], "capital of France?")
        self.assertEqual(opener.timeouts, [12.0])

        opener = ChatOpener("Hi.")
        self.ask(Conversation(), "hello", opener, llm_refine_model="base/model", assistant_timeout=0)
        self.assertEqual(opener.bodies[0]["model"], "base/model")
        self.assertEqual(opener.timeouts, [30.0])

    def test_follow_ups_carry_the_recent_exchanges(self):
        clock = Clock()
        conversation = Conversation(clock=clock)
        opener = ChatOpener("Twenty degrees.", "Sixty-eight Fahrenheit.")
        self.ask(conversation, "How warm is twenty Celsius?", opener)
        clock.now += 60
        self.ask(conversation, "And in Fahrenheit?", opener)
        messages = opener.bodies[1]["messages"]
        self.assertEqual([(m["role"], m["content"]) for m in messages[1:]], [
            ("user", "How warm is twenty Celsius?"),
            ("assistant", "Twenty degrees."),
            ("user", "And in Fahrenheit?"),
        ])

    def test_memory_expires_after_a_quiet_spell_or_when_turned_off(self):
        for values, wait in (({}, 5 * 60 + 1), ({"assistant_memory_minutes": 0}, 1)):
            clock = Clock()
            conversation = Conversation(clock=clock)
            opener = ChatOpener("One.", "Two.")
            with self.subTest(values=values):
                self.ask(conversation, "first", opener, **values)
                clock.now += wait
                self.ask(conversation, "second", opener, **values)
                self.assertEqual(len(opener.bodies[1]["messages"]), 2)

    def test_only_the_latest_exchanges_are_kept(self):
        conversation = Conversation(clock=Clock())
        opener = ChatOpener(*[f"Reply {n}." for n in range(voice_assistant.MAX_EXCHANGES + 2)])
        for n in range(voice_assistant.MAX_EXCHANGES + 2):
            self.ask(conversation, f"question {n}", opener)
        history = opener.bodies[-1]["messages"][1:-1]
        self.assertEqual(len(history), 2 * voice_assistant.MAX_EXCHANGES)
        self.assertEqual(history[0]["content"], "question 1")

    def test_failures_raise_assistant_errors_and_are_not_remembered(self):
        conversation = Conversation(clock=Clock())
        with self.assertRaisesRegex(AssistantError, "nothing was asked"):
            self.ask(conversation, "  ", ChatOpener())
        error = urllib.error.URLError("offline")
        with self.assertRaisesRegex(AssistantError, "cannot reach"):
            self.ask(conversation, "hello", ChatOpener(error=error))
        with self.assertRaisesRegex(AssistantError, "API key"):
            conversation.ask("hello", settings(), credential_lookup=lambda name: None, opener=ChatOpener("x"))
        opener = ChatOpener("   ", "Hello.")
        with self.assertRaisesRegex(AssistantError, "empty"):
            self.ask(conversation, "hello", opener)
        self.ask(conversation, "hello again", opener)
        self.assertEqual(len(opener.bodies[1]["messages"]), 2)

    def test_forget_starts_a_new_conversation(self):
        conversation = Conversation(clock=Clock())
        opener = ChatOpener("One.", "Two.")
        self.ask(conversation, "first", opener)
        conversation.forget()
        self.ask(conversation, "second", opener)
        self.assertEqual(len(opener.bodies[1]["messages"]), 2)


class AssistantAppTests(unittest.TestCase):
    """The daemon side: ask, speak the reply, and report what can't be spoken."""

    @classmethod
    def setUpClass(cls):
        cls.App = _import_main_isolated().hyprwhsprApp
        cls.AssistantError = app_global(cls.App, 'AssistantError')
        cls.SpeechError = app_global(cls.App, 'SpeechError')

    def app(self, reply="Paris, naturally.", error=None, **values):
        app = self.App.__new__(self.App)
        app.config = mock.Mock()
        app.config.get_setting.side_effect = lambda key, default=None: values.get(key, default)
        app.is_recording = False
        app._voice_start_phrase, app._voice_end_phrase = 'hey jarvis', 'hey mycroft'
        app.audio_manager = mock.Mock()
        app._notify_user = mock.Mock()
        app._assistant = mock.Mock()
        app._assistant.ask.side_effect = error
        app._assistant.ask.return_value = reply
        app._assistant_speech = mock.Mock()
        return app

    def test_the_reply_is_spoken_in_the_assistant_voice(self):
        app = self.app(assistant_voice='Lily')
        self.assertTrue(app._assistant_respond('capital of France?'))
        args, kwargs = app._assistant.ask.call_args
        self.assertEqual(args[0], 'capital of France?')
        self.assertEqual(kwargs['avoid_phrases'], ('hey jarvis', 'hey mycroft'))
        app._assistant_speech.speak.assert_called_once_with('Paris, naturally.', 'lily')
        app._notify_user.assert_not_called()

    def test_no_answer_is_reported(self):
        app = self.app(error=self.AssistantError("no API key stored for OpenRouter"))
        self.assertFalse(app._assistant_respond('hello'))
        app.audio_manager.play_error_sound.assert_called_once()
        self.assertIn('no API key', app._notify_user.call_args.args[1])
        app._assistant_speech.speak.assert_not_called()

    def test_a_cancelled_question_is_not_answered(self):
        app = self.app()
        self.assertFalse(app._assistant_respond('hello', still_wanted=lambda: False))
        app._assistant_speech.speak.assert_not_called()
        app._notify_user.assert_not_called()

    def test_replies_that_cannot_be_spoken_are_shown(self):
        app = self.app(reply="Use <b> & smile.")
        app._assistant_speech.speak.side_effect = self.SpeechError('no audio player found')
        self.assertFalse(app._assistant_respond('hello'))
        body = app._notify_user.call_args.args[1]
        self.assertIn('Use &lt;b&gt; &amp; smile.', body)
        self.assertIn('no audio player found', body)

        # Still recording (continuous mode): the microphone would hear the reply
        app = self.app()
        app.is_recording = True
        self.assertTrue(app._assistant_respond('hello'))
        app._assistant_speech.speak.assert_not_called()
        self.assertIn('Paris, naturally.', app._notify_user.call_args.args[1])

    def test_the_conversation_and_player_are_made_on_first_use(self):
        app = self.App.__new__(self.App)
        app.config = mock.Mock()
        self.assertIs(app._assistant_conversation(), app._assistant_conversation())
        self.assertFalse(app._stop_assistant_speech())
        player = mock.Mock()
        with patch_app_global(self.App, 'SpeechPlayer', player):
            reader = app._assistant_reader()
        self.assertIs(reader, app._assistant_reader())
        self.assertEqual(player.call_args.kwargs['name'], 'AssistantSpeech')


if __name__ == "__main__":
    unittest.main()
