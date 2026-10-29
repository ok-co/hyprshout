import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import voice_command  # noqa: E402
from voice_command import (  # noqa: E402
    ACTIONS, SCRATCHPAD_ACTIONS, DecisionError, DictationPlan, VoiceDecision, build_request,
    decide, parse_response, phrase_for, plan_for, strip_wake_phrase,
)
from text_injector import InjectionOutcome, _PostTranscriptionHookOutcome  # noqa: E402

from tests.text_injector_helpers import ConfigStub, make_injector  # noqa: E402


KEY = "sk-or-v1-secretsecretsecret"
LANGS = ("en", "de")


def settings(**values):
    return lambda key, default=None: values.get(key, default)


def answers(action="dictate", confidence=0.9, spoken="auto", target="none", language_confidence=0.9):
    return {"answers": {
        "action": {"type": "choice", "choice": action, "confidence": confidence},
        "spoken_language": {"type": "choice", "choice": spoken, "confidence": language_confidence},
        "target_language": {"type": "choice", "choice": target, "confidence": language_confidence},
    }}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingOpener:
    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else answers()
        self.error = error
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.payload)


class RequestTests(unittest.TestCase):
    def test_request_offers_every_action_and_the_configured_languages(self):
        body = build_request("translate this into German", settings(voice_control_languages=["en", "de"]))
        self.assertEqual(body["model"], voice_command.DEFAULT_MODEL)
        self.assertEqual(body["state"]["spoken_command"], "translate this into German")
        questions = body["questions"]
        self.assertEqual(questions["action"]["type"], "choice")
        self.assertEqual(set(questions["action"]["criteria"]), set(ACTIONS))
        self.assertEqual(set(questions["spoken_language"]["criteria"]), {"auto", "en", "de"})
        self.assertEqual(set(questions["target_language"]["criteria"]), {"none", "en", "de"})
        self.assertIn("German", questions["target_language"]["criteria"]["de"])

    def test_no_languages_means_no_language_questions(self):
        body = build_request("dictate", settings(voice_control_languages=[]))
        self.assertEqual(set(body["questions"]), {"action"})

    def test_configured_model_is_used(self):
        body = build_request("x", settings(voice_control_decision_model="perplexity/pplx-decider-flash-v1.2"))
        self.assertEqual(body["model"], "perplexity/pplx-decider-flash-v1.2")


class ParseTests(unittest.TestCase):
    def test_confident_answers_become_a_decision(self):
        decision = parse_response(answers("translate", target="de"), LANGS, 0.5)
        self.assertEqual(decision, VoiceDecision("translate", 0.9, None, "de"))

    def test_unsure_action_is_not_understood(self):
        decision = parse_response(answers("dictate", confidence=0.3), LANGS, 0.5)
        self.assertEqual(decision.action, "nothing")
        self.assertTrue(decision.uncertain)

    def test_unsure_language_falls_back_to_automatic(self):
        decision = parse_response(answers("dictate", spoken="de", language_confidence=0.2), LANGS, 0.5)
        self.assertEqual(decision.action, "dictate")
        self.assertIsNone(decision.spoken_language)

    def test_confidence_falls_back_to_the_choice_probability(self):
        data = {"answers": {"action": {"choice": "copy_last", "probabilities": {"copy_last": 0.4}}}}
        self.assertTrue(parse_response(data, (), 0.5).uncertain)

    def test_invalid_answers_are_errors(self):
        for data in ({}, {"answers": []}, {"answers": {"action": {"choice": "rm -rf"}}},
                     answers(spoken="xx")):
            with self.subTest(data=data), self.assertRaises(DecisionError):
                parse_response(data, LANGS, 0.5)


class DecideTests(unittest.TestCase):
    def decide(self, opener, key=KEY, **values):
        return decide("dictate in German", settings(voice_control_languages=["en", "de"], **values),
                      credential_lookup=lambda name: key if name == "openrouter" else None,
                      opener=opener)

    def test_posts_to_the_decisions_api_with_the_openrouter_key(self):
        opener = RecordingOpener(answers("dictate", spoken="de"))
        decision = self.decide(opener)
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, voice_command.DEFAULT_ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(timeout, voice_command.DEFAULT_TIMEOUT)
        self.assertEqual(json.loads(request.data)["state"]["spoken_command"], "dictate in German")
        self.assertEqual(decision.spoken_language, "de")

    def test_missing_key_and_unsafe_endpoints_are_refused_before_sending(self):
        cases = (
            ({"key": ""}, "API key"),
            ({"voice_control_decision_endpoint": "http://example.com/d"}, "plain http"),
            ({"voice_control_decision_endpoint": "ftp://example.com/d"}, "http"),
        )
        for values, message in cases:
            opener = RecordingOpener()
            with self.subTest(values=values), self.assertRaisesRegex(DecisionError, message):
                self.decide(opener, **values)
            self.assertEqual(opener.requests, [])

    def test_local_http_endpoint_is_allowed(self):
        opener = RecordingOpener()
        self.decide(opener, voice_control_decision_endpoint="http://127.0.0.1:8080/decisions")
        self.assertEqual(len(opener.requests), 1)

    def test_transport_errors_become_decision_errors(self):
        http = urllib.error.HTTPError(voice_command.DEFAULT_ENDPOINT, 402, "x", {},
                                      io.BytesIO(b'{"error": {"message": "Insufficient credits"}}'))
        for error, message in ((http, "402: Insufficient credits"),
                               (urllib.error.URLError("down"), "cannot reach"),
                               (TimeoutError(), "timed out")):
            with self.subTest(error=error), self.assertRaisesRegex(DecisionError, message):
                self.decide(RecordingOpener(error=error))


class PlanTests(unittest.TestCase):
    def test_dictation_actions_map_to_routes(self):
        self.assertEqual(plan_for(VoiceDecision("dictate", 1.0)), DictationPlan())
        self.assertEqual(plan_for(VoiceDecision("dictate_raw", 1.0)), DictationPlan(raw=True))
        self.assertEqual(plan_for(VoiceDecision("translate", 1.0)), DictationPlan(route="translate"))
        self.assertEqual(plan_for(VoiceDecision("scratchpad", 1.0)), DictationPlan(route="scratchpad"))
        for action in ("paste_last", "copy_last", "nothing") + SCRATCHPAD_ACTIONS:
            self.assertIsNone(plan_for(VoiceDecision(action, 1.0)), action)

    def test_a_named_translation_language_overrides_the_configured_one(self):
        plan = plan_for(VoiceDecision("translate", 1.0, target_language="de"))
        self.assertEqual(plan, DictationPlan(route="translate", translate_language="German"))

    def test_scratchpad_actions_are_offered_to_the_model(self):
        self.assertTrue(set(SCRATCHPAD_ACTIONS) <= set(ACTIONS))


class TranscriptHelperTests(unittest.TestCase):
    def test_wake_phrases_come_from_names_and_model_files(self):
        self.assertEqual(phrase_for("hey_jarvis"), "hey jarvis")
        self.assertEqual(phrase_for("/models/hey_computer_v0.2.onnx"), "hey computer")

    def test_phrases_are_stripped_only_at_their_own_edge(self):
        cases = (
            ("Take a note. Hey Mycroft.", None, "hey mycroft", "Take a note."),
            ("translate into german hey micro", None, "hey mycroft", "translate into german"),
            ("Hey, Jarvis. Dictate in Polish", "hey jarvis", None, "Dictate in Polish"),
            ("Hey Mycroft is nice", None, "hey mycroft", "Hey Mycroft is nice"),
            ("Alex said hello", "alexa", "alexa", "said hello"),
            ("Hey Mycroft.", None, "hey mycroft", ""),
        )
        for text, start, end, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(strip_wake_phrase(text, start, end), expected)


class InjectorVoiceOptionTests(unittest.TestCase):
    def inject(self, settings, **options):
        injector = make_injector()
        injector.config_manager = ConfigStub({"append_trailing_space": False, **settings})
        preserved = lambda text: mock.Mock(outcome=_PostTranscriptionHookOutcome.PRESERVE, text=text)
        with mock.patch("text_injector.llm_refiner.refine_text", return_value="REFINED") as llm, \
                mock.patch.object(injector, "_run_post_transcription_hook", side_effect=preserved), \
                mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", return_value=True) as paste:
            self.assertEqual(injector.inject_text("hello there", **options), InjectionOutcome.INJECTED)
        return llm, paste

    def test_raw_skips_refinement(self):
        llm, paste = self.inject({"llm_refine_enabled": True}, raw=True)
        llm.assert_not_called()
        paste.assert_called_once_with("hello there", retain=True)

    def test_raw_still_translates(self):
        llm, paste = self.inject({"llm_refine_enabled": False}, translate=True, raw=True)
        llm.assert_called_once()
        paste.assert_called_once_with("REFINED", retain=True)

    def test_translate_language_overrides_the_configured_target(self):
        llm, _paste = self.inject({"llm_translate_language": "English"}, translate=True,
                                  translate_language="German")
        get_setting = llm.call_args.args[1]
        self.assertEqual(get_setting("llm_translate_language"), "German")
        self.assertTrue(llm.call_args.kwargs["translate"])

    def test_without_options_the_configured_target_applies(self):
        llm, _paste = self.inject({"llm_translate_language": "French"}, translate=True)
        self.assertEqual(llm.call_args.args[1]("llm_translate_language"), "French")


if __name__ == "__main__":
    unittest.main()
