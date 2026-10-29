import importlib
import io
import json
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import llm_refiner
from llm_refiner import DEFAULT_PROMPT, LLMRefineError, build_system_prompt, refine_text, resolve_target
from realtime_protocols import live_typing_conflict
from text_injector import InjectionOutcome, _PostTranscriptionHookOutcome

from tests.text_injector_helpers import ConfigStub, make_injector


KEY = "sk-or-v1-secretsecretsecret"


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


def reply(content, finish_reason="stop"):
    return FakeResponse({"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]})


class RecordingOpener:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else reply("Cleaned.")
        self.error = error
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def body(self):
        return json.loads(self.requests[-1][0].data)

    def header(self, name):
        return self.requests[-1][0].get_header(name.capitalize())


class ResolveTargetTests(unittest.TestCase):
    def test_openrouter_is_the_default_with_its_default_model(self):
        target = resolve_target(settings())
        self.assertEqual(target["provider_id"], "openrouter")
        self.assertEqual(target["endpoint"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(target["model"], "anthropic/claude-haiku-5.5")
        self.assertEqual(target["credential"], "openrouter")

    def test_configured_model_overrides_the_default(self):
        self.assertEqual(resolve_target(settings(llm_refine_model=" openai/gpt-5.4-nano "))["model"],
                         "openai/gpt-5.4-nano")

    def test_custom_needs_a_valid_url_and_a_model(self):
        with self.assertRaisesRegex(LLMRefineError, "endpoint_url is required"):
            resolve_target(settings(llm_refine_provider="custom", llm_refine_model="m"))
        with self.assertRaisesRegex(LLMRefineError, "http"):
            resolve_target(settings(llm_refine_provider="custom", llm_refine_model="m",
                                    llm_refine_endpoint_url="localhost:11434"))
        with self.assertRaisesRegex(LLMRefineError, "llm_refine_model"):
            resolve_target(settings(llm_refine_provider="custom",
                                    llm_refine_endpoint_url="http://localhost:11434/v1/chat/completions"))
        target = resolve_target(settings(llm_refine_provider="custom", llm_refine_model="m",
                                         llm_refine_endpoint_url="http://localhost:11434/v1/chat/completions"))
        self.assertEqual(target["credential"], "llm-custom")

    def test_unknown_provider_is_an_error(self):
        with self.assertRaisesRegex(LLMRefineError, "unknown"):
            resolve_target(settings(llm_refine_provider="nope"))


class SystemPromptTests(unittest.TestCase):
    def test_default_prompt_covers_self_corrections_and_does_not_answer(self):
        prompt = build_system_prompt(settings())
        self.assertEqual(prompt, DEFAULT_PROMPT)
        self.assertIn("self-corrections", prompt)
        self.assertIn("never answer it", prompt)

    def test_inline_prompt_replaces_default_and_extra_is_appended(self):
        prompt = build_system_prompt(settings(llm_refine_prompt="Make it formal.",
                                              llm_refine_extra_instructions="Use British spelling."))
        self.assertTrue(prompt.startswith("Make it formal."))
        self.assertNotIn("dictation editor", prompt)
        self.assertTrue(prompt.endswith("Use British spelling."))

    def test_prompt_file_wins_and_is_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text("First prompt\n", encoding="utf-8")
            get = settings(llm_refine_prompt_file=str(path), llm_refine_prompt="inline")
            self.assertEqual(build_system_prompt(get), "First prompt")
            path.write_text("Second prompt", encoding="utf-8")
            self.assertEqual(build_system_prompt(get), "Second prompt")

    def test_unreadable_or_empty_prompt_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(LLMRefineError, "could not read"):
                build_system_prompt(settings(llm_refine_prompt_file=str(Path(tmp) / "missing")))
            empty = Path(tmp) / "empty"
            empty.write_text("  \n", encoding="utf-8")
            with self.assertRaisesRegex(LLMRefineError, "empty"):
                build_system_prompt(settings(llm_refine_prompt_file=str(empty)))


class RefineTextTests(unittest.TestCase):
    def refine(self, text="so um hello", opener=None, key=KEY, **values):
        opener = opener or RecordingOpener()
        result = refine_text(text, settings(**values), credential_lookup=lambda name: key, opener=opener)
        return result, opener

    def test_sends_openai_compatible_request_to_openrouter(self):
        result, opener = self.refine(llm_refine_timeout=7)
        self.assertEqual(result, "Cleaned.")
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(timeout, 7.0)
        self.assertEqual(opener.header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(opener.header("X-title"), "hyprwhspr")
        body = opener.body
        self.assertEqual(body["model"], "anthropic/claude-haiku-5.5")
        self.assertEqual(body["messages"][0], {"role": "system", "content": DEFAULT_PROMPT})
        self.assertEqual(body["messages"][1]["content"], "<transcript>\nso um hello\n</transcript>")
        self.assertNotIn("temperature", body)

    def test_temperature_is_sent_only_when_configured(self):
        _, opener = self.refine(llm_refine_temperature=0)
        self.assertEqual(opener.body["temperature"], 0.0)

    def test_invalid_timeout_falls_back_to_default(self):
        for value in (0, -1, "abc", None, 1000):
            with self.subTest(value=value):
                _, opener = self.refine(llm_refine_timeout=value)
                self.assertEqual(opener.requests[0][1], float(llm_refiner.DEFAULT_TIMEOUT))

    def test_echoed_wrapper_and_whitespace_are_stripped(self):
        result, _ = self.refine(opener=RecordingOpener(reply("  <transcript>\nHello.\n</transcript>\n")))
        self.assertEqual(result, "Hello.")

    def test_list_content_parts_are_joined(self):
        response = FakeResponse({"choices": [{"message": {"content": [
            {"type": "text", "text": "Hello, "}, {"type": "text", "text": "world."}]}}]})
        result, _ = self.refine(opener=RecordingOpener(response))
        self.assertEqual(result, "Hello, world.")

    def test_missing_openrouter_key_fails_without_a_request(self):
        opener = RecordingOpener()
        with self.assertRaisesRegex(LLMRefineError, "no API key"):
            self.refine(opener=opener, key=None)
        self.assertEqual(opener.requests, [])

    def test_custom_endpoint_works_without_a_key(self):
        result, opener = self.refine(key=None, llm_refine_provider="custom", llm_refine_model="llama3",
                                     llm_refine_endpoint_url="http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(result, "Cleaned.")
        self.assertIsNone(opener.header("Authorization"))
        self.assertIsNone(opener.header("X-title"))

    def test_key_is_never_sent_over_plain_http_to_a_remote_host(self):
        opener = RecordingOpener()
        with self.assertRaisesRegex(LLMRefineError, "plain http"):
            self.refine(opener=opener, llm_refine_provider="custom", llm_refine_model="m",
                        llm_refine_endpoint_url="http://llm.example.com/v1/chat/completions")
        self.assertEqual(opener.requests, [])
        for host in ("localhost", "127.0.0.1", "[::1]"):
            with self.subTest(host=host):
                _, opener = self.refine(llm_refine_provider="custom", llm_refine_model="m",
                                        llm_refine_endpoint_url=f"http://{host}:8080/v1/chat/completions")
                self.assertEqual(opener.header("Authorization"), f"Bearer {KEY}")

    def test_http_error_reports_provider_message_without_the_key(self):
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions", 401, "Unauthorized", {},
            io.BytesIO(json.dumps({"error": {"message": "User not found."}}).encode()))
        with self.assertRaises(LLMRefineError) as caught:
            self.refine(opener=RecordingOpener(error=error))
        self.assertEqual(str(caught.exception), "HTTP 401: User not found.")
        self.assertNotIn(KEY, str(caught.exception))

    def test_transport_failures_become_refine_errors(self):
        for error, pattern in (
            (urllib.error.URLError("Name or service not known"), "cannot reach"),
            (TimeoutError(), "timed out"),
            (ConnectionResetError(), "ConnectionResetError"),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(LLMRefineError, pattern):
                    self.refine(opener=RecordingOpener(error=error))

    def test_unusable_responses_are_rejected(self):
        for response, pattern in (
            (reply("Half of the dictat", finish_reason="length"), "cut off"),
            (reply("   "), "empty"),
            (reply(None), "no text"),
            (FakeResponse({"error": "nope"}), "shape"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(LLMRefineError, pattern):
                    self.refine(opener=RecordingOpener(response))


class InjectorRefinementTests(unittest.TestCase):
    def test_disabled_refinement_makes_no_call(self):
        injector = make_injector()
        injector.config_manager = ConfigStub({"llm_refine_enabled": False})
        with mock.patch("text_injector.llm_refiner.refine_text") as refine:
            self.assertEqual(injector._refine_with_llm("hello"), "hello")
        refine.assert_not_called()

    def test_enabled_refinement_replaces_text(self):
        injector = make_injector()
        injector.config_manager = ConfigStub({"llm_refine_enabled": True})
        with mock.patch("text_injector.llm_refiner.refine_text", return_value="Hello.") as refine:
            self.assertEqual(injector._refine_with_llm("hello"), "Hello.")
        refine.assert_called_once_with("hello", injector.config_manager.get_setting)

    def test_failure_keeps_original_and_logs_no_dictation(self):
        injector = make_injector()
        injector.config_manager = ConfigStub({"llm_refine_enabled": True})
        for error in (LLMRefineError("HTTP 500"), RuntimeError("private words")):
            with self.subTest(error=error):
                with mock.patch("text_injector.llm_refiner.refine_text", side_effect=error), \
                        mock.patch("text_injector.log") as log:
                    self.assertEqual(injector._refine_with_llm("private words"), "private words")
                self.assertNotIn("private words", log.call_args[0][0])

    def test_refined_text_reaches_the_hook_and_the_paste(self):
        injector = make_injector()
        injector.config_manager = ConfigStub({"llm_refine_enabled": True, "append_trailing_space": False})
        preserved = types.SimpleNamespace(outcome=_PostTranscriptionHookOutcome.PRESERVE, text="Hello there.")
        with mock.patch("text_injector.llm_refiner.refine_text", return_value="Hello there."), \
                mock.patch.object(injector, "_run_post_transcription_hook", return_value=preserved) as hook, \
                mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", return_value=True) as inject:
            self.assertEqual(injector.inject_text("um hello there"), InjectionOutcome.INJECTED)
        hook.assert_called_once_with("Hello there.")
        inject.assert_called_once_with("Hello there.", retain=True)

    def test_live_typing_is_off_while_refinement_is_enabled(self):
        values = {"transcription_backend": "realtime-ws", "websocket_provider": "custom",
                  "websocket_live_text": "append_only", "llm_refine_enabled": True}
        self.assertIn("llm_refine_enabled", live_typing_conflict(lambda key, default=None: values.get(key, default)))


def _stub_if_missing(name, **attrs):
    try:
        importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing("rich")
_stub_if_missing("rich.prompt", Prompt=object, Confirm=object)

from cli import llm  # noqa: E402


class LlmCliTests(unittest.TestCase):
    def config(self, **values):
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: values.get(key, default)
        config.save_config.return_value = True
        return config

    def test_prompt_command_seeds_file_from_builtin_prompt_and_points_config_at_it(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "llm_prompt.md"
            with mock.patch.object(llm, "ConfigManager", return_value=config), \
                    mock.patch.object(llm, "DEFAULT_PROMPT_FILE", prompt_file), \
                    mock.patch.object(llm.subprocess, "run") as run, \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                llm.edit_llm_prompt()
            self.assertEqual(prompt_file.read_text(encoding="utf-8"), DEFAULT_PROMPT + "\n")
        config.set_setting.assert_called_once_with("llm_refine_prompt_file", str(prompt_file))
        config.save_config.assert_called_once()
        self.assertEqual(run.call_args[0][0][1], str(prompt_file))

    def test_prompt_command_keeps_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "mine.md"
            prompt_file.write_text("Mine", encoding="utf-8")
            config = self.config(llm_refine_prompt_file=str(prompt_file))
            with mock.patch.object(llm, "ConfigManager", return_value=config), \
                    mock.patch.object(llm.subprocess, "run"), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                llm.edit_llm_prompt()
            self.assertEqual(prompt_file.read_text(encoding="utf-8"), "Mine")
        config.set_setting.assert_not_called()

    def test_test_command_reports_failure(self):
        config = self.config()
        with mock.patch.object(llm, "refine_text", side_effect=LLMRefineError("no API key")), \
                mock.patch.object(llm, "log_error") as log_error, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertFalse(llm.test_llm("hello", config))
        self.assertIn("no API key", log_error.call_args[0][0])

    def test_forget_key_deletes_the_configured_providers_credential(self):
        config = self.config(llm_refine_provider="custom")
        with mock.patch.object(llm, "ConfigManager", return_value=config), \
                mock.patch.object(llm, "delete_credential", return_value=True) as delete, \
                mock.patch.object(llm, "log_success"):
            llm.forget_llm_key()
        delete.assert_called_once_with("llm-custom")

    def test_disable_keeps_settings(self):
        config = self.config(llm_refine_enabled=True)
        with mock.patch.object(llm, "ConfigManager", return_value=config), \
                mock.patch.object(llm, "log_success"), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            llm.disable_llm()
        config.set_setting.assert_called_once_with("llm_refine_enabled", False)


if __name__ == "__main__":
    unittest.main()
