"""Scratchpad preview panel: the view model, the daemon-side runner, and the
app hooks that publish the draft while it is being written."""

import json
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.scratchpad_panel_runner import ScratchpadPanelRunner
from mic_osd.scratchpad_view import PanelState, changed_spans, panel_css, render_markup

from tests.test_scratchpad_trigger import ScratchpadConfig
from tests.test_suspend_resume_recovery import _import_main_isolated, app_global, patch_app_global

MARK = ("[", "]")


class PanelStateTests(unittest.TestCase):
    def test_malformed_states_read_as_hidden(self):
        for data in (None, [], {"state": "nope"}, {"draft": "x"}):
            with self.subTest(data=data):
                self.assertEqual(PanelState.from_dict(data).state, "hidden")
        state = PanelState.from_dict({"state": "updated", "draft": 3, "previous": 4, "hold": True, "at": "x"})
        self.assertEqual((state.draft, state.previous, state.hold, state.at), ("", None, None, 0.0))

    def test_held_states_expire_and_active_ones_stay(self):
        held = PanelState.from_dict({"state": "updated", "draft": "a b", "hold": 8, "at": 100.0})
        self.assertTrue(held.visible(now=107.9))
        self.assertFalse(held.visible(now=108.1))
        self.assertAlmostEqual(held.hides_in(now=105.0), 3.0)
        listening = PanelState.from_dict({"state": "listening", "draft": "", "at": 100.0})
        self.assertTrue(listening.visible(now=10_000.0))
        self.assertIsNone(listening.hides_in(now=10_000.0))
        self.assertFalse(PanelState().visible())

    def test_status_and_word_count(self):
        error = PanelState.from_dict({"state": "error", "draft": "one two three", "message": "request timed out"})
        self.assertEqual(error.status, "Edit failed: request timed out")
        self.assertEqual(error.words, 3)
        self.assertEqual(PanelState.from_dict({"state": "updating"}).status, "Updating draft…")


class ChangeHighlightTests(unittest.TestCase):
    def test_appended_text_is_highlighted(self):
        markup, first = render_markup("Fix the parser.", "Fix the parser. Add tests.", MARK)
        self.assertEqual(markup, "Fix the parser. [Add tests.]")
        self.assertEqual(first, len("Fix the parser."))

    def test_a_replaced_word_is_highlighted_alone(self):
        markup, _ = render_markup("Use Postgres for storage.", "Use SQLite for storage.", MARK)
        self.assertEqual(markup, "Use [SQLite] for storage.")

    def test_a_deletion_marks_where_it_happened_without_highlight(self):
        spans, first = changed_spans("Keep this. Drop this.", "Keep this.")
        self.assertEqual(spans, [])
        self.assertEqual(first, len("Keep this."))

    def test_first_draft_and_unchanged_drafts_are_not_highlighted(self):
        for previous in (None, "", "Same."):
            with self.subTest(previous=previous):
                self.assertEqual(render_markup(previous, "Same.", MARK), ("Same.", None))

    def test_markup_is_escaped_and_offsets_are_utf8_bytes(self):
        markup, first = render_markup("Grüße <b>", "Grüße <b> & mehr", MARK)
        self.assertEqual(markup, "Grüße &lt;b&gt; [&amp; mehr]")
        self.assertEqual(first, len("Grüße <b>".encode("utf-8")))

    def test_css_uses_the_theme_colors(self):
        theme = SimpleNamespace(
            background=(0.1, 0.1, 0.15, 0.95), border=(0.2, 0.8, 1.0), text=(1.0, 1.0, 1.0, 1.0),
            bar_left=(0.2, 0.8, 1.0), recording_dot=(1.0, 0.2, 0.33, 1.0), paused_dot=(1.0, 0.75, 0.0, 1.0),
            processing_dot=(0.2, 0.9, 0.4, 1.0), error_dot=(1.0, 0.2, 0.2, 1.0), success_dot=(0.0, 1.0, 0.5, 1.0),
        )
        css = panel_css(theme)
        self.assertIn("background-color: rgba(26, 26, 38, 0.95)", css)
        self.assertIn(".scratchpad-dot.error { color: rgba(255, 51, 51, 1.00); }", css)


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class PanelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.state_file = root / "runtime" / "scratchpad_preview.json"
        self.processes = []

        def popen(args, **kwargs):
            self.processes.append((args, kwargs))
            process = FakeProcess()
            self.processes[-1] = (args, kwargs, process)
            return process

        self.runner = ScratchpadPanelRunner(self.state_file, root / "panel.log", popen=popen)
        patcher = mock.patch("mic_osd.scratchpad_panel_runner.MicOSDRunner._layer_shell_environment",
                             return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def published(self):
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def test_publish_writes_a_private_state_and_starts_the_panel_once(self):
        self.assertTrue(self.runner.publish("listening", "draft", hint="paste key"))
        self.assertTrue(self.runner.publish("updating", "draft"))
        state = self.published()
        self.assertEqual((state["state"], state["draft"], state["hint"]), ("updating", "draft", ""))
        self.assertEqual(stat.S_IMODE(self.state_file.stat().st_mode), 0o600)
        self.assertEqual(len(self.processes), 1)
        args, kwargs, _ = self.processes[0]
        self.assertIn("from mic_osd.scratchpad_panel import main", args[-1])
        self.assertTrue(kwargs["start_new_session"])
        self.assertIn("HYPRWHSPR_PARENT_PID", kwargs["env"])

    def test_hidden_never_starts_the_panel(self):
        self.assertTrue(self.runner.publish("hidden", ""))
        self.assertEqual(self.processes, [])

    def test_highlight_is_kept_while_the_draft_is_unchanged(self):
        self.runner.publish("updated", "new", previous="old")
        self.runner.publish("listening", "new")
        self.assertEqual(self.published()["previous"], "old")
        self.runner.publish("listening", "newer")
        self.assertIsNone(self.published()["previous"])

    def test_an_idle_exit_restarts_and_crashes_give_up(self):
        self.runner.publish("listening", "")
        self.processes[-1][2].returncode = 0  # quit after idling
        self.assertTrue(self.runner.publish("listening", ""))
        self.assertEqual(len(self.processes), 2)
        for _ in range(ScratchpadPanelRunner.MAX_FAILURES):
            self.processes[-1][2].returncode = 1
            self.runner.publish("listening", "")
        self.assertTrue(self.runner.disabled)
        self.assertFalse(self.runner.publish("updated", "x"))

    def test_stop_quits_the_panel_and_removes_the_draft_copy(self):
        self.runner.publish("updated", "secret draft")
        process = self.processes[0][2]
        self.runner.stop()
        self.assertTrue(process.terminated)
        self.assertFalse(self.state_file.exists())
        self.assertEqual(self.runner.state, "hidden")

    def test_unknown_states_are_a_programming_error(self):
        with self.assertRaises(ValueError):
            self.runner.publish("dancing", "")


class AppPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()
        cls.llm_refiner = app_global(cls.main.hyprwhsprApp, "llm_refiner")

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        log = patch_app_global(self.main.hyprwhsprApp, "log", mock.Mock())
        log.__enter__()
        self.addCleanup(log.__exit__, None, None, None)
        patcher = mock.patch("mic_osd.scratchpad_panel_runner.MicOSDRunner._layer_shell_environment",
                             return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _app(self, route="scratchpad", **settings):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = ScratchpadConfig({"recording_mode": "toggle", **settings})
        app._current_route = route
        app.is_recording = True
        app._longform = mock.Mock(route="paste")
        app._mic_osd_runner = None
        app._cancel_pending_hide_lock = threading.Lock()
        app._scratchpad = self.main.Scratchpad(Path(self.tempdir.name) / "scratchpad.md")
        app._scratchpad_notification_id = None
        app._scratchpad_panel = ScratchpadPanelRunner(
            Path(self.tempdir.name) / "preview.json", Path(self.tempdir.name) / "panel.log",
            popen=lambda *a, **k: FakeProcess())
        app._notify_user = mock.Mock()
        app._notify_scratchpad = mock.Mock()
        app.text_injector = mock.Mock()
        app.text_injector.deliver_prepared.return_value = (True, "Paste dispatched")
        app._recording_control_server = mock.Mock()
        app._recording_control_server.has_capture_subscriber.return_value = False
        app._mark_delivery_healthy = mock.Mock()
        return app

    def states(self, app):
        return app._scratchpad_panel.last and app._scratchpad_panel.last["state"]

    def test_a_scratchpad_recording_is_followed_on_the_panel(self):
        app = self._app()
        app._scratchpad.set("Fix the parser.")
        app._show_mic_osd()
        self.assertEqual(app._scratchpad_panel.last["state"], "listening")
        self.assertEqual(app._scratchpad_panel.last["draft"], "Fix the parser.")
        app._set_visualizer_state("processing")
        self.assertEqual(self.states(app), "transcribing")
        app._set_visualizer_state("success")  # a result comes from the edit, not the visualizer
        self.assertEqual(self.states(app), "transcribing")

    def test_other_recordings_leave_the_panel_alone(self):
        app = self._app(route="paste")
        app._show_mic_osd()
        app._set_visualizer_state("processing")
        self.assertIsNone(app._scratchpad_panel.last)

    def test_long_form_follows_the_session_route(self):
        app = self._app(route="paste", recording_mode="long_form")
        app._longform.route = "scratchpad"
        app._set_visualizer_state("paused")
        self.assertEqual(self.states(app), "paused")

    def test_an_edit_shows_updating_then_the_highlighted_result(self):
        app = self._app(scratchpad_preview_seconds=5)
        app.is_recording = False
        app._scratchpad.set("Fix the parser.")
        published = []
        original = app._scratchpad_panel.publish
        app._scratchpad_panel.publish = lambda state, draft, **kw: published.append(state) or original(
            state, draft, **kw)
        with mock.patch.object(self.llm_refiner, "edit_draft", return_value="Fix the parser. Add tests."):
            app._inject_text("also add tests", route="scratchpad")
        self.assertEqual(published, ["updating", "updated"])
        last = app._scratchpad_panel.last
        self.assertEqual((last["draft"], last["previous"], last["hold"]),
                         ("Fix the parser. Add tests.", "Fix the parser.", 5.0))
        self.assertIn("scratchpad paste", last["hint"])
        app._notify_scratchpad.assert_not_called()

    def test_continuous_segments_keep_listening_and_the_end_shows_the_result(self):
        app = self._app(recording_mode="continuous")
        app._scratchpad.set("One.")
        with mock.patch.object(self.llm_refiner, "edit_draft", return_value="One. Two."):
            app._inject_text("two", route="scratchpad")
        self.assertEqual(self.states(app), "listening")
        app._hide_mic_osd()
        last = app._scratchpad_panel.last
        self.assertEqual((last["state"], last["previous"]), ("updated", "One."))

    def test_a_recording_that_never_reached_the_llm_hides_the_panel(self):
        app = self._app()
        app._show_mic_osd()
        app._hide_mic_osd()
        self.assertEqual(self.states(app), "hidden")

    def test_a_result_survives_the_osd_hiding(self):
        app = self._app()
        app._scratchpad_panel.publish("updated", "draft", previous="d", hold=8)
        app._hide_mic_osd()
        self.assertEqual(self.states(app), "updated")

    def test_a_failed_edit_is_shown_on_the_panel_and_notified(self):
        app = self._app()
        app._scratchpad.set("Keep me.")
        with mock.patch.object(self.llm_refiner, "edit_draft",
                               side_effect=self.llm_refiner.LLMRefineError("request timed out")):
            app._inject_text("drop it", route="scratchpad")
        last = app._scratchpad_panel.last
        self.assertEqual((last["state"], last["message"], last["draft"]), ("error", "request timed out", "Keep me."))
        app._notify_user.assert_called_once()

    def test_paste_copy_clear_and_undo_are_shown_briefly(self):
        app = self._app(scratchpad_preview_seconds=0)
        app._scratchpad.set("Prompt.")
        app._handle_scratchpad_request("scratchpad_copy")
        self.assertEqual((self.states(app), app._scratchpad_panel.last["hold"]), ("copied", 1.5))
        app._handle_scratchpad_request("scratchpad_paste")
        self.assertEqual((self.states(app), app._scratchpad_panel.last["draft"]), ("pasted", "Prompt."))
        app._handle_scratchpad_request("scratchpad_undo")
        last = app._scratchpad_panel.last
        self.assertEqual((last["state"], last["draft"], last["previous"], last["hold"]),
                         ("updated", "Prompt.", "", None))  # 0: stays until paste or clear
        app._handle_scratchpad_request("scratchpad_clear")
        self.assertEqual(self.states(app), "cleared")
        app._notify_scratchpad.assert_not_called()

    def test_without_a_panel_changes_go_to_notifications(self):
        app = self._app()
        app._scratchpad_panel = False
        app._scratchpad.set("Fix the parser.")
        with mock.patch.object(self.llm_refiner, "edit_draft", return_value="Fix the parser. Add tests."):
            app._inject_text("also add tests", route="scratchpad")
        app._notify_scratchpad.assert_called_once_with("Fix the parser. Add tests.")

    def test_the_panel_is_created_once_and_only_when_enabled(self):
        app = self._app(scratchpad_preview=False)
        app._scratchpad_panel = None
        self.assertIsNone(app._scratchpad_panel_runner())
        self.assertIs(app._scratchpad_panel, False)

        app = self._app()
        app._scratchpad_panel = None
        with mock.patch("mic_osd.ScratchpadPanelRunner.is_supported", return_value=False) as supported:
            self.assertIsNone(app._scratchpad_panel_runner())
            self.assertIsNone(app._scratchpad_panel_runner())
        supported.assert_called_once_with()

        app = self._app()
        app._scratchpad_panel = None
        with mock.patch("mic_osd.ScratchpadPanelRunner.is_supported", return_value=True):
            runner = app._scratchpad_panel_runner()
        self.assertEqual(type(runner).__name__, "ScratchpadPanelRunner")
        self.assertIs(app._scratchpad_panel_runner(), runner)


if __name__ == "__main__":
    unittest.main()
