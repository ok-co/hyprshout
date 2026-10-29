"""CLI scratchpad commands: file reads work without the daemon; changes go through it."""
import io
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))

from cli import scratchpad as cli_scratchpad
from scratchpad import Scratchpad


class ScratchpadCliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.draft_file = root / "runtime" / "scratchpad.md"
        self.socket_file = root / "hyprwhspr.sock"
        for name, value in (("SCRATCHPAD_FILE", self.draft_file), ("SOCKET_FILE", self.socket_file)):
            patcher = mock.patch.object(cli_scratchpad, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _serve(self, response):
        """Answer one request on the socket; returns the list the request lands in."""
        received = []
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_file))
        server.listen(1)
        self.addCleanup(server.close)

        def answer():
            conn, _ = server.accept()
            with conn:
                data = b""
                while b"\n" not in data:
                    data += conn.recv(4096)
                received.append(json.loads(data))
                conn.sendall(json.dumps(response).encode() + b"\n")

        worker = threading.Thread(target=answer, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        return received

    def test_show_prints_the_draft_without_the_daemon(self):
        Scratchpad(self.draft_file).set("line one\nline two")
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            cli_scratchpad.scratchpad_command("show")
        self.assertEqual(out.getvalue(), "line one\nline two\n")

    def test_empty_show_keeps_stdout_clean(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            cli_scratchpad.scratchpad_command("show")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("empty", err.getvalue())

    def test_daemon_actions_send_their_verb(self):
        for action in cli_scratchpad.DAEMON_ACTIONS:
            with self.subTest(action=action):
                received = self._serve({"ok": True, "message": "done"})
                with mock.patch.object(cli_scratchpad, "log_success") as success:
                    cli_scratchpad.scratchpad_command(action)
                self.assertEqual(received, [{"verb": f"scratchpad_{action}"}])
                success.assert_called_once_with("done")
                self.socket_file.unlink()

    def test_daemon_error_exits_non_zero(self):
        self._serve({"ok": False, "error": "The draft is empty"})
        with mock.patch.object(cli_scratchpad, "log_error") as log_error, \
                self.assertRaises(SystemExit) as caught:
            cli_scratchpad.scratchpad_command("paste")
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("The draft is empty", log_error.call_args[0][0])

    def test_missing_daemon_is_reported(self):
        with mock.patch.object(cli_scratchpad, "log_error") as log_error, \
                self.assertRaises(SystemExit):
            cli_scratchpad.scratchpad_command("clear")
        self.assertIn("daemon is running", log_error.call_args[0][0])

    def test_speak_sends_a_voice_for_this_reading(self):
        received = self._serve({"ok": True, "message": "Reading the draft aloud (lily)"})
        with mock.patch.object(cli_scratchpad, "log_success"):
            cli_scratchpad.scratchpad_command("speak", "Lily")
        self.assertEqual(received, [{"verb": "scratchpad_speak", "voice": "lily"}])

    def test_unknown_voice_or_stray_argument_exits_non_zero(self):
        for action, value in (("speak", "nobody"), ("voice", "nobody"), ("paste", "lily")):
            with self.subTest(action=action), mock.patch.object(cli_scratchpad, "log_error"), \
                    self.assertRaises(SystemExit) as caught:
                cli_scratchpad.scratchpad_command(action, value)
            self.assertEqual(caught.exception.code, 1)

    def test_voice_saves_the_choice_and_tells_the_daemon(self):
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: default
        config.save_config.return_value = True
        received = self._serve({"ok": True, "message": "ok"})
        with mock.patch.object(cli_scratchpad, "ConfigManager", return_value=config), \
                mock.patch.object(cli_scratchpad, "log_success"):
            cli_scratchpad.scratchpad_command("voice", "George")
        config.set_setting.assert_called_once_with("scratchpad_voice", "george")
        config.save_config.assert_called_once_with()
        self.assertEqual(received, [{"verb": "scratchpad_voice", "voice": "george"}])

    def test_voice_lists_the_voices_and_marks_the_current_one(self):
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: "bella" if key == "scratchpad_voice" else default
        with mock.patch.object(cli_scratchpad, "ConfigManager", return_value=config), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            cli_scratchpad.scratchpad_command("voice")
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 21)
        self.assertIn("* bella", lines)
        self.assertIn("  sarah", lines)

    def test_edit_creates_a_private_file_and_opens_the_editor(self):
        with mock.patch.dict(os.environ, {"EDITOR": "vi"}), \
                mock.patch.object(cli_scratchpad.subprocess, "run") as run:
            cli_scratchpad.scratchpad_command("edit")
        run.assert_called_once_with(["vi", str(self.draft_file)], check=True)
        self.assertEqual(stat.S_IMODE(self.draft_file.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
