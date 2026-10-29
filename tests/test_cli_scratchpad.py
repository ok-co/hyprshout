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

    def test_edit_creates_a_private_file_and_opens_the_editor(self):
        with mock.patch.dict(os.environ, {"EDITOR": "vi"}), \
                mock.patch.object(cli_scratchpad.subprocess, "run") as run:
            cli_scratchpad.scratchpad_command("edit")
        run.assert_called_once_with(["vi", str(self.draft_file)], check=True)
        self.assertEqual(stat.S_IMODE(self.draft_file.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
