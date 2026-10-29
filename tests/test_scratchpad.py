import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))

from scratchpad import Scratchpad, ScratchpadConflict


class ScratchpadStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "runtime" / "scratchpad.md"
        self.pad = Scratchpad(self.path, history_limit=3)

    def test_missing_file_is_an_empty_draft(self):
        self.assertEqual(self.pad.read(), "")
        self.assertFalse(self.pad.can_undo)

    def test_apply_passes_draft_and_dictation_and_stores_the_result(self):
        calls = []

        def edit(draft, dictation):
            calls.append((draft, dictation))
            return f"{draft} {dictation}".strip() + "\n\n"

        self.assertEqual(self.pad.apply("Fix the parser.", edit), "Fix the parser.")
        self.assertEqual(self.pad.apply("Add tests.", edit), "Fix the parser. Add tests.")
        self.assertEqual(calls, [("", "Fix the parser."), ("Fix the parser.", "Add tests.")])
        self.assertEqual(self.path.read_text(encoding="utf-8"), "Fix the parser. Add tests.\n")

    def test_draft_file_is_private_and_written_atomically(self):
        self.pad.set("secret prompt")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["scratchpad.md"])

    def test_failed_edit_leaves_the_draft_and_history_unchanged(self):
        self.pad.set("Keep me.")

        def broken(draft, dictation):
            raise RuntimeError("provider down")

        with self.assertRaises(RuntimeError):
            self.pad.apply("change everything", broken)
        self.assertEqual(self.pad.read(), "Keep me.")
        self.assertEqual(self.pad.undo(), "")
        self.assertIsNone(self.pad.undo())

    def test_a_manual_edit_during_the_call_is_never_overwritten(self):
        self.pad.set("v1")

        def edit(draft, dictation):
            self.path.write_text("typed in $EDITOR\n", encoding="utf-8")
            return "v2"

        with self.assertRaises(ScratchpadConflict):
            self.pad.apply("more", edit)
        self.assertEqual(self.pad.read(), "typed in $EDITOR")

    def test_manual_edits_are_read_before_the_next_change(self):
        self.pad.set("v1")
        self.path.write_text("v1, edited by hand\n", encoding="utf-8")
        self.pad.apply("x", lambda draft, dictation: draft + " plus x")
        self.assertEqual(self.pad.read(), "v1, edited by hand plus x")
        self.assertEqual(self.pad.undo(), "v1, edited by hand")

    def test_undo_walks_back_a_bounded_history(self):
        for text in ("a", "b", "c", "d"):
            self.pad.set(text)
        self.assertEqual([self.pad.undo() for _ in range(4)], ["c", "b", "a", None])
        self.assertEqual(self.pad.read(), "a")

    def test_clear_removes_the_file_and_can_be_undone(self):
        self.assertFalse(self.pad.clear())
        self.pad.set("draft")
        self.assertTrue(self.pad.clear())
        self.assertFalse(self.path.exists())
        self.assertEqual(self.pad.undo(), "draft")
        self.assertEqual(self.pad.read(), "draft")

    def test_an_unchanged_result_adds_no_history(self):
        self.pad.set("same")
        self.pad.apply("noop", lambda draft, dictation: draft)
        self.assertEqual(self.pad.undo(), "")
        self.assertIsNone(self.pad.undo())

    def test_edits_run_one_at_a_time_in_order(self):
        started = threading.Event()
        release = threading.Event()

        def slow(draft, dictation):
            started.set()
            release.wait(2)
            return draft + "1"

        worker = threading.Thread(target=self.pad.apply, args=("one", slow))
        worker.start()
        self.assertTrue(started.wait(2))
        second = threading.Thread(target=self.pad.apply, args=("two", lambda draft, dictation: draft + "2"))
        second.start()
        release.set()
        worker.join(2)
        second.join(2)
        self.assertEqual(self.pad.read(), "12")


if __name__ == "__main__":
    unittest.main()
