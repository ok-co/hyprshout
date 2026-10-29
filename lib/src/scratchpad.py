"""
Scratchpad draft: text built up over several dictations before it is pasted.

The draft lives in one file (paths.SCRATCHPAD_FILE, in the runtime dir: it
survives a service restart, not a logout or reboot). The file is the source
of truth and is re-read before every change, so edits made in $EDITOR
(hyprwhspr scratchpad edit) mix with spoken ones. Writes are atomic and the
file is only readable by the user.

Every change made here (an applied dictation, clear, set) pushes the draft it
replaced onto a bounded in-memory history that undo pops.

Stdlib only, so core stays importable without desktop or LLM extras.
"""

import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional

try:
    from .paths import SCRATCHPAD_FILE
except ImportError:
    from paths import SCRATCHPAD_FILE


HISTORY_LIMIT = 20


class ScratchpadConflict(Exception):
    """The draft file changed while an edit was in flight; it was left as is."""


class Scratchpad:
    """The draft file plus its undo history. Thread-safe."""

    def __init__(self, path=None, history_limit: int = HISTORY_LIMIT):
        self.path = Path(path) if path is not None else SCRATCHPAD_FILE
        self.history_limit = history_limit
        self._history: List[str] = []
        self._lock = threading.Lock()

    def read(self) -> str:
        """The current draft without trailing whitespace; empty if there is none."""
        return self._read_raw().rstrip()

    def _read_raw(self) -> str:
        try:
            return self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return ''

    def apply(self, dictation: str, edit: Callable[[str, str], str]) -> str:
        """Run edit(draft, dictation) and store its result as the new draft.

        Changes are serialized: one edit at a time, in the order they arrive.
        Exceptions from edit propagate with the draft unchanged. If the file
        changed while edit ran (an $EDITOR save), the result is discarded and
        ScratchpadConflict is raised, so a manual edit is never overwritten.
        """
        with self._lock:
            before = self._read_raw()
            updated = edit(before.rstrip(), dictation).strip()
            if self._read_raw() != before:
                raise ScratchpadConflict('the draft changed while the edit was running')
            self._replace(before, updated)
            return updated

    def set(self, text: str) -> None:
        """Replace the draft (empty clears it); undo restores the previous one."""
        with self._lock:
            self._replace(self._read_raw(), (text or '').strip())

    def clear(self, expected: Optional[str] = None) -> bool:
        """Empty the draft. False if it was already empty.

        With expected, clear only if the draft still is that text (a paste
        must not clear a dictation that landed after the draft was read).
        """
        with self._lock:
            before = self._read_raw()
            if not before.strip() or (expected is not None and before.rstrip() != expected):
                return False
            self._replace(before, '')
            return True

    def undo(self) -> Optional[str]:
        """Restore the draft before the last change; None if there is nothing to undo."""
        with self._lock:
            if not self._history:
                return None
            previous = self._history.pop()
            self._write(previous)
            return previous

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def _replace(self, before: str, text: str) -> None:
        """Caller holds _lock."""
        if before.rstrip() == text:
            return
        self._write(text)
        self._history.append(before.rstrip())
        del self._history[:-self.history_limit]

    def _write(self, text: str) -> None:
        """Atomically store text (mode 0600); empty removes the file."""
        if not text:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix='.scratchpad-')  # created 0600
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text + '\n')
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
