"""
Daemon side of the scratchpad preview panel (scratchpad_panel.py).

publish() writes the panel's state file and starts the panel process when
it isn't running. The panel quits by itself after a while hidden, so it
costs nothing between drafting sessions. GTK never loads in the daemon.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from ..src.paths import SCRATCHPAD_PANEL_LOG_FILE, SCRATCHPAD_PREVIEW_FILE
except ImportError:
    from src.paths import SCRATCHPAD_PANEL_LOG_FILE, SCRATCHPAD_PREVIEW_FILE

from .runner import LayerShellResolutionError, MicOSDRunner
from .scratchpad_view import ACTIVE_STATES, STATES

# previous=KEEP reuses the last highlight while the draft is unchanged.
KEEP = object()


class ScratchpadPanelRunner:
    """Publish panel states and keep the panel process alive while needed."""

    # A panel that crashes this many times in a row is given up on; the
    # scratchpad falls back to notifications.
    MAX_FAILURES = 3

    def __init__(self, state_file=None, log_file=None, popen=subprocess.Popen):
        self.state_file = Path(state_file) if state_file is not None else SCRATCHPAD_PREVIEW_FILE
        self.log_file = Path(log_file) if log_file is not None else SCRATCHPAD_PANEL_LOG_FILE
        self._popen = popen
        self._process = None
        self._lock = threading.Lock()
        self._failures = 0
        self.disabled = False
        self.last = None

    @staticmethod
    def is_supported() -> bool:
        """GTK4 + layer shell installed and the compositor speaks layer shell."""
        return MicOSDRunner.is_available() and MicOSDRunner.layer_shell_active()

    @property
    def state(self):
        return self.last['state'] if self.last else 'hidden'

    def publish(self, state, draft, previous=KEEP, message='', hint='', hold=None) -> bool:
        """Show state with draft. False when the panel can't be shown."""
        if state not in STATES:
            raise ValueError(f'unknown panel state {state!r}')
        with self._lock:
            if self.disabled:
                return False
            if previous is KEEP:
                last = self.last
                previous = last['previous'] if last and last['draft'] == draft else None
            payload = {
                'state': state, 'draft': draft, 'previous': previous, 'message': message,
                'hint': hint, 'hold': hold, 'at': time.time(),
            }
            try:
                MicOSDRunner._atomic_write_runtime_file(self.state_file, json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                print(f"[SCRATCHPAD] Failed to write panel state: {e}", flush=True)
                return False
            self.last = payload
            if state == 'hidden':
                return True
            return self._ensure_process()

    def is_active(self) -> bool:
        """The panel shows a state that waits for the daemon's next one."""
        return self.state in ACTIVE_STATES

    def _ensure_process(self) -> bool:
        """Caller holds _lock."""
        process = self._process
        if process is not None:
            code = process.poll()
            if code is None:
                return True
            self._process = None
            if code != 0:  # 0 is the panel's idle exit
                self._failures += 1
                print(f"[SCRATCHPAD] Preview panel exited with code {code}{self._log_tail()}", flush=True)
                if self._failures >= self.MAX_FAILURES:
                    print("[SCRATCHPAD] Preview panel keeps failing; using notifications instead", flush=True)
                    self.disabled = True
                    return False
        lib_dir = Path(__file__).resolve().parent.parent
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(lib_dir)!r})\n"
            "from mic_osd.scratchpad_panel import main\n"
            "sys.exit(main())\n"
        )
        try:
            env = MicOSDRunner._layer_shell_environment()
        except LayerShellResolutionError as e:
            print(f"[SCRATCHPAD] Preview panel unavailable: {e}", flush=True)
            self.disabled = True
            return False
        env['HYPRWHSPR_PARENT_PID'] = str(os.getpid())
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(self.log_file, 'w', encoding='utf-8') as log:
                self._process = self._popen(
                    [sys.executable or 'python3', '-c', code],
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True,
                )
        except Exception as e:
            print(f"[SCRATCHPAD] Failed to start preview panel: {e}", flush=True)
            self._failures += 1
            if self._failures >= self.MAX_FAILURES:
                self.disabled = True
            return False
        return True

    def _log_tail(self) -> str:
        try:
            tail = self.log_file.read_text(encoding='utf-8', errors='replace').strip()[-400:]
        except OSError:
            return ''
        return f": {tail}" if tail else ''

    def stop(self):
        """Quit the panel and remove its state (it may hold draft text)."""
        with self._lock:
            process, self._process = self._process, None
            self.last = None
            try:
                self.state_file.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
        except OSError:
            pass
