"""
Scratchpad preview panel: a layer-shell overlay that shows the draft while
the scratchpad is in use, with the last change highlighted.

Runs as its own process, started by the daemon on first use
(ScratchpadPanelRunner). It follows the state file the daemon publishes
(paths.SCRATCHPAD_PREVIEW_FILE), never takes keyboard focus or pointer
input, so the paste target keeps both, and exits after a while hidden or
when the daemon is gone.
"""

import json
import os
import signal
import sys

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
gi.require_version('Gtk4LayerShell', '1.0')
import cairo
from gi.repository import Gdk, GLib, Gtk, Gtk4LayerShell, Pango

from .scratchpad_view import PanelState, highlight_tags, panel_css, render_markup
from .theme import ThemeWatcher, theme

try:
    from ..src.paths import SCRATCHPAD_PREVIEW_FILE
except ImportError:
    from src.paths import SCRATCHPAD_PREVIEW_FILE


POLL_MS = 150
# Quit after this long hidden; the daemon starts the panel again when needed.
IDLE_EXIT_SECONDS = 600
WIDTH = 560
MAX_DRAFT_HEIGHT = 320
# Sits above the mic OSD (anchored bottom with a 130px margin).
BOTTOM_MARGIN = 214
HIDE_FADE_MS = 170


class ScratchpadPanel(Gtk.Window):
    """The overlay: a status header, the draft, and a hint line."""

    def __init__(self):
        super().__init__()
        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_namespace(self, 'hyprwhspr-scratchpad')
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, True)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, BOTTOM_MARGIN)
        Gtk4LayerShell.set_exclusive_zone(self, -1)
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)

        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(WIDTH, -1)
        self.add_css_class('scratchpad-panel-window')
        self.connect('realize', self._on_realize)

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.box.add_css_class('scratchpad-panel')
        self.box.set_size_request(WIDTH, -1)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.dot = Gtk.Label(label='●')
        self.dot.add_css_class('scratchpad-dot')
        title = Gtk.Label(label='Scratchpad')
        title.add_css_class('scratchpad-title')
        self.meta = Gtk.Label()
        self.meta.add_css_class('scratchpad-meta')
        self.status = Gtk.Label(xalign=1.0, hexpand=True)
        self.status.add_css_class('scratchpad-status')
        self.status.set_ellipsize(Pango.EllipsizeMode.END)
        for widget in (self.dot, title, self.meta, self.status):
            header.append(widget)

        self.draft = Gtk.Label(xalign=0.0, yalign=0.0, wrap=True)
        self.draft.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        self.draft.set_max_width_chars(72)
        self.draft.add_css_class('scratchpad-draft')
        self.scroller = Gtk.ScrolledWindow()
        self.scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.scroller.set_propagate_natural_height(True)
        self.scroller.set_max_content_height(MAX_DRAFT_HEIGHT)
        self.scroller.set_child(self.draft)

        self.hint = Gtk.Label(xalign=0.0)
        self.hint.add_css_class('scratchpad-hint')
        self.hint.set_ellipsize(Pango.EllipsizeMode.END)

        self.box.append(header)
        self.box.append(self.scroller)
        self.box.append(self.hint)
        self.set_child(self.box)

        self._dot_state = None
        self._shown_draft = None
        self._hide_timer = None

    def _on_realize(self, _widget):
        # Clicks fall through to the window underneath
        try:
            self.get_surface().set_input_region(cairo.Region())
        except Exception:
            pass

    def render(self, state: PanelState):
        if self._dot_state:
            self.dot.remove_css_class(self._dot_state)
        self._dot_state = state.state
        self.dot.add_css_class(state.state)
        words = state.words
        self.meta.set_label(f"{words} word{'' if words == 1 else 's'}" if state.draft else '')
        self.status.set_label(state.status)
        self.hint.set_label(state.hint)
        self.hint.set_visible(bool(state.hint))

        if (state.draft, state.previous) == self._shown_draft:
            return
        self._shown_draft = (state.draft, state.previous)
        if state.draft:
            markup, first_change = render_markup(state.previous, state.draft, highlight_tags(theme))
            self.draft.remove_css_class('empty')
            self.draft.set_markup(markup)
        else:
            first_change = None
            self.draft.add_css_class('empty')
            self.draft.set_label('Empty. Dictate to start a draft.')
        # After the new text is laid out
        GLib.timeout_add(40, self._scroll_to, first_change)

    def _scroll_to(self, byte_index):
        """Bring the first change into view; with none, the end of the draft."""
        adjustment = self.scroller.get_vadjustment()
        if byte_index is None:
            adjustment.set_value(adjustment.get_upper() - adjustment.get_page_size())
            return False
        try:
            rect = self.draft.get_layout().index_to_pos(byte_index)
            _, offset_y = self.draft.get_layout_offsets()
            y = rect.y / Pango.SCALE + offset_y
            adjustment.set_value(max(0.0, y - 24.0))
        except Exception:
            pass
        return False

    def show_panel(self):
        if self._hide_timer:
            GLib.source_remove(self._hide_timer)
            self._hide_timer = None
        if not self.get_visible():
            self.set_visible(True)
        GLib.idle_add(self._fade_in)

    def _fade_in(self):
        self.box.add_css_class('shown')
        return False

    def hide_panel(self):
        if not self.get_visible() or self._hide_timer:
            return
        self.box.remove_css_class('shown')
        self._hide_timer = GLib.timeout_add(HIDE_FADE_MS, self._finish_hide)

    def _finish_hide(self):
        self._hide_timer = None
        self.set_visible(False)
        return False


class ScratchpadPanelApp:
    """Poll the state file, show or hide the panel, quit when idle."""

    def __init__(self, state_file=SCRATCHPAD_PREVIEW_FILE, parent_pid=None):
        self.state_file = state_file
        self.parent_pid = parent_pid
        self.loop = None
        self.window = None
        self.css = None
        self.theme_watcher = None
        self.state = PanelState()
        self._stamp = None
        self._hidden_since = GLib.get_monotonic_time()

    def run(self) -> int:
        Gtk.init()
        if not Gtk4LayerShell.is_supported():
            print('[SCRATCHPAD] Layer shell is not supported by this compositor', file=sys.stderr, flush=True)
            return 2
        self.css = Gtk.CssProvider()
        self._load_css()
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), self.css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1)
        self.window = ScratchpadPanel()
        self.window.set_visible(False)
        self.theme_watcher = ThemeWatcher(on_theme_changed=self._on_theme_changed)
        self.theme_watcher.start()
        GLib.timeout_add(POLL_MS, self._poll)
        self._poll()
        self.loop = GLib.MainLoop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, self._on_signal)
        try:
            self.loop.run()
        finally:
            if self.theme_watcher:
                self.theme_watcher.stop()
        return 0

    def _on_signal(self):
        self.quit()
        return False

    def quit(self):
        if self.loop:
            self.loop.quit()

    def _load_css(self):
        self.css.load_from_string(panel_css(theme))

    def _on_theme_changed(self):
        self._load_css()
        if self.window:
            self.window._shown_draft = None
            self.window.render(self.state)

    def _read_state(self):
        try:
            stat = os.stat(self.state_file)
        except FileNotFoundError:
            return PanelState(), None
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if stamp == self._stamp:
            return self.state, stamp
        try:
            with open(self.state_file, encoding='utf-8') as f:
                return PanelState.from_dict(json.load(f)), stamp
        except (OSError, ValueError):
            return self.state, self._stamp  # a torn read; try again next poll

    def _poll(self):
        if self.parent_pid and not _alive(self.parent_pid):
            self.quit()
            return False
        state, stamp = self._read_state()
        if stamp != self._stamp:
            self._stamp = stamp
            self.state = state
            if state.state != 'hidden':
                self.window.render(state)
        if self.state.visible():
            self._hidden_since = None
            self.window.show_panel()
        else:
            self.window.hide_panel()
            now = GLib.get_monotonic_time()
            if self._hidden_since is None:
                self._hidden_since = now
            elif now - self._hidden_since > IDLE_EXIT_SECONDS * 1_000_000:
                self.quit()
                return False
        return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main():
    parent = os.environ.get('HYPRWHSPR_PARENT_PID', '')
    return ScratchpadPanelApp(parent_pid=int(parent) if parent.isdigit() else None).run()


if __name__ == '__main__':
    sys.exit(main())
