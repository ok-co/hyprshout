"""hyprwhspr settings: a full-screen editor for every setting.

Sections on the left, their settings on the right, help for the selected one
below. Space flips an on/off setting or steps through choices, Enter edits
anything (long prompts and structured settings open in your text editor),
Backspace puts a setting back to its default, / searches every setting and
its description, s saves, checking the whole config against the schema
first, and offers to restart the service so the change applies.

Stdlib curses only: the CLI runs under the system Python. The drawing and
the dialogs are methods tests can replace; the model is settings_model.
"""

import curses
import json
import locale
import os
import shutil
import subprocess
import tempfile
import textwrap

try:
    from ..settings_model import Credential, Section, SettingsDocument, UNSET, builtin_text, describe_kind, \
        format_choice, format_value
except ImportError:
    from settings_model import Credential, Section, SettingsDocument, UNSET, builtin_text, describe_kind, \
        format_choice, format_value

from ._shared import SERVICE_NAME

HINTS = '↑↓ move  ←→ sections  Enter edit  Space toggle  ⌫ default  / search  s save  ? help  q quit'
HELP = (
    ('↑ ↓  PgUp PgDn  Home End', 'move'),
    ('← →  Tab', 'switch between sections and settings'),
    ('Enter', 'edit the setting (or open the section)'),
    ('Space', 'turn on/off, or step to the next choice (Shift+Tab: previous)'),
    ('Backspace  Delete', 'back to the default (API keys: remove the key)'),
    ('e', 'edit any setting in your text editor ($VISUAL or $EDITOR)'),
    ('/', 'search every setting and its description; Esc clears it'),
    ('c', 'show only the settings you changed'),
    ('s', 'save: checks every setting first, then offers a service restart'),
    ('q  Esc', 'quit (asks about unsaved changes)'),
)
MIN_HEIGHT, MIN_WIDTH = 14, 60
# Escape sequences some terminals send without the keypad mode curses asks for
_SEQUENCES = {'[A': curses.KEY_UP, '[B': curses.KEY_DOWN, '[C': curses.KEY_RIGHT, '[D': curses.KEY_LEFT,
              'OA': curses.KEY_UP, 'OB': curses.KEY_DOWN, 'OC': curses.KEY_RIGHT, 'OD': curses.KEY_LEFT,
              '[H': curses.KEY_HOME, '[F': curses.KEY_END, 'OH': curses.KEY_HOME, 'OF': curses.KEY_END,
              '[1~': curses.KEY_HOME, '[4~': curses.KEY_END, '[3~': curses.KEY_DC, '[5~': curses.KEY_PPAGE,
              '[6~': curses.KEY_NPAGE, '[Z': curses.KEY_BTAB}


def read_key(window):
    """The next key from a window: a character, or a curses key code for keys
    whose escape sequence curses didn't decode itself. A lone Esc is '\x1b'."""
    key = window.get_wch()
    if key != '\x1b':
        return key
    window.nodelay(True)
    sequence = ''
    try:
        while len(sequence) < 3:
            try:
                char = window.get_wch()
            except curses.error:
                break
            if not isinstance(char, str):
                break
            sequence += char
            if sequence in _SEQUENCES:
                return _SEQUENCES[sequence]
            if sequence[0] not in '[O' or (len(sequence) > 1 and not (sequence[-1].isdigit() or sequence[-1] == '~')):
                break
    finally:
        window.nodelay(False)
    return '\x1b' if not sequence else _SEQUENCES.get(sequence, '\x1b')


def _service_active() -> bool:
    try:
        return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', SERVICE_NAME],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _restart_service() -> bool:
    try:
        return subprocess.run(['systemctl', '--user', 'restart', SERVICE_NAME], capture_output=True,
                              timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_editor():
    """The user's editor command, or None."""
    for variable in ('VISUAL', 'EDITOR'):
        command = os.environ.get(variable, '').strip()
        if command:
            return command
    for name in ('nano', 'vim', 'vi'):
        if shutil.which(name):
            return name
    return None


class SettingsTUI:
    """The settings editor: state and key handling, plus its curses drawing and dialogs."""

    def __init__(self, doc: SettingsDocument, service_active=_service_active, restart=_restart_service):
        self.doc = doc
        self.service_active = service_active
        self.restart = restart
        self.section_index = 0
        self.row = 0
        self.focus = 'sections'
        self.query = None
        self.only_changed = False
        self.status = ''
        self.status_kind = 'info'
        self.done = False
        self.screen = None
        self._attrs = {}
        self._scroll = 0
        self._section_scroll = 0

    # ------------------------ What is shown ------------------------

    def sections(self):
        return self.doc.sections()

    def section(self) -> Section:
        if self.query is not None:
            keys = self.doc.search(self.query)
            return Section('search', f'Search: {self.query}',
                           f'{len(keys)} setting{"" if len(keys) == 1 else "s"} match.' if keys
                           else 'Nothing matches. Esc clears the search.', keys)
        if self.only_changed:
            return self.doc.customized_section()
        sections = self.sections()
        self.section_index = max(0, min(self.section_index, len(sections) - 1))
        return sections[self.section_index]

    def keys(self):
        return self.section().keys

    def current_key(self):
        keys = self.keys()
        if not keys:
            return None
        self.row = max(0, min(self.row, len(keys) - 1))
        return keys[self.row]

    def item(self, key):
        if key is None:
            return None
        if key.startswith('credential:'):
            return self.doc.credential(key)
        return self.doc.fields.get(key)

    def value_text(self, key, width=40):
        item = self.item(key)
        if isinstance(item, Credential):
            return self.doc.credential_status(item)
        return format_value(item, self.doc.get(key), width)

    def say(self, message, kind='info'):
        self.status, self.status_kind = message, kind

    # ------------------------ Keys ------------------------

    def handle_key(self, key):
        """One key press; strings are characters, ints curses key codes."""
        if key in ('q', 'Q'):
            self.quit()
        elif key == '\x1b':
            if self.query is not None or self.only_changed:
                self.query, self.only_changed, self.row = None, False, 0
            else:
                self.quit()
        elif key == '?':
            self.show_help()
        elif key in ('s', 'S', '\x13'):
            self.save()
        elif key == '/':
            query = self.ask_line('Search: ', self.query or '')
            if query is not None:
                self.query = query.strip() or None
                self.only_changed = False
                self.row, self.focus = 0, 'settings'
        elif key == 'c':
            self.only_changed, self.query = not self.only_changed, None
            self.row, self.focus = 0, 'settings'
        elif key in (curses.KEY_UP, 'k'):
            self.move(-1)
        elif key in (curses.KEY_DOWN, 'j'):
            self.move(1)
        elif key == curses.KEY_PPAGE:
            self.move(-10)
        elif key == curses.KEY_NPAGE:
            self.move(10)
        elif key == curses.KEY_HOME:
            self.move(-10 ** 6)
        elif key == curses.KEY_END:
            self.move(10 ** 6)
        elif key in (curses.KEY_LEFT, 'h'):
            self.focus = 'sections'
        elif key in (curses.KEY_RIGHT, 'l', '\t'):
            if self.focus == 'sections' or key != '\t':
                self.focus = 'settings'
            else:
                self.focus = 'sections'
        elif key in ('\n', '\r', curses.KEY_ENTER):
            if self.focus == 'sections':
                self.focus = 'settings'
            else:
                self.edit()
        elif key == ' ':
            if self.focus == 'settings':
                self.toggle(1)
        elif key == curses.KEY_BTAB:
            if self.focus == 'settings':
                self.toggle(-1)
        elif key in (curses.KEY_BACKSPACE, curses.KEY_DC, '\x7f', '\b'):
            if self.focus == 'settings':
                self.reset()
        elif key == 'e':
            if self.focus == 'settings':
                self.edit(external=True)

    def move(self, step):
        if self.focus == 'sections' and self.query is None and not self.only_changed:
            count = len(self.sections())
            self.section_index = max(0, min(count - 1, self.section_index + step))
            self.row = 0
            self._scroll = 0
        else:
            count = len(self.keys())
            self.row = max(0, min(count - 1, self.row + step)) if count else 0

    # ------------------------ Editing ------------------------

    def toggle(self, step):
        key = self.current_key()
        item = self.item(key)
        if item is None:
            return
        if item.kind in ('bool', 'choice'):
            self.doc.cycle(key, step)
            self.say(f'{item.label}: {self.value_text(key)}')
        else:
            self.edit()

    def edit(self, external=False):
        key = self.current_key()
        item = self.item(key)
        if item is None:
            return
        if isinstance(item, Credential):
            self.edit_credential(item)
        elif external or item.kind in ('prompt', 'json'):
            self.edit_long(key, item)
        elif item.kind == 'bool':
            self.toggle(1)
        elif item.kind == 'choice':
            current = self.doc.get(key)
            index = self.pick(item.label, [format_choice(c) for c in item.choices],
                              item.choices.index(current) if current in item.choices else 0)
            if index is not None:
                self.doc.set(key, item.choices[index])
                self.say(f'{item.label}: {self.value_text(key)}')
        else:
            value = self.doc.get(key)
            if value is None or value is UNSET:
                initial = ''
            elif isinstance(value, str):
                initial = value
            else:
                initial = json.dumps(value) if isinstance(value, (bool, list, dict)) else str(value)
            text = self.ask_line(f'{item.label}: ', initial)
            if text is None:
                return
            error = self.doc.apply_text(key, text)
            if error:
                self.say(f'{item.label}: {error} (not changed)', 'error')
            else:
                self.say(f'{item.label}: {self.value_text(key)}')

    def edit_long(self, key, item):
        value = self.doc.get(key)
        if item.kind == 'prompt' or (item.kind == 'text' and isinstance(value, (str, type(None)))):
            started_empty = value in (None, '', UNSET)
            initial = (builtin_text(key) or '') if started_empty else str(value)
            text = self.edit_external(initial, '.md')
            if text is None:
                return
            text = text.rstrip('\n')
            if started_empty and text == initial.rstrip('\n'):
                self.say(f'{item.label}: unchanged')
                return
            error = self.doc.apply_text(key, text if text.strip() else '')
        else:
            initial = json.dumps({} if value is UNSET else value, indent=2, ensure_ascii=False)
            while True:
                text = self.edit_external(initial, '.json')
                if text is None:
                    return
                error = self.doc.apply_text(key, text) if item.kind == 'json' else self._apply_json_text(key, text)
                if not error or self.confirm(f'{item.label}: {error}. Edit again?', 'yn') != 'y':
                    break
                initial = text
        if error:
            self.say(f'{item.label}: {error} (not changed)', 'error')
        else:
            self.say(f'{item.label}: {self.value_text(key)}')

    def _apply_json_text(self, key, text):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as e:
            return f'not valid JSON: {e.msg}'
        error = self.doc.check(key, value)
        if not error:
            self.doc.set(key, value)
        return error

    def edit_credential(self, cred):
        secret = self.ask_line(f'{cred.label} (hidden, empty keeps it): ', '', secret=True)
        if not secret or not secret.strip():
            return
        if self.doc.set_credential(cred, secret):
            self.say(f'{cred.label} saved in the credential store', 'ok')
        else:
            self.say(f'{cred.label}: could not be saved', 'error')

    def reset(self):
        key = self.current_key()
        item = self.item(key)
        if item is None:
            return
        if isinstance(item, Credential):
            if self.doc.credential_status(item) == 'not set':
                return
            if self.confirm(f'Remove the {item.label}?', 'yn') == 'y':
                ok = self.doc.delete_credential(item)
                self.say(f'{item.label} removed' if ok else f'{item.label}: could not be removed',
                         'ok' if ok else 'error')
            return
        self.doc.reset(key)
        self.say(f'{item.label}: back to the default ({self.value_text(key)})')

    # ------------------------ Saving and quitting ------------------------

    def save(self) -> bool:
        errors = self.doc.errors()
        if errors:
            lines = [f'{self._label(key)} ({key}): {message}' for key, message in errors[:12]]
            if len(errors) > 12:
                lines.append(f'… and {len(errors) - 12} more')
            self.show_message('Not saved: fix these settings first', lines)
            self.say(f'Not saved: {len(errors)} invalid setting{"s" if len(errors) > 1 else ""}', 'error')
            return False
        if not self.doc.dirty:
            self.say('Nothing to save')
            return True
        changed = len(self.doc.changed_keys())
        if not self.doc.save():
            self.say('Could not write the config file', 'error')
            return False
        if self.service_active() and self.confirm(
                f'Saved {changed} change{"s" if changed > 1 else ""}. Restart hyprwhspr now to apply them?',
                'yn') == 'y':
            if self.restart():
                self.say('Saved; hyprwhspr restarted with the new settings', 'ok')
            else:
                self.say('Saved, but the restart failed: try systemctl --user restart hyprwhspr', 'error')
        else:
            self.say('Saved. Restart hyprwhspr to apply: systemctl --user restart hyprwhspr', 'ok')
        return True

    def quit(self):
        if self.doc.dirty:
            count = len(self.doc.changed_keys())
            answer = self.confirm(f'{count} unsaved change{"" if count == 1 else "s"}. Save before quitting? '
                                  'y saves, n discards them, c keeps editing.', 'ync')
            if answer == 'y':
                if not self.save():
                    return
            elif answer != 'n':
                return
        self.done = True

    def _label(self, key):
        item = self.item(key)
        return item.label if item is not None else key

    # ------------------------ Dialogs (curses; tests replace these) ------------------------

    def ask_line(self, prompt, initial='', secret=False):
        """A line typed at the bottom of the screen; None on Esc."""
        screen = self.screen
        height, width = screen.getmaxyx()
        buffer, cursor = list(initial), len(initial)
        curses.curs_set(1)
        try:
            while True:
                shown = ('•' * len(buffer)) if secret else ''.join(buffer)
                room = max(1, width - len(prompt) - 2)
                start = max(0, cursor - room + 1)
                screen.move(height - 1, 0)
                screen.clrtoeol()
                self._put(height - 1, 0, prompt, self.attr('section'))
                self._put(height - 1, len(prompt), shown[start:start + room])
                screen.move(height - 1, min(width - 1, len(prompt) + cursor - start))
                key = read_key(screen)
                if key in ('\n', '\r', curses.KEY_ENTER):
                    return ''.join(buffer)
                if key == '\x1b':
                    return None
                if key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                    if cursor:
                        del buffer[cursor - 1]
                        cursor -= 1
                elif key == curses.KEY_DC:
                    if cursor < len(buffer):
                        del buffer[cursor]
                elif key == curses.KEY_LEFT:
                    cursor = max(0, cursor - 1)
                elif key == curses.KEY_RIGHT:
                    cursor = min(len(buffer), cursor + 1)
                elif key in (curses.KEY_HOME, '\x01'):
                    cursor = 0
                elif key in (curses.KEY_END, '\x05'):
                    cursor = len(buffer)
                elif key == '\x15':  # Ctrl+U
                    buffer, cursor = [], 0
                elif key == curses.KEY_RESIZE:
                    height, width = screen.getmaxyx()
                elif isinstance(key, str) and key.isprintable():
                    buffer.insert(cursor, key)
                    cursor += 1
        finally:
            curses.curs_set(0)

    def pick(self, title, options, current=0):
        """An option chosen from a list; its index, or None on Esc."""
        index = current
        while True:
            lines = [('› ' if i == index else '  ') + option for i, option in enumerate(options)]
            window = self._popup(title, lines, highlight=index, footer='Enter choose · Esc cancel')
            key = read_key(window)
            if key in (curses.KEY_UP, 'k'):
                index = (index - 1) % len(options)
            elif key in (curses.KEY_DOWN, 'j', ' '):
                index = (index + 1) % len(options)
            elif key in ('\n', '\r', curses.KEY_ENTER):
                return index
            elif key in ('\x1b', 'q'):
                return None

    def confirm(self, question, choices):
        """The key pressed among choices (lower case), or 'esc'."""
        window = self._popup('hyprwhspr settings', textwrap.wrap(question, 56) or [question],
                             footer=' / '.join(choices))
        while True:
            key = read_key(window)
            if key == '\x1b':
                return 'esc'
            if isinstance(key, str) and key.lower() in choices:
                return key.lower()

    def show_message(self, title, lines, wrap=True):
        wrapped = [part for line in lines for part in (textwrap.wrap(line, 70) or [''])] if wrap else list(lines)
        window = self._popup(title, wrapped, footer='any key closes')
        read_key(window)

    def show_help(self):
        width = max(len(keys) for keys, _ in HELP) + 2
        lines = []
        for keys, what in HELP:
            parts = textwrap.wrap(what, 70 - width) or ['']
            lines.append(f'{keys:<{width}}{parts[0]}')
            lines.extend(' ' * width + part for part in parts[1:])
        lines += [''] + textwrap.wrap('● marks a setting changed from its default, * one not saved yet. Changes '
                                      'apply once you save and the service restarts.', 70)
        self.show_message('Keys', lines, wrap=False)

    def edit_external(self, text, suffix):
        """text after the user's editor; None if there is no editor or it failed."""
        editor = find_editor()
        if editor is None:
            self.show_message('No text editor', ['Set $EDITOR (for example: export EDITOR=nano) to edit '
                                                 'long text and structured settings.'])
            return None
        fd, path = tempfile.mkstemp(prefix='hyprwhspr-setting-', suffix=suffix)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text if text.endswith('\n') or not text else text + '\n')
            curses.def_prog_mode()
            curses.endwin()
            try:
                result = subprocess.run(editor.split() + [path])
            finally:
                curses.reset_prog_mode()
                self.screen.refresh()
            if result.returncode != 0:
                self.say(f'The editor exited with {result.returncode}; nothing changed', 'error')
                return None
            with open(path, encoding='utf-8') as f:
                return f.read()
        except OSError as e:
            self.say(f'Cannot run the editor: {e}', 'error')
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ------------------------ Drawing ------------------------

    def attr(self, name):
        return self._attrs.get(name, 0)

    def setup_colors(self):
        self._attrs = {'title': curses.A_REVERSE | curses.A_BOLD, 'selected': curses.A_REVERSE,
                       'dim': curses.A_DIM, 'section': curses.A_BOLD, 'custom': curses.A_BOLD,
                       'error': curses.A_BOLD, 'ok': curses.A_BOLD}
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        pairs = {'section': curses.COLOR_CYAN, 'custom': curses.COLOR_YELLOW, 'error': curses.COLOR_RED,
                 'ok': curses.COLOR_GREEN}
        for number, (name, color) in enumerate(pairs.items(), start=1):
            curses.init_pair(number, color, background)
            self._attrs[name] = curses.color_pair(number) | curses.A_BOLD

    def _put(self, y, x, text, attr=0):
        height, width = self.screen.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        try:
            self.screen.addnstr(y, x, text, max(0, width - x - (1 if y == height - 1 else 0)), attr)
        except curses.error:
            pass

    def _popup(self, title, lines, highlight=None, footer=''):
        height, width = self.screen.getmaxyx()
        inner = min(width - 6, max([len(title) + 4, len(footer) + 4] + [len(line) for line in lines]) + 2)
        rows = min(height - 4, len(lines) + 4)
        top, left = max(1, (height - rows) // 2), max(2, (width - inner - 2) // 2)
        window = curses.newwin(rows, inner + 2, top, left)
        window.keypad(True)
        window.erase()
        window.box()
        try:
            window.addnstr(0, 2, f' {title} ', inner - 2, self.attr('section'))
            visible = rows - 4
            first = 0 if highlight is None else max(0, min(highlight - visible + 1, len(lines) - visible))
            first = max(0, first)
            for i, line in enumerate(lines[first:first + visible]):
                window.addnstr(2 + i, 2, line, inner - 2,
                               self.attr('selected') if highlight == first + i else 0)
            if footer:
                window.addnstr(rows - 1, max(2, inner - len(footer) - 1), f' {footer} ', inner - 2, self.attr('dim'))
        except curses.error:
            pass
        window.refresh()
        return window

    def draw(self):
        screen = self.screen
        screen.erase()
        height, width = screen.getmaxyx()
        if height < MIN_HEIGHT or width < MIN_WIDTH:
            self._put(0, 0, f'Make the terminal at least {MIN_WIDTH}x{MIN_HEIGHT} (q quits)')
            screen.refresh()
            return
        changed = len(self.doc.changed_keys())
        title = ' hyprwhspr settings'
        state = f'{changed} unsaved change{"" if changed == 1 else "s"} · s saves ' if changed else 'all saved '
        self._put(0, 0, title.ljust(width - len(state)) + state, self.attr('title'))

        detail = 7
        body_top, body_bottom = 1, height - detail - 1
        longest = max(len(section.title) for section in self.sections())
        left = max(20, min(longest + 5, width // 3))
        self._draw_sections(body_top, body_bottom, left)
        for y in range(body_top, body_bottom):
            self._put(y, left, '│', self.attr('dim'))
        self._draw_settings(body_top, body_bottom, left + 2, width - left - 3)
        self._draw_detail(body_bottom, width, detail)
        if self.status:
            self._put(height - 1, 0, ' ' + self.status, self.attr(self.status_kind) if self.status_kind != 'info' else 0)
        else:
            self._put(height - 1, 0, ' ' + HINTS, self.attr('dim'))
        screen.refresh()

    def _draw_sections(self, top, bottom, width):
        sections = self.sections()
        rows = bottom - top
        index = self.section_index
        browsing = self.query is None and not self.only_changed
        if index < self._section_scroll:
            self._section_scroll = index
        elif index >= self._section_scroll + rows:
            self._section_scroll = index - rows + 1
        for i, section in enumerate(sections[self._section_scroll:self._section_scroll + rows]):
            number = self._section_scroll + i
            customized = sum(1 for k in section.keys if not k.startswith('credential:') and self.doc.is_customized(k))
            mark = f'{customized}' if customized and section.id != 'credentials' else ''
            text = f' {section.title}'[:width - len(mark) - 2].ljust(width - len(mark) - 1) + mark
            if number == index and browsing:
                attr = self.attr('selected') if self.focus == 'sections' else self.attr('section')
            else:
                attr = self.attr('dim') if not browsing else 0
            self._put(top + i, 0, text, attr)

    def _draw_settings(self, top, bottom, x, width):
        section = self.section()
        self._put(top, x, section.title, self.attr('section'))
        self._put(top + 1, x, section.blurb[:width], self.attr('dim'))
        rows = bottom - top - 3
        keys = section.keys
        self.current_key()
        if self.row < self._scroll:
            self._scroll = self.row
        elif self.row >= self._scroll + rows:
            self._scroll = self.row - rows + 1
        label_width = min(36, max(18, width // 2))
        unsaved_keys = set(self.doc.changed_keys())
        for i, key in enumerate(keys[self._scroll:self._scroll + rows]):
            number = self._scroll + i
            item = self.item(key)
            if item is None:
                continue
            customized = not isinstance(item, Credential) and self.doc.is_customized(key)
            unsaved = key in unsaved_keys
            mark = '●' if customized else ' '
            value = self.value_text(key, max(8, width - label_width - 4))
            line = f'{mark} {item.label[:label_width - 1]:<{label_width}}{value}{" *" if unsaved else ""}'
            attr = 0
            if number == self.row and self.focus == 'settings':
                attr = self.attr('selected')
            elif customized:
                attr = self.attr('custom')
            self._put(top + 3 + i, x, line[:width].ljust(width) if attr == self.attr('selected') else line[:width], attr)
        if not keys:
            self._put(top + 3, x, 'Nothing here.', self.attr('dim'))

    def _draw_detail(self, top, width, height):
        self._put(top, 0, '─' * width, self.attr('dim'))
        key = self.current_key()
        item = self.item(key)
        if item is None:
            return
        if isinstance(item, Credential):
            head, hint, default = item.label, describe_kind(item), ''
        else:
            head = f'{item.label}  ({key})'
            hint = describe_kind(item)
            default_value = self.doc.default(key)
            default = 'Default: ' + format_value(item, default_value, 40)
        self._put(top + 1, 1, head, self.attr('section'))
        self._put(top + 2, 1, hint + (f'   ·   {default}' if default else ''), self.attr('dim'))
        error = None if isinstance(item, Credential) else self.doc.check(key, self.doc.get(key))
        lines = textwrap.wrap(item.description, max(20, width - 2))
        if error:
            lines = [f'Invalid: {error}'] + lines
        for i, line in enumerate(lines[:height - 3]):
            self._put(top + 3 + i, 1, line, self.attr('error') if error and i == 0 else 0)

    # ------------------------ Main loop ------------------------

    def run(self, screen):
        self.screen = screen
        curses.curs_set(0)
        screen.keypad(True)
        self.setup_colors()
        while not self.done:
            self.draw()
            key = read_key(screen)
            if key == curses.KEY_RESIZE:
                continue
            self.status = ''
            self.handle_key(key)


def settings_command() -> bool:
    """Open the settings editor; False when there is no terminal to draw in."""
    import sys
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("hyprwhspr settings needs an interactive terminal. "
              "Use 'hyprwhspr config edit' or 'hyprwhspr config show' instead.", file=sys.stderr)
        return False
    locale.setlocale(locale.LC_ALL, '')
    os.environ.setdefault('ESCDELAY', '25')
    tui = SettingsTUI(SettingsDocument())
    curses.wrapper(tui.run)
    if tui.doc.dirty:
        print('Unsaved changes were discarded.')
    return True
