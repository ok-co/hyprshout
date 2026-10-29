"""
View model for the scratchpad preview panel, kept free of GTK so it is
testable anywhere: the state the daemon publishes, the draft rendered as
Pango markup with the last change highlighted, and the panel's CSS.
"""

from __future__ import annotations

import difflib
import html
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# The daemon is busy with the draft: shown until the next state replaces it.
ACTIVE_STATES = ('listening', 'paused', 'transcribing', 'updating')
# A result: shown for `hold` seconds (or until replaced when hold is null).
RESULT_STATES = ('updated', 'error', 'pasted', 'copied', 'cleared')
STATES = ACTIVE_STATES + RESULT_STATES + ('hidden',)

STATUS_TEXT = {
    'listening': 'Listening…',
    'paused': 'Paused',
    'transcribing': 'Transcribing…',
    'updating': 'Updating draft…',
    'updated': 'Updated',
    'error': 'Edit failed',
    'pasted': 'Pasted',
    'copied': 'Copied',
    'cleared': 'Cleared',
}

# Status dot color per state, as theme.Theme property names.
DOT_COLOR = {
    'listening': 'recording_dot',
    'paused': 'paused_dot',
    'transcribing': 'processing_dot',
    'updating': 'processing_dot',
    'updated': 'success_dot',
    'pasted': 'success_dot',
    'copied': 'success_dot',
    'error': 'error_dot',
    'cleared': 'text',
}

# Words, runs of whitespace, and single punctuation marks: a change is
# highlighted word by word, never mid-word.
_TOKEN_RE = re.compile(r'\s+|\w+|[^\w\s]', re.UNICODE)


@dataclass(frozen=True)
class PanelState:
    state: str = 'hidden'
    draft: str = ''
    previous: Optional[str] = None
    message: str = ''
    hint: str = ''
    hold: Optional[float] = None
    at: float = 0.0

    @classmethod
    def from_dict(cls, data) -> 'PanelState':
        """Validate a published state; anything malformed reads as hidden."""
        if not isinstance(data, dict):
            return cls()
        state = data.get('state')
        if state not in STATES:
            return cls()

        def text(key):
            value = data.get(key)
            return value if isinstance(value, str) else ''

        previous = data.get('previous')
        hold = data.get('hold')
        at = data.get('at')
        return cls(
            state=state,
            draft=text('draft'),
            previous=previous if isinstance(previous, str) else None,
            message=text('message'),
            hint=text('hint'),
            hold=float(hold) if isinstance(hold, (int, float)) and not isinstance(hold, bool) and hold > 0 else None,
            at=float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) else 0.0,
        )

    def visible(self, now: Optional[float] = None) -> bool:
        if self.state == 'hidden':
            return False
        if self.hold is None:
            return True
        return (time.time() if now is None else now) < self.at + self.hold

    def hides_in(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds until a held state expires; None if it stays."""
        if self.state == 'hidden' or self.hold is None:
            return None
        return max(0.0, self.at + self.hold - (time.time() if now is None else now))

    @property
    def status(self) -> str:
        label = STATUS_TEXT.get(self.state, '')
        if self.state == 'error' and self.message:
            return f'{label}: {self.message}'
        return label

    @property
    def words(self) -> int:
        return len(self.draft.split())


def changed_spans(previous: Optional[str], current: str):
    """Character ranges of current that differ from previous.

    Returns (spans, first_change): spans are (start, end) offsets of inserted
    or replaced text, first_change the offset of the earliest change,
    deletions included, or None when nothing changed. No previous draft, or
    an empty one, means nothing to compare: the first dictation is not
    highlighted as a whole.
    """
    if not previous or previous == current:
        return [], None
    old = _TOKEN_RE.findall(previous)
    new = _TOKEN_RE.findall(current)
    offsets = [0]
    for token in new:
        offsets.append(offsets[-1] + len(token))
    spans = []
    first = None
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        if first is None:
            first = offsets[j1]
        if tag in ('insert', 'replace') and ''.join(new[j1:j2]).strip():
            start, end = offsets[j1], offsets[j2]
            # Don't paint the whitespace around a change
            while start < end and current[start].isspace():
                start += 1
            while end > start and current[end - 1].isspace():
                end -= 1
            if spans and start <= spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
    return spans, first


def render_markup(previous: Optional[str], current: str,
                  highlight: Tuple[str, str] = ('<span>', '</span>')) -> Tuple[str, Optional[int]]:
    """Pango markup for current with changed text wrapped in highlight.

    Returns (markup, first_change_byte): the UTF-8 byte offset of the first
    change in the plain text (what Pango's index_to_pos takes), or None.
    """
    spans, first = changed_spans(previous, current)
    parts = []
    cursor = 0
    for start, end in spans:
        parts.append(html.escape(current[cursor:start], quote=False))
        parts.append(highlight[0] + html.escape(current[start:end], quote=False) + highlight[1])
        cursor = end
    parts.append(html.escape(current[cursor:], quote=False))
    first_byte = None if first is None else len(current[:first].encode('utf-8'))
    return ''.join(parts), first_byte


def _rgba(color, alpha=None) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in color[:3])
    a = color[3] if alpha is None and len(color) > 3 else (1.0 if alpha is None else alpha)
    return f'rgba({r}, {g}, {b}, {a:.2f})'


def _hex(color) -> str:
    return '#' + ''.join(f'{max(0, min(255, round(c * 255))):02x}' for c in color[:3])


def highlight_tags(theme) -> Tuple[str, str]:
    """Pango span that marks changed text, in the theme's accent color."""
    return (f'<span background="{_hex(theme.bar_left)}" bgalpha="28%" underline="single" '
            f'underline_color="{_hex(theme.bar_left)}">', '</span>')


def panel_css(theme) -> str:
    """CSS for the panel from the mic-osd theme (Omarchy/Noctalia aware)."""
    background = theme.background
    text = theme.text
    return f"""
window.scratchpad-panel-window,
window.scratchpad-panel-window.background {{
    background: transparent;
    box-shadow: none;
}}
.scratchpad-panel {{
    background-color: {_rgba(background)};
    border: 1px solid {_rgba(theme.border, 0.55)};
    border-radius: 14px;
    padding: 12px 16px 10px 16px;
    color: {_rgba(text)};
    opacity: 0;
    transition: opacity 160ms ease-out;
}}
.scratchpad-panel.shown {{
    opacity: 1;
}}
.scratchpad-title {{
    font-weight: bold;
}}
.scratchpad-meta, .scratchpad-hint {{
    color: {_rgba(text, 0.6)};
    font-size: 0.85em;
}}
.scratchpad-status {{
    color: {_rgba(text, 0.85)};
    font-size: 0.9em;
}}
.scratchpad-draft {{
    font-size: 1.05em;
}}
.scratchpad-draft.empty {{
    color: {_rgba(text, 0.5)};
    font-style: italic;
}}
""" + ''.join(
        f'.scratchpad-dot.{state} {{ color: {_rgba(getattr(theme, prop))}; }}\n'
        for state, prop in DOT_COLOR.items()
    )
