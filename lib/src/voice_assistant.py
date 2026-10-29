"""
Voice assistant: a spoken question answered out loud.

The question goes to the llm_refine_* chat provider (OpenRouter by default)
with a system prompt written for speech, and the caller reads the reply
aloud with speech_output. Recent exchanges are kept for a few minutes, so a
follow-up ("and in Fahrenheit?") has its context.

Stdlib only, like llm_refiner, so core stays importable.
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from . import llm_refiner
    from .speech_output import normalize_voice
except ImportError:
    import llm_refiner
    from speech_output import normalize_voice


DEFAULT_PROMPT = """\
You are Jarvis, a voice assistant on the user's Linux desktop. The user talks to you: speech recognition turns their words into text, and a text-to-speech voice reads your reply aloud.

Personality: helpful first, with a dry, understated wit, like a well-read butler who finds the user mildly amusing. A light joke or wry aside is welcome when it fits, at most one per reply, and never at the cost of the answer. Stay warm; tease the situation, never the user. When the user is in a hurry, upset, or the matter is serious, drop the jokes.

How to answer:
- Lead with the answer. Keep it to one to three short sentences unless the user asks for more.
- Write for the ear: plain spoken sentences only. No Markdown, lists, headings, tables, code, emoji or links. Say numbers, units and symbols the way a person would say them.
- The text comes from speech recognition and may contain misheard words; go with the most plausible meaning. If it is truly unclear, ask one short question back.
- You can't see the screen, browse the web, look up live data such as weather, news or prices, or do anything on the computer. If asked to, say so briefly (a little self-deprecation is allowed) and help with what you know.
- If you don't know, say so. Never invent facts.
- Reply in the language the user spoke."""

DEFAULT_VOICE = 'george'
DEFAULT_TIMEOUT = 30
DEFAULT_MEMORY_MINUTES = 5
# Exchanges sent along with a question; older ones are dropped.
MAX_EXCHANGES = 8


class AssistantError(Exception):
    """The assistant has no answer to read."""


def _clean_string(value) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(get_setting: Callable, key: str, default: float, low: float, high: float) -> float:
    value = get_setting(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high:
        return float(value)
    return float(default)


def build_system_prompt(get_setting: Callable, avoid_phrases: Iterable[str] = (),
                        now: Optional[datetime] = None) -> str:
    """The configured prompt, the current time, and phrases the reply must not say.

    assistant_prompt_file wins over assistant_prompt, which wins over the
    built-in prompt. The file is read on every question, so edits apply
    without a service restart. avoid_phrases are the wake words: a reply that
    said one aloud would trigger voice control through the speakers.
    """
    prompt_file = _clean_string(get_setting('assistant_prompt_file', None))
    if prompt_file:
        try:
            prompt = Path(prompt_file).expanduser().read_text(encoding='utf-8').strip()
        except (OSError, UnicodeError) as e:
            raise AssistantError(f'could not read assistant_prompt_file: {type(e).__name__}') from e
        if not prompt:
            raise AssistantError('assistant_prompt_file is empty')
    else:
        prompt = _clean_string(get_setting('assistant_prompt', None)) or DEFAULT_PROMPT
    now = now or datetime.now().astimezone()
    prompt += f"\n\nIt is now {now.strftime('%A, %d %B %Y, %H:%M')}."
    phrases = [p for p in dict.fromkeys(avoid_phrases) if p]
    if phrases:
        quoted = ', '.join(f'"{p}"' for p in phrases)
        prompt += f'\nNever say these phrases, because they control the app you run in: {quoted}.'
    return prompt


def configured_voice(get_setting: Callable) -> str:
    return normalize_voice(get_setting('assistant_voice', DEFAULT_VOICE)) or DEFAULT_VOICE


class Conversation:
    """The assistant's short memory: recent exchanges, forgotten after a quiet spell."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._exchanges = []
        self._last = None

    def forget(self) -> None:
        with self._lock:
            self._exchanges = []
            self._last = None

    def history(self, get_setting: Callable) -> list:
        """(role, text) turns still remembered, oldest first."""
        minutes = _number(get_setting, 'assistant_memory_minutes', DEFAULT_MEMORY_MINUTES, 0, 1440)
        with self._lock:
            if self._last is None or minutes <= 0 or self._clock() - self._last > minutes * 60:
                self._exchanges = []
            turns = []
            for question, reply in self._exchanges[-MAX_EXCHANGES:]:
                turns += [('user', question), ('assistant', reply)]
            return turns

    def ask(self, question: str, get_setting: Callable, avoid_phrases: Iterable[str] = (),
            credential_lookup: Optional[Callable] = None, opener: Optional[Callable] = None) -> str:
        """The assistant's reply to question, remembered for follow-ups; or raise AssistantError."""
        question = (question or '').strip()
        if not question:
            raise AssistantError('nothing was asked')
        system = build_system_prompt(get_setting, avoid_phrases)
        kwargs = {'credential_lookup': credential_lookup}
        if opener is not None:
            kwargs['opener'] = opener
        try:
            reply = llm_refiner.chat_reply(
                system, question, get_setting, history=self.history(get_setting),
                model=_clean_string(get_setting('assistant_model', None)),
                timeout=_number(get_setting, 'assistant_timeout', DEFAULT_TIMEOUT, 1, 300), **kwargs)
        except llm_refiner.LLMRefineError as e:
            raise AssistantError(str(e)) from e
        with self._lock:
            self._exchanges = (self._exchanges + [(question, reply)])[-MAX_EXCHANGES:]
            self._last = self._clock()
        return reply
