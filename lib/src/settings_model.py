"""Settings editor model: every setting from config.schema.json, grouped and editable.

The schema is the source of truth: each property becomes a Field with its
type, choices, range, default and description, so a new setting appears in
the editor without code here. SECTIONS only decides where it is listed.
SettingsDocument edits the user's config through ConfigManager (sparse,
atomic saves) and validates it against the schema before saving. API keys
live in credential_manager, never in config.json.

No curses here, so all of it is testable; cli/settings_tui.py draws it.
"""

import contextlib
import copy
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA_FILE = Path(__file__).resolve().parents[2] / 'share' / 'config.schema.json'

UNSET = object()  # a setting with no value and no default ('threads': auto)


# (id, title, blurb, exact keys, key prefixes). A key goes to the first
# section that names it, then the first whose prefix it has, else Advanced.
SECTIONS = (
    ('recording', 'Recording & shortcuts', 'How you start, stop and cancel a dictation.',
     ('primary_shortcut', 'secondary_shortcut', 'secondary_language', 'translate_shortcut', 'cancel_shortcut',
      'recording_mode', 'push_to_talk_lock_seconds', 'silence_timeout', 'continuous_silence_seconds',
      'continuous_silence_threshold', 'grab_keys', 'external_bindings', 'auto_submit'), ()),
    ('transcription', 'Transcription', 'The speech recognizer, its model and language.',
     ('transcription_backend', 'model', 'language', 'threads', 'task', 'sampling_strategy', 'beam_size',
      'whisper_prompt', 'whisper_prompt_en', 'pywhispercpp_use_vad'),
     ('whisper_prompt_', 'faster_whisper_', 'onnx_asr_', 'parakeet_cpp_', 'qwen3_asr_', 'cohere_transcribe_')),
    ('remote', 'Cloud & realtime backends', 'REST and WebSocket transcription services.',
     (), ('rest_', 'websocket_', 'realtime_')),
    ('text', 'Text cleanup', 'Word replacements, filler words and the post-transcription hook.',
     ('word_overrides', 'filter_filler_words', 'filler_words', 'symbol_replacements', 'hallucination_markers',
      'append_trailing_space', 'post_transcription_hook'), ()),
    ('llm', 'LLM refinement & translation', 'Clean up or translate dictations with an LLM, and its prompt.',
     (), ('llm_',)),
    ('scratchpad', 'Scratchpad', 'Draft text over several dictations, then paste it.', (), ('scratchpad_',)),
    ('voice', 'Voice control', 'Wake words, spoken commands and the decision model.', (), ('voice_control_',)),
    ('assistant', 'Voice assistant', 'Questions answered aloud: model, personality, voice.', (), ('assistant_',)),
    ('paste', 'Paste & typing', 'How text reaches the focused app, per app if needed.',
     ('paste_mode', 'hyprland_paste_backend', 'paste_keycode_wev', 'paste_keycode', 'ydotool_modifier_overrides',
      'applications', 'shift_paste', 'inject_mode', 'prefer_clipboard_paste'), ('clipboard_',)),
    ('feedback', 'Sounds, overlay & notifications', 'What you see and hear while recording.',
     ('audio_feedback', 'audio_volume', 'start_sound_volume', 'stop_sound_volume', 'error_sound_volume',
      'start_sound_path', 'stop_sound_path', 'error_sound_path', 'notification_timeout_ms', 'mute_detection',
      'audio_ducking', 'audio_ducking_mode', 'audio_ducking_percent'), ('mic_osd_',)),
    ('longform', 'Long-form sessions', 'Long recordings with pause, resume and submit.', (), ('long_form_',)),
    ('devices', 'Microphone & keyboards', 'Which devices hyprwhspr listens to.',
     ('keepalive_stream', 'stream_start_retry_delay', 'debug_recordings'),
     ('audio_device_', 'selected_device_', 'keyboard_')),
    ('advanced', 'Advanced', 'Everything else.', (), ()),
)

# The settings most people change, listed first
ESSENTIALS = (
    'transcription_backend', 'model', 'language', 'recording_mode', 'primary_shortcut', 'cancel_shortcut',
    'auto_submit', 'paste_mode', 'audio_feedback', 'mic_osd_enabled', 'llm_refine_enabled',
    'llm_translate_language', 'scratchpad_shortcut', 'scratchpad_paste_shortcut', 'voice_control_enabled',
    'voice_control_engine',
)

# Long text: edited in a text editor, with the built-in text as a starting point
PROMPT_KEYS = ('llm_refine_prompt', 'llm_refine_extra_instructions', 'assistant_prompt',
               'whisper_prompt', 'whisper_prompt_en')
_PROMPT_PATTERN = re.compile(r'^whisper_prompt_[a-z]{2}$')

_ACRONYMS = {'llm': 'LLM', 'osd': 'OSD', 'vad': 'VAD', 'url': 'URL', 'urls': 'URLs', 'api': 'API',
             'ms': '(ms)', 'mb': '(MB)', 'id': 'ID', 'ptt': 'PTT', 'tts': 'TTS', 'asr': 'ASR', 'cpp': 'C++',
             'ws': 'WS', 'wev': 'wev', 'onnx': 'ONNX', 'gpu': 'GPU', 'en': '(English)'}


@dataclass
class Field:
    key: str
    schema: dict
    section: str
    kind: str  # bool | choice | int | number | text | prompt | json | credential
    choices: Tuple = ()
    nullable: bool = False
    label: str = ''
    known: bool = True

    @property
    def description(self) -> str:
        return str(self.schema.get('description', ''))


@dataclass
class Credential:
    """An API key in credential_manager, shown in the editor as a setting."""
    id: str
    label: str
    description: str
    kind: str = 'credential'
    key: str = ''

    def __post_init__(self):
        self.key = f'credential:{self.id}'


@dataclass
class Section:
    id: str
    title: str
    blurb: str
    keys: List[str] = field(default_factory=list)


def load_schema(path: Path = SCHEMA_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def section_for(key: str) -> str:
    for section_id, _, _, keys, _ in SECTIONS:
        if key in keys:
            return section_id
    for section_id, _, _, _, prefixes in SECTIONS:
        if any(key.startswith(prefix) for prefix in prefixes):
            return section_id
    return 'advanced'


# Prefixes the section title already says; elsewhere the full name stays ('audio_device_name')
_SHORTENED = {'llm': 'llm_', 'scratchpad': 'scratchpad_', 'voice': 'voice_control_',
              'assistant': 'assistant_', 'longform': 'long_form_'}


def label_for(key: str, section: str) -> str:
    """'voice_control_pause_seconds' in Voice control -> 'Pause seconds'."""
    prefix = _SHORTENED.get(section)
    name = key[len(prefix):] if prefix and key.startswith(prefix) and len(key) > len(prefix) else key
    words = [_ACRONYMS.get(w, w) for w in name.split('_') if w]
    text = ' '.join(words)
    return text[:1].upper() + text[1:] if text else key


def _types(schema) -> Tuple[str, ...]:
    kind = schema.get('type')
    if isinstance(kind, list):
        return tuple(kind)
    return (kind,) if isinstance(kind, str) else ()


def make_field(key: str, schema: dict, known: bool = True) -> Field:
    section = section_for(key) if known else 'advanced'
    types = _types(schema)
    nullable = 'null' in types
    concrete = tuple(t for t in types if t != 'null')
    if 'enum' in schema:
        choices = tuple(schema['enum'])
        if nullable and None not in choices:
            choices = (None,) + choices
        kind = 'choice'
    elif key in PROMPT_KEYS or _PROMPT_PATTERN.match(key):
        kind, choices = 'prompt', ()
    elif concrete == ('boolean',):
        kind, choices = ('choice', (None, True, False)) if nullable else ('bool', ())
    elif concrete == ('integer',):
        kind, choices = 'int', ()
    elif concrete == ('number',):
        kind, choices = 'number', ()
    elif concrete == ('string',):
        kind, choices = 'text', ()
    elif concrete and set(concrete) <= {'string', 'integer', 'number', 'boolean'}:
        kind, choices = 'text', ()  # parsed by type, e.g. a device index or name
    else:
        kind, choices = 'json', ()
    return Field(key, schema, section, kind, choices, nullable, label_for(key, section), known)


def describe_kind(f: Field) -> str:
    """How a field is edited, for the help line."""
    if f.kind == 'credential':
        return 'secret · Enter to set, Backspace to remove'
    lo, hi = _bounds(f.schema)
    if f.kind == 'bool':
        return 'on/off · Space toggles'
    if f.kind == 'choice':
        return 'choice · Space cycles, Enter lists'
    if f.kind in ('int', 'number'):
        span = ''
        if lo is not None or hi is not None:
            span = f" {'' if lo is None else _fmt_num(lo)}–{'' if hi is None else _fmt_num(hi)}"
        return f"{'whole number' if f.kind == 'int' else 'number'}{span} · Enter to type"
    if f.kind == 'prompt':
        return 'long text · Enter opens your editor'
    if f.kind == 'json':
        return 'structured (JSON) · Enter opens your editor'
    return 'text · Enter to type' + (', empty = default' if f.nullable else '')


def _fmt_num(value):
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _bounds(schema):
    lo = schema.get('minimum', schema.get('exclusiveMinimum'))
    hi = schema.get('maximum', schema.get('exclusiveMaximum'))
    return lo, hi


def format_value(f: Field, value, width: int = 40) -> str:
    """A short, one-line rendering of a value."""
    if value is UNSET:
        text = 'auto'
    elif value is None:
        text = 'default'
    elif isinstance(value, bool):
        text = 'on' if value else 'off'
    elif isinstance(value, (int, float)):
        text = _fmt_num(value)
    elif isinstance(value, str):
        text = value.replace('\n', ' ⏎ ') if value else '(empty)'
    elif isinstance(value, dict):
        text = '{}' if not value else f'{len(value)} entr{"y" if len(value) == 1 else "ies"}'
    elif isinstance(value, list):
        text = ', '.join(str(v) for v in value) if value else '(none)'
    else:
        text = json.dumps(value)
    return text if len(text) <= width else text[:max(1, width - 1)] + '…'


def format_choice(value) -> str:
    if value is None:
        return 'default (none)'
    if isinstance(value, bool):
        return 'on' if value else 'off'
    return str(value)


def parse_text(f: Field, text: str):
    """The value typed for a field, or raise ValueError with the reason."""
    text = text.strip() if f.kind != 'prompt' else text
    types = _types(f.schema)
    if text == '' and ('null' in types or f.kind in ('text', 'prompt', 'int', 'number')):
        return None  # cleared: default (or unset), never an empty shortcut or name
    if f.kind == 'int':
        try:
            return int(text)
        except ValueError:
            raise ValueError('enter a whole number') from None
    if f.kind == 'number':
        try:
            value = float(text.replace(',', '.'))
        except ValueError:
            raise ValueError('enter a number') from None
        return int(value) if value.is_integer() and 'integer' in types else value
    if f.kind == 'json':
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f'not valid JSON: {e.msg} (line {e.lineno})') from None
    if f.kind == 'text':
        lowered = text.lower()
        if 'boolean' in types and lowered in ('true', 'false', 'on', 'off'):
            return lowered in ('true', 'on')
        if 'integer' in types and re.fullmatch(r'-?\d+', text):
            return int(text)
        if 'number' in types and re.fullmatch(r'-?\d+(\.\d+)?', text):
            return float(text)
    return text


class SettingsDocument:
    """The user's config, edited in memory and validated before it's saved."""

    def __init__(self, manager=None, schema: Optional[dict] = None,
                 credentials: Optional[dict] = None):
        if manager is None:
            try:
                from .config_manager import ConfigManager
            except ImportError:
                from config_manager import ConfigManager
            with contextlib.redirect_stdout(io.StringIO()):
                manager = ConfigManager(verbose=False)
        self.manager = manager
        self.values = manager.config  # raw: ${VAR} tokens stay as written
        self.defaults = manager.default_config
        self.schema = schema if schema is not None else load_schema()
        self.saved = copy.deepcopy(self.values)
        self.credentials = credentials if credentials is not None else _credential_store()
        self.fields: Dict[str, Field] = {}
        for key, prop in self.schema.get('properties', {}).items():
            if key != '$schema':
                self.fields[key] = make_field(key, prop)
        patterns = self.schema.get('patternProperties', {})
        for key in self.values:
            if key in self.fields or key == '$schema':
                continue
            prop = next((p for pattern, p in patterns.items() if re.match(pattern, key)), None)
            self.fields[key] = make_field(key, prop or {'description': 'Not a known setting; kept as written.'},
                                          known=prop is not None)
        self._validator = None

    # ------------------------ Values ------------------------

    def get(self, key):
        if key in self.values:
            return self.values[key]
        return self.default(key)

    def default(self, key):
        if key in self.defaults:
            return self.defaults[key]
        prop = self.fields[key].schema if key in self.fields else {}
        return prop.get('default', UNSET)

    def set(self, key, value):
        self.values[key] = value

    def reset(self, key):
        """Back to the default; a setting without one goes back to unset."""
        if key in self.defaults:
            self.values[key] = copy.deepcopy(self.defaults[key])
        else:
            self.values.pop(key, None)

    def is_customized(self, key) -> bool:
        if key not in self.values:
            return False
        return key not in self.defaults or self.values[key] != self.defaults[key]

    def changed_keys(self) -> List[str]:
        keys = set(self.values) | set(self.saved)
        return sorted(k for k in keys if k != '$schema' and self.values.get(k, UNSET) != self.saved.get(k, UNSET))

    @property
    def dirty(self) -> bool:
        return bool(self.changed_keys())

    def cycle(self, key, step: int = 1):
        f = self.fields[key]
        if f.kind == 'bool':
            self.set(key, not bool(self.get(key)))
        elif f.kind == 'choice' and f.choices:
            current = self.get(key)
            index = f.choices.index(current) if current in f.choices else -1
            self.set(key, f.choices[(index + step) % len(f.choices)])

    def apply_text(self, key, text) -> Optional[str]:
        """Set a field from typed text; the error message, or None when it took."""
        f = self.fields[key]
        try:
            value = parse_text(f, text)
        except ValueError as e:
            return str(e)
        if value is None and not f.nullable:
            self.reset(key)  # empty: back to the default, or unset
            return None
        error = self.check(key, value)
        if error:
            return error
        self.set(key, value)
        return None

    def check(self, key, value) -> Optional[str]:
        """Why value is invalid for key, or None."""
        f = self.fields.get(key)
        if f is None or not f.known:
            return None
        validator = self._field_validator(f.schema)
        if validator is None:
            return None
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
        return _error_text(errors[0], skip=0) if errors else None

    def errors(self) -> List[Tuple[str, str]]:
        """(key, message) for every invalid setting, checked like 'config validate'."""
        validator = self._root_validator()
        if validator is None:
            return []
        found = []
        sparse = {k: v for k, v in self.values.items() if k != '$schema'}
        for error in validator.iter_errors(sparse):
            key = error.path[0] if error.path else '(config)'
            found.append((str(key), _error_text(error, skip=1)))
        return found

    def save(self) -> bool:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ok = self.manager.save_config()
        if ok:
            self.saved = copy.deepcopy(self.values)
        return ok

    def _root_validator(self):
        if self._validator is None:
            try:
                from jsonschema import Draft202012Validator
            except ImportError:
                return None
            self._validator = Draft202012Validator(self.schema)
        return self._validator

    def _field_validator(self, schema):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            return None
        wrapped = dict(schema)
        if '$defs' in self.schema:
            wrapped['$defs'] = self.schema['$defs']
        return Draft202012Validator(wrapped)

    # ------------------------ Browsing ------------------------

    def sections(self) -> List[Section]:
        """Essentials, then every schema section with its keys, then API keys."""
        by_id = {sid: Section(sid, title, blurb) for sid, title, blurb, _, _ in SECTIONS}
        for key, f in self.fields.items():
            by_id[f.section].keys.append(key)
        # The keys a section names come first, in that order; then the schema's order
        order = {key: i for i, key in enumerate(self.schema.get('properties', {}))}
        named = {sid: keys for sid, _, _, keys, _ in SECTIONS}
        for section in by_id.values():
            listed = named[section.id]
            section.keys.sort(key=lambda k: (0, listed.index(k)) if k in listed else (1, order.get(k, len(order))))
        essentials = Section('essentials', 'Essentials', 'The settings most people change first.',
                             [k for k in ESSENTIALS if k in self.fields])
        keys = Section('credentials', 'API keys', 'Keys for OpenRouter, custom LLM endpoints and services. '
                       'Stored in the credential store, not in config.json.',
                       [c.key for c in self.credential_fields()])
        return [essentials] + [s for s in by_id.values() if s.keys] + [keys]

    def customized_section(self) -> Section:
        return Section('customized', 'Changed from default', 'Every setting you have changed.',
                       [k for k in self.fields if self.is_customized(k)])

    def search(self, query: str) -> List[str]:
        words = [w for w in query.lower().split() if w]
        if not words:
            return []
        found = []
        for key, f in self.fields.items():
            haystack = f'{key} {f.label} {f.description}'.lower()
            if all(w in haystack for w in words):
                found.append(key)
        return sorted(found, key=lambda k: (not all(w in k for w in words), k))

    # ------------------------ API keys ------------------------

    def credential_fields(self) -> List[Credential]:
        found = [
            Credential('openrouter', 'OpenRouter API key',
                       'Used by LLM refinement, translation, the scratchpad, voice control decisions, the voice '
                       'assistant and reading aloud. Starts with sk-or-. Get one at openrouter.ai/keys.'),
            Credential('llm-custom', 'Custom LLM endpoint key',
                       "Sent to llm_refine_endpoint_url when llm_refine_provider is 'custom'. Optional."),
            Credential('huggingface', 'Hugging Face token',
                       'Downloads gated models from the Hugging Face Hub. Optional.'),
        ]
        provider = self.values.get('rest_api_provider') or None
        if isinstance(provider, str) and provider.strip() and provider not in ('openrouter', 'llm-custom'):
            found.append(Credential(provider.strip(), f'REST provider key ({provider.strip()})',
                                    'Sent with REST transcription requests (rest_api_provider).'))
        return found

    def credential(self, key) -> Optional[Credential]:
        return next((c for c in self.credential_fields() if c.key == key), None)

    def credential_status(self, cred: Credential) -> str:
        value = self.credentials['get'](cred.id)
        if not value:
            return 'not set'
        return f"set ({self.credentials['mask'](value)})"

    def set_credential(self, cred: Credential, secret: str) -> bool:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bool(self.credentials['save'](cred.id, secret.strip()))

    def delete_credential(self, cred: Credential) -> bool:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bool(self.credentials['delete'](cred.id))


def builtin_text(key) -> Optional[str]:
    """The built-in prompt a custom one replaces, as a starting point for editing."""
    try:
        if key == 'llm_refine_prompt':
            try:
                from . import llm_refiner
            except ImportError:
                import llm_refiner
            return llm_refiner.DEFAULT_PROMPT
        if key == 'assistant_prompt':
            try:
                from . import voice_assistant
            except ImportError:
                import voice_assistant
            return voice_assistant.DEFAULT_PROMPT
    except Exception:
        return None
    return None


def _credential_store() -> Dict[str, Callable]:
    try:
        from . import credential_manager as store
    except ImportError:
        import credential_manager as store
    return {'get': store.get_credential, 'save': store.save_credential,
            'delete': store.delete_credential, 'mask': store.mask_api_key}


def _error_text(error, skip: int) -> str:
    message = error.message
    if error.validator == 'enum':
        message = 'choose one of: ' + ', '.join(format_choice(v) for v in error.validator_value)
    elif error.validator in ('minimum', 'exclusiveMinimum'):
        message = f'must be at least {_fmt_num(error.validator_value)}' + (
            ' (and more than it)' if error.validator == 'exclusiveMinimum' else '')
    elif error.validator in ('maximum', 'exclusiveMaximum'):
        message = f'must be at most {_fmt_num(error.validator_value)}' + (
            ' (and less than it)' if error.validator == 'exclusiveMaximum' else '')
    elif error.validator == 'type':
        expected = error.validator_value
        expected = ' or '.join(expected) if isinstance(expected, list) else expected
        message = f'must be {expected}'
    inner = list(error.path)[skip:]
    return (f"{'.'.join(str(p) for p in inner)}: " if inner else '') + message
