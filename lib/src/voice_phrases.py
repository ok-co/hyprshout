"""Spoken command phrases: what a short command means without the decision model.

Two tables. Command phrases follow the start word when nothing is recording
("hey jarvis, send the draft"); dictation phrases follow it during a
dictation ("..., hey jarvis, cancel"). A phrase matches a whole command,
never part of one, so "please cancel my subscription" stays a dictation.

Vosk listens for these phrases directly, so they act without transcription
or a network call. Other engines transcribe the command first, and the
transcript is matched against the same tables before the decision model is
asked. voice_control_phrases replaces an action's built-in phrases.

Stdlib only, so core stays importable.
"""

import re
from typing import Callable, Dict, Iterable, Optional, Tuple

try:
    from .voice_command import LANGUAGE_NAMES, configured_languages
except ImportError:
    from voice_command import LANGUAGE_NAMES, configured_languages


# Actions a command phrase can run (app/voice.py). The dictation ones start a
# recording like the decision model's; the draft_* ones edit the scratchpad.
COMMAND_ACTIONS = (
    'cancel', 'dictate', 'dictate_raw', 'translate', 'scratchpad',
    'scratchpad_paste', 'scratchpad_copy', 'scratchpad_undo', 'scratchpad_clear', 'scratchpad_speak',
    'draft_translate', 'draft_shorten', 'draft_polish', 'draft_list',
    'paste_last', 'copy_last', 'assistant', 'help',
)
# What a phrase said during a dictation does with it
DICTATION_ACTIONS = ('finish', 'submit', 'cancel', 'scratchpad', 'raw', 'translate', 'ask', 'continue')

# Instructions the draft_* commands give the scratchpad LLM, like a dictated edit
DRAFT_INSTRUCTIONS = {
    'draft_translate': 'Translate the whole draft into {language}.',
    'draft_shorten': 'Make the draft shorter and tighter. Keep every requirement, name and detail that matters.',
    'draft_polish': ('Polish the draft into a clear, well-structured prompt for a coding agent: '
                     'state the goal first, then the details as short imperative points, with file '
                     'paths, commands and identifiers in backticks. Fix grammar and remove repetition. '
                     'Never add requirements that were not dictated.'),
    'draft_list': 'Turn the draft into a concise bulleted list, one point per item.',
}

COMMAND_PHRASES = {
    'en': {
        'cancel': ('cancel', 'cancel that', 'never mind', 'nevermind', 'forget it', 'nothing', 'stop', 'nothing thanks'),
        'dictate': ('dictate', 'dictation', 'take a message', 'type this', 'write this', 'start dictation'),
        'dictate_raw': ('dictate raw', 'raw', 'type exactly', 'word for word', 'verbatim'),
        'translate': ('translate', 'translate this', 'translation'),
        'scratchpad': ('take a note', 'note', 'add to the draft', 'draft', 'scratchpad', 'edit the draft'),
        'scratchpad_paste': ('send the draft', 'paste the draft', 'send it', 'paste it'),
        'scratchpad_copy': ('copy the draft', 'copy it'),
        'scratchpad_undo': ('undo', 'undo that'),
        'scratchpad_clear': ('clear the draft', 'new draft', 'start over'),
        'scratchpad_speak': ('read the draft', 'read it', 'read it to me', 'read me the draft'),
        'draft_translate': ('translate the draft', 'translate it'),
        'draft_shorten': ('shorten the draft', 'make it shorter', 'shorter'),
        'draft_polish': ('polish the draft', 'polish it', 'clean up the draft', 'clean it up'),
        'draft_list': ('make it a list', 'make a list', 'bullet points'),
        'paste_last': ('paste again', 'paste that again', 'paste the last one'),
        'copy_last': ('copy that', 'copy the last one'),
        'assistant': ('question', 'i have a question', 'i need you', 'are you there'),
        'help': ('help', 'what can i say'),
    },
    'de': {
        'cancel': ('abbrechen', 'vergiss es', 'egal', 'stopp', 'nichts'),
        'dictate': ('diktieren', 'diktat', 'schreib mit'),
        'dictate_raw': ('wörtlich', 'wörtlich diktieren'),
        'translate': ('übersetzen', 'übersetze'),
        'scratchpad': ('notiz', 'notiere', 'zum entwurf', 'entwurf'),
        'scratchpad_paste': ('entwurf senden', 'entwurf einfügen', 'schick es'),
        'scratchpad_copy': ('entwurf kopieren',),
        'scratchpad_undo': ('rückgängig',),
        'scratchpad_clear': ('entwurf löschen', 'neuer entwurf'),
        'scratchpad_speak': ('entwurf vorlesen', 'vorlesen', 'lies vor'),
        'draft_translate': ('entwurf übersetzen',),
        'draft_shorten': ('entwurf kürzen', 'kürzer'),
        'draft_polish': ('entwurf verbessern', 'entwurf aufräumen'),
        'draft_list': ('als liste',),
        'paste_last': ('nochmal einfügen', 'noch einmal einfügen'),
        'copy_last': ('kopieren',),
        'assistant': ('frage', 'ich habe eine frage'),
        'help': ('hilfe', 'was kann ich sagen'),
    },
    'pl': {
        'cancel': ('anuluj', 'nieważne', 'zapomnij', 'stop', 'nic'),
        'dictate': ('dyktuj', 'dyktowanie', 'pisz'),
        'dictate_raw': ('dosłownie', 'dyktuj dosłownie'),
        'translate': ('tłumacz', 'przetłumacz'),
        'scratchpad': ('notatka', 'zanotuj', 'do szkicu', 'szkic'),
        'scratchpad_paste': ('wyślij szkic', 'wklej szkic'),
        'scratchpad_copy': ('skopiuj szkic',),
        'scratchpad_undo': ('cofnij',),
        'scratchpad_clear': ('wyczyść szkic', 'nowy szkic'),
        'scratchpad_speak': ('przeczytaj szkic', 'przeczytaj'),
        'draft_translate': ('przetłumacz szkic',),
        'draft_shorten': ('skróć szkic', 'krócej'),
        'draft_polish': ('popraw szkic', 'uporządkuj szkic'),
        'draft_list': ('zrób listę',),
        'paste_last': ('wklej ponownie', 'wklej jeszcze raz'),
        'copy_last': ('skopiuj to',),
        'assistant': ('pytanie', 'mam pytanie'),
        'help': ('pomoc', 'co mogę powiedzieć'),
    },
    'fr': {
        'cancel': ('annuler', 'laisse tomber', 'rien'),
        'dictate': ('dicter', 'dictée'),
        'dictate_raw': ('mot pour mot',),
        'translate': ('traduire', 'traduis'),
        'scratchpad': ('note', 'prends une note', 'ajoute au brouillon', 'brouillon'),
        'scratchpad_paste': ('envoie le brouillon', 'colle le brouillon'),
        'scratchpad_copy': ('copie le brouillon',),
        'scratchpad_undo': ('défaire', 'annule ça'),
        'scratchpad_clear': ('efface le brouillon', 'nouveau brouillon'),
        'scratchpad_speak': ('lis le brouillon',),
        'draft_translate': ('traduis le brouillon',),
        'draft_shorten': ('raccourcis le brouillon',),
        'draft_polish': ('améliore le brouillon',),
        'draft_list': ('fais une liste',),
        'paste_last': ('colle encore',),
        'copy_last': ('copie ça',),
        'assistant': ('question', "j'ai une question"),
        'help': ('aide',),
    },
    'es': {
        'cancel': ('cancelar', 'olvídalo', 'nada'),
        'dictate': ('dictar', 'dictado'),
        'dictate_raw': ('literal',),
        'translate': ('traducir', 'traduce'),
        'scratchpad': ('nota', 'toma nota', 'añade al borrador', 'borrador'),
        'scratchpad_paste': ('envía el borrador', 'pega el borrador'),
        'scratchpad_copy': ('copia el borrador',),
        'scratchpad_undo': ('deshacer', 'deshaz'),
        'scratchpad_clear': ('borra el borrador', 'nuevo borrador'),
        'scratchpad_speak': ('lee el borrador',),
        'draft_translate': ('traduce el borrador',),
        'draft_shorten': ('acorta el borrador',),
        'draft_polish': ('mejora el borrador',),
        'draft_list': ('haz una lista',),
        'paste_last': ('pega otra vez',),
        'copy_last': ('cópialo',),
        'assistant': ('pregunta', 'tengo una pregunta'),
        'help': ('ayuda', 'qué puedo decir'),
    },
}

DICTATION_PHRASES = {
    'en': {
        'finish': ('done', 'finish', 'finished', "that's it", 'that is it', 'over', 'send'),
        'submit': ('submit', 'send and submit', 'and submit', 'enter', 'press enter'),
        'cancel': ('cancel', 'cancel that', 'never mind', 'scrap that', 'delete that', 'forget it'),
        'scratchpad': ('to the draft', 'into the draft', 'add to the draft', 'note it', 'save as a note'),
        'raw': ('raw', 'paste raw', 'no cleanup'),
        'translate': ('translate', 'translate it', 'translate that'),
        'ask': ('ask', 'answer that', 'answer it'),
        'continue': ('continue', 'keep going', 'go on', 'not yet'),
    },
    'de': {
        'finish': ('fertig', 'ende', 'senden'),
        'submit': ('abschicken', 'senden und abschicken', 'enter'),
        'cancel': ('abbrechen', 'verwerfen', 'vergiss es'),
        'scratchpad': ('in den entwurf', 'zum entwurf', 'als notiz'),
        'raw': ('wörtlich',),
        'translate': ('übersetzen', 'übersetze das'),
        'ask': ('beantworten', 'beantworte das'),
        'continue': ('weiter', 'weitermachen', 'noch nicht'),
    },
    'pl': {
        'finish': ('gotowe', 'koniec', 'wyślij'),
        'submit': ('zatwierdź', 'wyślij i zatwierdź', 'enter'),
        'cancel': ('anuluj', 'nieważne', 'skasuj to'),
        'scratchpad': ('do szkicu', 'zanotuj to'),
        'raw': ('dosłownie',),
        'translate': ('przetłumacz', 'przetłumacz to'),
        'ask': ('odpowiedz', 'odpowiedz na to'),
        'continue': ('dalej', 'kontynuuj', 'jeszcze nie'),
    },
    'fr': {
        'finish': ('terminé', 'fini', 'envoie'),
        'submit': ('valide', 'envoie et valide'),
        'cancel': ('annuler', 'laisse tomber', 'efface ça'),
        'scratchpad': ('dans le brouillon', 'en note'),
        'raw': ('mot pour mot',),
        'translate': ('traduis', 'traduis ça'),
        'ask': ('réponds', 'réponds à ça'),
        'continue': ('continue', 'pas encore'),
    },
    'es': {
        'finish': ('listo', 'terminado', 'envía'),
        'submit': ('confirma', 'envía y confirma'),
        'cancel': ('cancelar', 'olvídalo', 'bórralo'),
        'scratchpad': ('al borrador', 'como nota'),
        'raw': ('literal',),
        'translate': ('tradúcelo', 'traduce eso'),
        'ask': ('responde', 'responde eso'),
        'continue': ('sigue', 'continúa', 'todavía no'),
    },
}

# English phrases that name a language, generated for voice_control_languages
_LANGUAGE_TEMPLATES = {
    'translate': ('translate into {name}', 'translate to {name}', 'in {name}'),
    'draft_translate': ('translate the draft into {name}', 'translate the draft to {name}',
                        'translate it into {name}'),
}

# Politeness around a transcribed command; never part of a Vosk phrase
_FILLER_START = ('please', 'okay', 'ok', 'so', 'and', 'um', 'uh')
_FILLER_END = ('please', 'thanks', 'thank you', 'now')

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

Match = Tuple[str, Optional[str]]  # (action, language code or None)


def normalize(text) -> str:
    """'Never mind!' -> 'never mind': lower-case words, apostrophes kept, punctuation dropped."""
    if not isinstance(text, str):
        return ''
    return ' '.join(w.replace('’', "'") for w in _WORD_RE.findall(text.lower()))


def table_language(code) -> Optional[str]:
    """The built-in table for a language code or Vosk model setting ('en-us' -> 'en'); None if none."""
    if not isinstance(code, str):
        return None
    code = code.strip().lower().replace('_', '-')
    base = code.split('-')[0]
    return base if base in COMMAND_PHRASES else None


class PhraseBook:
    """Command and dictation phrases for some languages, with user overrides applied."""

    def __init__(self, get_setting: Callable, languages: Optional[Iterable[str]] = None):
        selected = tuple(languages) if languages is not None else tuple(COMMAND_PHRASES)
        overrides = get_setting('voice_control_phrases', None)
        overrides = overrides if isinstance(overrides, dict) else {}
        self.command = self._build(COMMAND_PHRASES, COMMAND_ACTIONS, selected, overrides.get('command'))
        self.dictation = self._build(DICTATION_PHRASES, DICTATION_ACTIONS, selected, overrides.get('dictation'))
        if 'en' in selected:
            self._add_language_phrases(get_setting, overrides.get('command'))

    @staticmethod
    def _build(tables, actions, languages, override) -> Dict[str, Match]:
        override = override if isinstance(override, dict) else {}
        book = {}
        for action in actions:
            if action in override:
                phrases = override[action] if isinstance(override[action], (list, tuple)) else ()
            else:
                phrases = [p for lang in languages for p in tables.get(lang, {}).get(action, ())]
            for phrase in phrases:
                phrase = normalize(phrase)
                if phrase:
                    book.setdefault(phrase, (action, None))
        return book

    def _add_language_phrases(self, get_setting, override):
        override = override if isinstance(override, dict) else {}
        for action, templates in _LANGUAGE_TEMPLATES.items():
            if action in override:
                continue
            for code in configured_languages(get_setting):
                name = LANGUAGE_NAMES.get(code)
                for template in templates if name else ():
                    self.command.setdefault(normalize(template.format(name=name)), (action, code))

    def phrases(self) -> Tuple[str, ...]:
        """Every phrase, for a recognizer's grammar."""
        return tuple(sorted(set(self.command) | set(self.dictation)))

    def match_command(self, text) -> Optional[Match]:
        return _match(self.command, text)

    def match_dictation(self, text) -> Optional[Match]:
        return _match(self.dictation, text)

    def first_phrase(self, table: str, action: str) -> Optional[str]:
        book = self.command if table == 'command' else self.dictation
        return next((p for p, (a, code) in book.items() if a == action and code is None), None)

    def help_text(self, start_phrase: str) -> str:
        """A short list of what can be said, for the 'help' command."""
        def listing(table, actions):
            said = [self.first_phrase(table, a) for a in actions]
            return ' · '.join(p for p in said if p)
        start = start_phrase or 'the start word'
        return (f'After "{start}": '
                + listing('command', ('dictate', 'scratchpad', 'scratchpad_paste', 'scratchpad_undo',
                                      'scratchpad_speak', 'draft_translate', 'draft_polish',
                                      'paste_last', 'assistant', 'cancel'))
                + f'\nDuring a dictation, "{start}" then: '
                + listing('dictation', DICTATION_ACTIONS)
                + ', or nothing to finish')


def _match(book, text) -> Optional[Match]:
    words = normalize(text)
    if not words:
        return None
    if words in book:
        return book[words]
    trimmed = _strip_filler(words)
    return book.get(trimmed) if trimmed != words else None


def _strip_filler(words: str) -> str:
    changed = True
    while changed and words:
        changed = False
        for filler in _FILLER_START:
            if words.startswith(filler + ' '):
                words, changed = words[len(filler) + 1:], True
        for filler in _FILLER_END:
            if words.endswith(' ' + filler):
                words, changed = words[:-len(filler) - 1], True
    return words


def remove_phrase(text: str, phrase: str, threshold: float = 0.75) -> str:
    """Drop every fuzzy occurrence of a spoken phrase from a transcript.

    Used when the start word was said inside a dictation that kept going
    ("hey jarvis, keep going"): the words were recorded, but aren't meant
    for the text.
    """
    import difflib

    target = re.sub(r'\W', '', (phrase or '').lower())
    size = len((phrase or '').split())
    if not text or not target or not size:
        return text
    words = list(re.finditer(r"\w[\w']*", text, re.UNICODE))
    candidates = []
    for i in range(len(words)):
        for count in (size, size + 1, size - 1):
            if count < 1 or i + count > len(words):
                continue
            joined = ''.join(m.group(0).lower() for m in words[i:i + count])
            ratio = difflib.SequenceMatcher(None, joined, target).ratio()
            if ratio >= threshold:
                candidates.append((ratio, i, i + count))
    # Best matches first, so a window one word too wide never wins over the exact one
    taken, cuts = set(), []
    for ratio, first, last in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if taken.isdisjoint(range(first, last)):
            taken.update(range(first, last))
            cuts.append((words[first].start(), words[last - 1].end()))
    for start, end in sorted(cuts, reverse=True):
        before, after = text[:start].rstrip(' \t,;:-'), text[end:].lstrip(' \t,;:-')
        # A full stop after the phrase ends the sentence before it, unless that one already ended
        if after[:1] in '.!?' and (not before or before.endswith(('.', '!', '?', '\n', ',', ';', ':'))):
            after = after[1:].lstrip()
        joiner = ' ' if before and after and after[:1] not in '.,!?;:' else ''
        text = before + joiner + after
    return re.sub(r'[ \t]{2,}', ' ', text).strip()
