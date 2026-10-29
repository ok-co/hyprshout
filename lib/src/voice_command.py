"""
Voice commands: one spoken command becomes one hyprwhspr action.

The command transcript goes to a decision model on OpenRouter's Decisions API
(Jev by default). Decision models pick from options the caller defines and
return no free text, so the model can only ever choose an entry of ACTIONS;
nothing it returns is executed as text.

Stdlib only (urllib), like llm_refiner, so core stays importable.
"""

import difflib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import urlsplit

try:
    from .backend_utils import redact_url
except ImportError:
    from backend_utils import redact_url


DEFAULT_ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'
DEFAULT_MODEL = 'typesafe/jev-1.14'
DEFAULT_TIMEOUT = 10
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_LANGUAGES = ('en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl')
# The OpenRouter key stored by 'hyprwhspr config llm'
CREDENTIAL = 'openrouter'

ACTIONS = {
    'dictate': 'Start a dictation: what the user says next is typed into the focused application.',
    'dictate_raw': 'Start a dictation typed exactly as spoken, without AI cleanup or rewriting '
                   '(verbatim, raw, exact words).',
    'translate': 'Start a dictation that is translated into another language before it is typed.',
    'scratchpad': 'Start a dictation into the scratchpad draft: new text for the draft, or an '
                  'instruction to change it (take a note, add to the draft, edit the draft).',
    'scratchpad_paste': 'Paste the scratchpad draft into the focused application (send the draft).',
    'scratchpad_copy': 'Copy the scratchpad draft to the clipboard.',
    'scratchpad_undo': 'Undo the last change to the scratchpad draft.',
    'scratchpad_clear': 'Clear the scratchpad draft and start a new one.',
    'paste_last': 'Paste the previous dictation again.',
    'copy_last': 'Copy the previous dictation to the clipboard.',
    'nothing': 'None of these: unclear, unrelated, or the user cancels or changes their mind.',
}

LANGUAGE_NAMES = {
    'ar': 'Arabic', 'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'fi': 'Finnish', 'fr': 'French', 'he': 'Hebrew',
    'hi': 'Hindi', 'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese',
    'ko': 'Korean', 'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese',
    'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sv': 'Swedish', 'th': 'Thai',
    'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese', 'zh': 'Chinese',
}


class DecisionError(Exception):
    """The decision model gave no usable answer."""


@dataclass(frozen=True)
class VoiceDecision:
    action: str
    confidence: float
    spoken_language: Optional[str] = None
    target_language: Optional[str] = None
    uncertain: bool = False


@dataclass(frozen=True)
class DictationPlan:
    """How one voice-started dictation is delivered."""
    route: str = 'paste'                      # the recording route: 'paste' | 'translate' | 'scratchpad'
    raw: bool = False                         # paste without LLM refinement
    translate_language: Optional[str] = None  # this dictation's target instead of llm_translate_language


DICTATION_PLANS = {
    'dictate': DictationPlan(),
    'dictate_raw': DictationPlan(raw=True),
    'translate': DictationPlan(route='translate'),
    'scratchpad': DictationPlan(route='scratchpad'),
}

# Immediate actions named after the scratchpad socket requests they make (app/scratchpad.py)
SCRATCHPAD_ACTIONS = ('scratchpad_paste', 'scratchpad_copy', 'scratchpad_undo', 'scratchpad_clear')


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def configured_languages(get_setting: Callable) -> tuple:
    value = get_setting('voice_control_languages', list(DEFAULT_LANGUAGES))
    if not isinstance(value, (list, tuple)):
        return DEFAULT_LANGUAGES
    seen = []
    for code in value:
        if isinstance(code, str) and code.strip() and code.strip().lower() not in seen:
            seen.append(code.strip().lower())
    return tuple(seen)


def build_request(transcript: str, get_setting: Callable) -> dict:
    """The Decisions API body for one command transcript."""
    model = get_setting('voice_control_decision_model', DEFAULT_MODEL) or DEFAULT_MODEL
    questions = {
        'action': {
            'type': 'choice',
            'instructions': 'The user said a wake word to the hyprwhspr dictation app and then '
                            'spoke this command. Which action do they want next?',
            'criteria': dict(ACTIONS),
        },
    }
    languages = configured_languages(get_setting)
    if languages:
        names = {code: f'{language_name(code)} ({code})' for code in languages}
        questions['spoken_language'] = {
            'type': 'choice',
            'instructions': 'If the user starts a dictation, which language will they speak in it? '
                            "Pick 'auto' unless the command names the language they will speak.",
            'criteria': {'auto': 'Not stated; detect it automatically.', **names},
        }
        questions['target_language'] = {
            'type': 'choice',
            'instructions': 'If the user wants their dictation translated, into which language? '
                            "Pick 'none' when the command asks for no translation or names no language.",
            'criteria': {'none': 'No translation, or no target language named.', **names},
        }
    return {
        'model': model,
        'state': {
            'app': 'hyprwhspr, a speech-to-text dictation tool for the Linux desktop',
            'spoken_command': transcript,
        },
        'questions': questions,
    }


def _answer_confidence(answer: dict) -> float:
    confidence = answer.get('confidence')
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        probabilities = answer.get('probabilities')
        confidence = probabilities.get(answer.get('choice')) if isinstance(probabilities, dict) else None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return 1.0
    return float(confidence)


def _choice(answers: dict, name: str, options) -> tuple:
    answer = answers.get(name)
    if not isinstance(answer, dict) or answer.get('choice') not in options:
        raise DecisionError(f'decision model returned no valid {name!r} choice')
    return answer['choice'], _answer_confidence(answer)


def parse_response(data, languages: Iterable[str], min_confidence: float) -> VoiceDecision:
    """VoiceDecision from a Decisions API response; low-confidence picks become defaults."""
    answers = data.get('answers') if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        raise DecisionError('unexpected response shape')
    action, confidence = _choice(answers, 'action', ACTIONS)
    if confidence < min_confidence:
        return VoiceDecision('nothing', confidence, uncertain=True)

    languages = tuple(languages)
    spoken = target = None
    if languages:
        code, sure = _choice(answers, 'spoken_language', ('auto',) + languages)
        spoken = code if code != 'auto' and sure >= min_confidence else None
        code, sure = _choice(answers, 'target_language', ('none',) + languages)
        target = code if code != 'none' and sure >= min_confidence else None
    return VoiceDecision(action, confidence, spoken, target)


def _is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or '').lower()
    return host in ('localhost', '127.0.0.1', '::1')


def _min_confidence(get_setting: Callable) -> float:
    value = get_setting('voice_control_min_confidence', DEFAULT_MIN_CONFIDENCE)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)
    return DEFAULT_MIN_CONFIDENCE


def _timeout(get_setting: Callable) -> float:
    value = get_setting('voice_control_decision_timeout', DEFAULT_TIMEOUT)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 120:
        return float(value)
    return float(DEFAULT_TIMEOUT)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode('utf-8', 'replace'))
        message = body.get('error', {}).get('message') if isinstance(body, dict) else None
    except Exception:
        message = None
    if isinstance(message, str) and message.strip():
        return f'HTTP {exc.code}: {message.strip()[:200]}'
    return f'HTTP {exc.code}'


def decide(transcript: str, get_setting: Callable,
           credential_lookup: Optional[Callable] = None,
           opener: Callable = urllib.request.urlopen) -> VoiceDecision:
    """Ask the decision model what a command transcript means, or raise DecisionError."""
    endpoint = get_setting('voice_control_decision_endpoint', DEFAULT_ENDPOINT) or DEFAULT_ENDPOINT
    if not isinstance(endpoint, str) or not endpoint.lower().startswith(('https://', 'http://')):
        raise DecisionError('voice_control_decision_endpoint must be an http:// or https:// URL')
    if endpoint.lower().startswith('http://') and not _is_loopback(endpoint):
        raise DecisionError('refusing to send an API key over plain http to a non-local host')
    if credential_lookup is None:
        try:
            from .credential_manager import get_credential as credential_lookup
        except ImportError:
            from credential_manager import get_credential as credential_lookup
    api_key = credential_lookup(CREDENTIAL)
    api_key = api_key.strip() if isinstance(api_key, str) else ''
    if not api_key:
        raise DecisionError("no OpenRouter API key stored; run 'hyprwhspr config llm'")

    body = build_request(transcript, get_setting)
    request = urllib.request.Request(
        endpoint, data=json.dumps(body).encode('utf-8'), method='POST', headers={
            'Content-Type': 'application/json',
            'User-Agent': 'hyprwhspr',
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://hyprwhspr.com',
            'X-Title': 'hyprwhspr',
        },
    )
    try:
        with opener(request, timeout=_timeout(get_setting)) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise DecisionError(_error_detail(e)) from e
    except urllib.error.URLError as e:
        raise DecisionError(f'cannot reach {redact_url(endpoint)}: {e.reason}') from e
    except TimeoutError as e:
        raise DecisionError('request timed out') from e
    except (OSError, ValueError) as e:
        raise DecisionError(f'request failed: {type(e).__name__}') from e
    return parse_response(data, configured_languages(get_setting), _min_confidence(get_setting))


def plan_for(decision: VoiceDecision) -> Optional[DictationPlan]:
    """The dictation a decision starts, or None for an immediate action.

    A translation the command names a language for targets that language;
    otherwise llm_translate_language applies.
    """
    plan = DICTATION_PLANS.get(decision.action)
    if plan is None or plan.route != 'translate' or not decision.target_language:
        return plan
    return DictationPlan(route='translate', translate_language=language_name(decision.target_language))


# ------------------------ Transcript helpers ------------------------

_WORD_RE = re.compile(r"\w[\w']*", re.UNICODE)


def phrase_for(word: str) -> str:
    """Spoken form of a wake word setting: 'hey_jarvis' or '.../hey_computer_v0.2.onnx' -> 'hey jarvis'."""
    stem = Path(word).name if word.endswith(('.onnx', '.tflite')) else word
    stem = re.sub(r'\.(onnx|tflite)$', '', stem)
    stem = re.sub(r'_v\d+(\.\d+)*$', '', stem)
    return ' '.join(stem.replace('-', '_').split('_')).strip().lower()


def strip_wake_phrase(text: str, start_phrase: Optional[str] = None,
                      end_phrase: Optional[str] = None, threshold: float = 0.75) -> str:
    """Drop a start phrase leading, or an end phrase trailing, the transcript.

    The listener cuts the words out of the audio, but a late detection can
    leave part of one; matching is fuzzy because recognizers spell names
    freely ("hey jervis", "Hey, Jarvis."). Each phrase is only looked for at
    its own edge, so dictated text that merely resembles it stays intact.
    """
    if not text:
        return text
    for phrase, at_end in ((end_phrase, True), (start_phrase, False)):
        target = re.sub(r'\W', '', (phrase or '').lower())
        if not target:
            continue
        words = list(_WORD_RE.finditer(text))
        if words:
            text = _strip_edge(text, words, target, len(phrase.split()), threshold, at_end)
    return text.strip()


def _strip_edge(text, words, target, size, threshold, at_end):
    for count in (size, size + 1, size - 1):
        if count < 1 or count > len(words):
            continue
        window = words[-count:] if at_end else words[:count]
        joined = ''.join(m.group(0).lower() for m in window)
        if difflib.SequenceMatcher(None, joined, target).ratio() >= threshold:
            if at_end:
                return text[:window[0].start()].rstrip(' \t,;:-')
            return text[window[-1].end():].lstrip(' \t,;:.!?-')
    return text
