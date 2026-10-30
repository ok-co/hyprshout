"""Voice control: wake words, quiet command sessions and voice-started dictations.

The start word opens a quiet command session: the wake word stream itself
keeps the audio (no recorder, OSD, ducking or paste) until the end word or,
with voice_control_end_on_pause, the pause after the command. A command that
is one of the phrases in voice_phrases ("send the draft", "cancel") runs at
once: Vosk spots it in the stream, other engines match its transcript.
Anything else goes to a decision model, which picks one action
(voice_command.ACTIONS). Dictation actions start a normal recording on the
plan's route (paste, translate, scratchpad, assistant), which the end word
stops like any other recording; scratchpad, last-dictation and 'ask' actions
run at once.

The start word inside a dictation opens a short window for a dictation
phrase: silence finishes the dictation, "cancel" drops it, "into the draft"
or "translate" changes where it goes, "keep going" carries on recording.
Whatever was said from the start word on is cut from the dictation.

Manual control always wins: a hotkey or CLI start cancels a pending command,
and the cancel shortcut or command cancels it too.

In long_form mode the listener runs one word instead, the pause word, which
pauses and resumes the long-form session.
"""

import dataclasses
import threading
import time

import llm_refiner
from paths import MODEL_UNLOADED_FILE
from service_log import log
from backend_utils import normalize_backend
from hallucination import is_hallucination
import voice_command
from voice_command import (SCRATCHPAD_ACTIONS, DecisionError, DictationPlan, VoiceDecision, language_name,
                           phrase_for, plan_for, strip_wake_phrase)
from voice_phrases import DRAFT_INSTRUCTIONS, PhraseBook, table_language
from wake_word import ENGINES, SAMPLE_RATE, WakeWordError, WakeWordListener, normalize_word

DEFAULT_START_WORD = 'hey_jarvis'
DEFAULT_END_WORD = 'hey_mycroft'
DEFAULT_PAUSE_SECONDS = 1.5
DEFAULT_SILENCE_SECONDS = 5.0

# What a voice-started dictation is called in its announcement
_DICTATION_LABELS = {
    'dictate': 'Dictating',
    'dictate_raw': 'Dictating, exactly as spoken',
    'translate': 'Dictating, translated',
    'scratchpad': 'Dictating into the draft',
    'assistant': 'Listening for your question',
}
# Routes a dictation phrase sends the dictation down
_DICTATION_ROUTES = {'scratchpad': 'scratchpad', 'translate': 'translate', 'ask': 'assistant'}
_DICTATION_ANNOUNCEMENTS = {
    'cancel': 'Dictation discarded',
    'scratchpad': 'Into the draft',
    'translate': 'Translating the dictation',
    'ask': 'Asking the assistant',
    'raw': 'Pasting exactly as spoken',
    'submit': 'Pasting and pressing Enter',
    'continue': 'Still listening',
}


class VoiceControlMixin:
    """Voice control: wake words, quiet command sessions and voice-started dictations."""

    # ------------------------ Lifecycle ------------------------

    def _start_voice_control(self):
        """Start the wake word listener when voice_control_enabled; False if it can't run.

        long_form mode runs the pause word listener instead.
        """
        long_form = self.config.get_setting('recording_mode', 'toggle') == 'long_form'
        if not self.config.get_setting('voice_control_enabled', False):
            return self._start_pause_word() if long_form else False
        reason = self._voice_control_unsupported()
        if reason:
            log(f"[VOICE] Voice control disabled: {reason}")
            self._notify_user("hyprwhspr", f"Voice control disabled: {reason}", urgency="normal")
            return self._start_pause_word() if long_form else False
        engine_options = self._wake_engine_options()
        if engine_options is None:
            engine = self.config.get_setting('voice_control_engine', 'openwakeword')
            self._notify_user("hyprwhspr", f"Voice control: unknown engine {engine!r}", urgency="normal")
            return False
        try:
            start_word = normalize_word(self.config.get_setting('voice_control_start_word', DEFAULT_START_WORD))
            end_word = normalize_word(self.config.get_setting('voice_control_end_word', DEFAULT_END_WORD))
        except WakeWordError as e:
            log(f"[VOICE] Voice control disabled: {e}")
            return False
        if start_word == end_word:
            log("[VOICE] Voice control disabled: the start and end word must differ")
            self._notify_user("hyprwhspr", "Voice control: the start and end word must differ", urgency="normal")
            return False
        self._voice_start_phrase = phrase_for(start_word)
        self._voice_end_phrase = phrase_for(end_word)
        self._voice_phrasebook = self._build_phrasebook(engine_options['engine'])
        if engine_options['engine'] == 'vosk':
            engine_options['phrases'] = self._voice_phrasebook.phrases()
        self._voice_listener = WakeWordListener(
            start_word, end_word, self._on_wake_event,
            threshold=self._get_float_setting('voice_control_threshold', 0.5),
            device=self.audio_capture.device_id,
            max_capture_seconds=self._get_float_setting('voice_control_max_command_seconds', 15.0),
            **engine_options,
        )
        return self._voice_listener.start()

    def _build_phrasebook(self, engine):
        """Phrases in the Vosk model's language, or every built-in language for transcript matching."""
        languages = None
        if engine == 'vosk':
            language = table_language(self.config.get_setting('voice_control_vosk_model', 'en-us'))
            languages = [language] if language else None
        return PhraseBook(self.config.get_setting, languages)

    def _voice_phrases(self):
        book = getattr(self, '_voice_phrasebook', None)
        if book is None:
            book = self._voice_phrasebook = PhraseBook(self.config.get_setting)
        return book

    def _voice_control_unsupported(self):
        if self.config.get_setting('recording_mode', 'toggle') == 'long_form':
            return 'not available in long_form recording mode'
        backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
        if backend == 'realtime-ws' and not self._voice_commands_stay_local():
            return ("the realtime-ws backend cannot transcribe voice commands; use a local or REST backend, "
                    "or voice_control_engine 'vosk', which transcribes them itself")
        if not self.audio_capture.input_is_shared():
            return 'the microphone is raw ALSA; the wake word stream needs PipeWire or PulseAudio'
        return None

    def _voice_commands_stay_local(self):
        """Vosk transcribes commands itself unless voice_control_command_transcriber is 'backend'."""
        engine = str(self.config.get_setting('voice_control_engine', 'openwakeword') or '').strip().lower()
        transcriber = self.config.get_setting('voice_control_command_transcriber', 'auto')
        return engine == 'vosk' and transcriber != 'backend'

    def _wake_engine_options(self):
        """WakeWordListener engine settings; None (logged) for an unknown voice_control_engine."""
        engine = str(self.config.get_setting('voice_control_engine', 'openwakeword') or '').strip().lower()
        if engine not in ENGINES:
            log(f"[VOICE] Voice control disabled: unknown voice_control_engine {engine!r}")
            return None
        return {
            'engine': engine,
            'keyword_threshold': self._get_float_setting('voice_control_keyword_threshold', 0.25),
            'keyword_boost': self._get_float_setting('voice_control_keyword_boost', 1.0),
            'vosk_model': self.config.get_setting('voice_control_vosk_model', 'en-us'),
            'vosk_confidence': self._get_float_setting('voice_control_vosk_confidence', 0.9),
        }

    def _start_pause_word(self):
        """Long-form mode: listen for long_form_pause_word; False when unset or it can't run."""
        word = (self.config.get_setting('long_form_pause_word', '') or '').strip()
        if not word:
            return False
        if not self.audio_capture.input_is_shared():
            log("[VOICE] Pause word disabled: the microphone is raw ALSA; "
                "the wake word stream needs PipeWire or PulseAudio")
            return False
        try:
            word = normalize_word(word)
        except WakeWordError as e:
            log(f"[VOICE] Pause word disabled: {e}")
            return False
        engine_options = self._wake_engine_options()
        if engine_options is None:
            return False
        self._voice_listener = WakeWordListener(
            word, None, self._on_pause_word,
            threshold=self._get_float_setting('voice_control_threshold', 0.5),
            device=self.audio_capture.device_id,
            **engine_options,
        )
        return self._voice_listener.start()

    def _on_pause_word(self, kind, detected_at):
        if kind != 'start':
            return
        trim = self._get_float_setting('voice_control_end_trim_seconds', 1.0)
        # Pausing joins the capture thread; keep the listener's worker free
        threading.Thread(target=self._longform.pause_word, args=(detected_at, trim),
                         daemon=True, name="PauseWord").start()

    def _stop_voice_control(self):
        listener = getattr(self, '_voice_listener', None)
        if listener is not None:
            listener.stop()

    # ------------------------ Wake word events ------------------------

    def _on_wake_event(self, kind, detected_at):
        """Listener callback (its worker thread): decide quickly, hand slow work to a thread."""
        if kind == 'start':
            self._voice_on_start_word(detected_at)
        elif kind == 'end':
            if self._voice_finish_command(detected_at):
                return
            if self._voice_dictation_resolve('finish'):
                return
            if self.is_recording:
                threading.Thread(target=self._voice_stop_recording, args=(detected_at,),
                                 daemon=True, name="VoiceStop").start()
        elif kind.startswith('phrase:'):
            self._voice_on_phrase(kind[len('phrase:'):])
        elif kind == 'pause':
            self._voice_on_pause(detected_at)
        elif kind == 'silence':
            self._voice_on_silence()
        elif kind == 'capture_limit':
            if self._voice_dictation_resolve('finish'):
                return
            if self._voice_cancel_command("too long - say the end word to finish a command"):
                self.audio_manager.play_error_sound()

    def _voice_on_start_word(self, detected_at):
        with self._voice_lock:
            state = self._voice_state
        if state == 'command':
            # Said twice: never mind
            if self._voice_cancel_command("the start word was said again"):
                self._voice_cue('stop')
                self._voice_announce('Cancelled')
            return
        if state in ('dictation', 'dictation_deciding'):
            self._voice_dictation_resolve('finish')  # said twice: done
            return
        if self._voice_dictation_window_allowed():
            self._voice_open_dictation_window(detected_at)
            return
        self._voice_begin_command(detected_at)

    def _voice_begin_command(self, detected_at=None):
        with self._recording_lock:
            busy = (self.is_recording or self.is_processing or self._recording_starting
                    or self._longform_active)
        if busy:
            return  # the start word inside a dictation is just a word
        with self._voice_lock:
            if self._voice_state != 'idle':
                return
            self._voice_state = 'command'
            self._voice_generation += 1
            # From the end of the start word: an engine that reports it late
            # (Vosk, at the next pause) must not lose the command's first words
            self._voice_listener.begin_capture(since=detected_at, **self._voice_command_watch())
        # The command must not be transcribed over the draft or a reply being read aloud
        self._stop_scratchpad_speech()
        self._stop_assistant_speech()
        log("[VOICE] Listening for a command")
        self._voice_cue('start')

    def _voice_command_watch(self):
        """How a command capture ends itself: the pause after it, or nothing said at all."""
        watch = {'silence_seconds': self._voice_silence_seconds()}
        if self.config.get_setting('voice_control_end_on_pause', True):
            watch['pause_seconds'] = self._voice_pause_seconds()
        return watch

    def _voice_pause_seconds(self):
        seconds = self._get_float_setting('voice_control_pause_seconds', DEFAULT_PAUSE_SECONDS)
        return seconds if seconds > 0 else DEFAULT_PAUSE_SECONDS

    def _voice_silence_seconds(self):
        seconds = self._get_float_setting('voice_control_silence_seconds', DEFAULT_SILENCE_SECONDS)
        return max(0.0, seconds)

    def _voice_finish_command(self, detected_at, trim_seconds=None):
        """End a command capture and decide on it; False when no command was open.

        trim_seconds defaults to the end word's length plus the time since it.
        """
        with self._voice_lock:
            if self._voice_state != 'command':
                return False
            self._voice_state = 'deciding'
            generation = self._voice_generation
            trim = self._voice_trim_seconds(detected_at) if trim_seconds is None else trim_seconds
            audio = self._voice_listener.end_capture(trim)
        self._voice_cue('stop')
        threading.Thread(target=self._run_voice_command, args=(generation, audio),
                         daemon=True, name="VoiceCommand").start()
        return True

    def _voice_cancel_command(self, reason):
        """Drop an open or deciding command; True if there was one."""
        lock = getattr(self, '_voice_lock', None)
        if lock is None:
            return False
        with lock:
            if self._voice_state not in ('command', 'deciding', 'acting'):
                return False
            self._voice_state = 'idle'
            self._voice_generation += 1
            if self._voice_listener is not None:
                self._voice_listener.cancel_capture()
        log(f"[VOICE] Voice command cancelled: {reason}")
        return True

    def _voice_preempt(self):
        """A manual recording start wins over a pending voice command."""
        self._voice_cancel_command("a recording was started manually")

    def _voice_trim_seconds(self, detected_at):
        """Audio to cut so the end word is not transcribed: the word plus time since it fired."""
        trim = self._get_float_setting('voice_control_end_trim_seconds', 1.0)
        return max(0.0, trim) + max(0.0, time.monotonic() - detected_at)

    def _voice_stop_recording(self, detected_at):
        """The end word stops any running recording, the way a stop command does."""
        if self.config.get_setting('recording_mode', 'toggle') == 'continuous':
            self._continuous_stop_and_wait()
        log("[VOICE] End word - stopping recording")
        self._stop_recording(trim_tail_seconds=self._voice_trim_seconds(detected_at))

    def _voice_on_phrase(self, phrase):
        """A command phrase Vosk spotted; it only counts inside a command or dictation window."""
        with self._voice_lock:
            state = self._voice_state
        book = self._voice_phrases()
        if state == 'command':
            match = book.command.get(phrase)
            if match is not None:
                self._voice_take_phrase(match, phrase)
        elif state in ('dictation', 'dictation_deciding'):
            match = book.dictation.get(phrase)
            if match is not None:
                self._voice_dictation_resolve(match[0], said=phrase)

    def _voice_on_pause(self, detected_at):
        with self._voice_lock:
            state = self._voice_state
        if state == 'command':
            # Nothing to trim: the pause, not a word, ended the command
            self._voice_finish_command(detected_at, trim_seconds=0.0)
        elif state == 'dictation':
            self._voice_dictation_heard()

    def _voice_on_silence(self):
        with self._voice_lock:
            state = self._voice_state
        if state == 'command':
            if self._voice_cancel_command("nothing was said"):
                self._voice_cue('stop')
        elif state == 'dictation':
            self._voice_dictation_resolve('finish')  # the start word and nothing else: done

    # ------------------------ Command → decision → action ------------------------

    def _voice_command_current(self, generation):
        with self._voice_lock:
            return self._voice_state == 'deciding' and self._voice_generation == generation

    def _voice_command_acting(self, generation):
        with self._voice_lock:
            return self._voice_state == 'acting' and (generation is None or self._voice_generation == generation)

    def _voice_backend_blocked(self):
        with self._recording_lock:
            if self._model_operation_active or self._model_initializing or self._backend_init_failed:
                return "the transcription model is loading"
            if self._longform_active:
                return "a long-form session is active"
        if (MODEL_UNLOADED_FILE.exists() or getattr(self.whisper_manager, '_model_manually_unloaded', False)
                or not self.whisper_manager.is_ready()):
            return "the transcription model is not loaded"
        return None

    def _voice_local_transcriber(self):
        """True when the wake word engine (Vosk) transcribes this command instead of the backend.

        'auto' uses it when the backend can't: realtime-ws, or a model not loaded.
        """
        listener = getattr(self, '_voice_listener', None)
        if listener is None or listener.can_transcribe() is not True:
            return False
        transcriber = self.config.get_setting('voice_control_command_transcriber', 'auto')
        if transcriber == 'vosk':
            return True
        if transcriber == 'backend':
            return False
        backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
        return backend == 'realtime-ws' or self._voice_backend_blocked() is not None

    def _voice_transcribe(self, audio):
        """A command's transcript with its wake phrases stripped; None (reported) if it can't be had."""
        if self._voice_local_transcriber():
            transcript = self._voice_listener.transcribe(audio)
        else:
            blocked = self._voice_backend_blocked()
            if blocked:
                log(f"[VOICE] Cannot take a command: {blocked}")
                self._notify_user("hyprwhspr", f"Voice command ignored: {blocked}", urgency="normal")
                return None
            transcript = self.whisper_manager.transcribe_audio(audio, sample_rate=SAMPLE_RATE)
        return strip_wake_phrase((transcript or '').strip(), self._voice_start_phrase, self._voice_end_phrase)

    def _run_voice_command(self, generation, audio):
        try:
            text = self._voice_transcribe(audio)
            if text is None or not self._voice_command_current(generation):
                return
            if not text or is_hallucination(text, self.config.get_hallucination_markers()):
                log("[VOICE] No command heard")
                self.audio_manager.play_error_sound()
                return
            log(f"[VOICE] Command: {text!r}")
            match = self._voice_phrases().match_command(text)
            if match is None:
                try:
                    decision = voice_command.decide(text, self.config.get_setting)
                except DecisionError as e:
                    log(f"[VOICE] Decision failed: {e}")
                    self.audio_manager.play_error_sound()
                    self._notify_user("hyprwhspr", f"Voice command failed: {e}", urgency="normal")
                    return
            # 'acting' until the action has started: a stale decision can no
            # longer act, and the start word can't open a second command
            # before a voice dictation is recording.
            with self._voice_lock:
                if self._voice_state != 'deciding' or self._voice_generation != generation:
                    return
                self._voice_state = 'acting'
            if match is not None:
                self._voice_execute_phrase(match, text, generation)
            else:
                self._voice_execute(decision, text, generation)
        except Exception as e:
            log(f"[VOICE] Voice command failed: {e}")
        finally:
            self._voice_settle(generation)

    def _voice_settle(self, generation):
        with self._voice_lock:
            if self._voice_generation == generation and self._voice_state in ('deciding', 'acting'):
                self._voice_state = 'idle'

    def _voice_take_phrase(self, match, phrase):
        """Run a spotted command phrase now: no transcript, no decision model."""
        with self._voice_lock:
            if self._voice_state != 'command':
                return
            self._voice_state = 'acting'
            generation = self._voice_generation
            self._voice_listener.cancel_capture()
        self._voice_cue('stop')
        threading.Thread(target=self._run_voice_phrase, args=(generation, match, phrase),
                         daemon=True, name="VoicePhrase").start()

    def _run_voice_phrase(self, generation, match, phrase):
        try:
            self._voice_execute_phrase(match, phrase, generation)
        except Exception as e:
            log(f"[VOICE] Voice command failed: {e}")
        finally:
            self._voice_settle(generation)

    def _voice_execute_phrase(self, match, text, generation=None):
        """A command phrase's action (voice_phrases.COMMAND_ACTIONS)."""
        action, language = match
        log(f"[VOICE] Phrase: {action}" + (f" ({language})" if language else ""))
        if action == 'cancel':
            self._voice_announce('Cancelled')
            return
        if action == 'help':
            self._voice_show_help()
            return
        if action in DRAFT_INSTRUCTIONS:
            target = language_name(language) if language else llm_refiner.translate_language(self.config.get_setting)
            self._voice_edit_draft(DRAFT_INSTRUCTIONS[action].format(language=target))
            return
        self._voice_execute(VoiceDecision(action, 1.0, target_language=language), text, generation)

    def _voice_execute(self, decision, text, generation=None):
        log(f"[VOICE] Decision: {decision.action} ({decision.confidence:.2f})"
            + (f", speak {decision.spoken_language}" if decision.spoken_language else "")
            + (f", into {decision.target_language}" if decision.target_language else ""))
        plan = plan_for(decision)
        if plan is not None:
            self._voice_announce_dictation(decision.action, plan)
            self._start_recording(language_override=decision.spoken_language, route=plan.route,
                                  voice_plan=plan)
            return
        if decision.action in ('paste_last', 'copy_last'):
            ok, message = self._handle_recovery(decision.action)
        elif decision.action in SCRATCHPAD_ACTIONS:
            ok, message = self._handle_scratchpad_request(decision.action)
        elif decision.action == 'scratchpad_edit':
            self._voice_edit_draft(text)
            return
        elif decision.action == 'help':
            self._voice_show_help()
            return
        elif decision.action == 'ask':
            # Reports its own failures; a cancel while the LLM runs drops the reply
            self._assistant_respond(text, still_wanted=lambda: self._voice_command_acting(generation))
            return
        else:
            if decision.uncertain:
                self.audio_manager.play_error_sound()
                self._notify_user("hyprwhspr", f"Didn't understand: {text}", urgency="low")
            return
        if not ok:
            self._notify_user("hyprwhspr", message, urgency="normal")

    def _voice_edit_draft(self, instruction):
        """Apply a spoken instruction to the scratchpad draft; the panel shows the change."""
        if not self._scratchpad.read().strip():
            self.audio_manager.play_error_sound()
            self._notify_user("hyprwhspr scratchpad", "The draft is empty", urgency="normal")
            return
        log("[VOICE] Editing the draft")
        self._scratchpad_apply(instruction)

    def _voice_show_help(self):
        self._notify_user("hyprwhspr voice", self._voice_phrases().help_text(self._voice_start_phrase),
                          urgency="normal")

    def _voice_cue(self, kind):
        if self.config.get_setting('voice_control_cues', True):
            self.audio_manager.play_cue_sound(kind)

    def _voice_announce(self, message):
        """A short notification of what a voice command did (voice_control_announce)."""
        if self.config.get_setting('voice_control_announce', True):
            self._notify_user("hyprwhspr voice", message, urgency="low")

    def _voice_announce_dictation(self, action, plan):
        label = _DICTATION_LABELS.get(action, 'Dictating')
        if plan.route == 'translate':
            target = plan.translate_language or llm_refiner.translate_language(self.config.get_setting)
            label = f'Dictating, translated into {target}'
        if self.config.get_setting('voice_control_in_dictation', True) and self._voice_start_phrase:
            label += f' - say "{self._voice_start_phrase}" when you\'re done'
        self._voice_announce(label)

    # ------------------------ The start word inside a dictation ------------------------

    def _voice_dictation_window_allowed(self):
        """The start word in this recording opens a window for a dictation phrase."""
        if not self.config.get_setting('voice_control_in_dictation', True):
            return False
        if self.config.get_setting('recording_mode', 'toggle') == 'continuous':
            return False  # segments are delivered as they come; there is no whole dictation to steer
        if getattr(self, '_live_typing', None) is not None:
            return False  # the words are already typed
        with self._recording_lock:
            return (self.is_recording and not self.is_processing and not self._recording_starting
                    and not self._longform_active)

    def _voice_open_dictation_window(self, detected_at):
        with self._voice_lock:
            if self._voice_state != 'idle':
                return
            self._voice_state = 'dictation'
            self._voice_generation += 1
            self._voice_window_at = detected_at
            pause = self._voice_pause_seconds()
            self._voice_listener.begin_capture(since=detected_at, pause_seconds=pause, silence_seconds=pause)
        log("[VOICE] Start word in a dictation - listening for a dictation command")
        self._voice_cue('start')

    def _voice_dictation_heard(self):
        """Speech followed the start word but no phrase was spotted: transcribe and match it."""
        with self._voice_lock:
            if self._voice_state != 'dictation':
                return
            self._voice_state = 'dictation_deciding'
            generation = self._voice_generation
            audio = self._voice_listener.end_capture()
        threading.Thread(target=self._run_dictation_phrase, args=(generation, audio),
                         daemon=True, name="VoiceDictationCommand").start()

    def _run_dictation_phrase(self, generation, audio):
        said = None
        try:
            said = self._voice_transcribe_dictation_window(audio)
        except Exception as e:
            log(f"[VOICE] Cannot transcribe the dictation command: {e}")
        with self._voice_lock:
            if self._voice_state != 'dictation_deciding' or self._voice_generation != generation:
                return
        match = self._voice_phrases().match_dictation(said) if said else None
        if match is not None:
            self._voice_dictation_resolve(match[0], said=said)
        else:
            # The start word was said by accident, or as part of the text: carry on
            log(f"[VOICE] No dictation command in {said!r}; still recording")
            self._voice_dictation_resolve('continue', said=said, quiet=True)

    def _voice_transcribe_dictation_window(self, audio):
        if audio is None or not len(audio):
            return ''
        listener = self._voice_listener
        if listener.can_transcribe() is True:
            text = listener.transcribe(audio)
        else:
            backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
            if backend == 'realtime-ws' or self._voice_backend_blocked():
                return ''  # the recording owns the realtime stream
            text = self.whisper_manager.transcribe_audio(audio, sample_rate=SAMPLE_RATE)
        return strip_wake_phrase((text or '').strip(), self._voice_start_phrase, self._voice_end_phrase)

    def _voice_close_dictation_window(self):
        """Close an open dictation window; its start time, or None if none was open."""
        lock = getattr(self, '_voice_lock', None)
        if lock is None:
            return None
        with lock:
            if self._voice_state not in ('dictation', 'dictation_deciding'):
                return None
            self._voice_state = 'idle'
            self._voice_generation += 1
            if self._voice_listener is not None:
                self._voice_listener.cancel_capture()
            return getattr(self, '_voice_window_at', None) or time.monotonic()

    def _voice_dictation_resolve(self, action, said=None, quiet=False):
        """Do what a dictation phrase asks; False when no dictation window was open."""
        opened_at = self._voice_close_dictation_window()
        if opened_at is None:
            return False
        log(f"[VOICE] Dictation command: {action}")
        if not quiet and action in _DICTATION_ANNOUNCEMENTS:
            self._voice_announce(_DICTATION_ANNOUNCEMENTS[action])
        if action == 'continue':
            # The words were recorded: take them out of the transcript instead
            phrases = [self._voice_start_phrase]
            if said:
                phrases.insert(0, f'{self._voice_start_phrase} {said}')
            with self._recording_lock:
                self._voice_strip_phrases = list(getattr(self, '_voice_strip_phrases', None) or []) + phrases
            self._voice_cue('stop')
            return True
        if action == 'cancel':
            threading.Thread(target=self._cancel_recording, daemon=True, name="VoiceCancel").start()
            return True
        with self._recording_lock:
            if not self.is_recording:
                return True
            if action in _DICTATION_ROUTES:
                self._current_route = _DICTATION_ROUTES[action]
            elif action in ('raw', 'submit'):
                plan = getattr(self, '_voice_plan', None) or DictationPlan(route=self._current_route)
                self._voice_plan = dataclasses.replace(plan, **{'raw' if action == 'raw' else 'submit': True})
        threading.Thread(target=self._voice_stop_dictation, args=(opened_at,),
                         daemon=True, name="VoiceStop").start()
        return True

    def _voice_stop_dictation(self, opened_at):
        """Stop the recording, cutting everything from the start word on."""
        log("[VOICE] Start word - finishing the dictation")
        self._stop_recording(trim_tail_seconds=self._voice_trim_seconds(opened_at),
                             tail_phrase=self._voice_start_phrase)

    def _voice_recording_ended(self):
        """A recording stopped some other way: close its dictation window, drop the start word from it."""
        if self._voice_close_dictation_window() is not None and self._voice_start_phrase:
            with self._recording_lock:
                self._voice_strip_phrases = (list(getattr(self, '_voice_strip_phrases', None) or [])
                                             + [self._voice_start_phrase])

    # ------------------------ Voice-started dictation delivery ------------------------

    @staticmethod
    def _voice_delivery_options(plan):
        """TextInjector.inject_text options a voice plan adds to its route."""
        if plan is None:
            return {}
        options = {}
        if plan.raw:
            options['raw'] = True
        if plan.translate_language:
            options['translate_language'] = plan.translate_language
        return options
