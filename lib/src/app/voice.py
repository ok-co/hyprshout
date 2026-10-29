"""Voice control: wake words, quiet command sessions and voice-started dictations.

The start word opens a quiet command session: the wake word stream itself
keeps the audio (no recorder, OSD, ducking or paste) until the end word. The
command is transcribed with the loaded backend and a decision model picks one
action (voice_command.ACTIONS). Dictation actions start a normal recording on
the plan's route (paste, translate, scratchpad, assistant), which the end word
stops like any other recording; scratchpad, last-dictation and 'ask' actions
run at once.

Manual control always wins: a hotkey or CLI start cancels a pending command,
and the cancel shortcut or command cancels it too.
"""

import threading
import time

from paths import MODEL_UNLOADED_FILE
from service_log import log
from backend_utils import normalize_backend
from hallucination import is_hallucination
import voice_command
from voice_command import SCRATCHPAD_ACTIONS, DecisionError, phrase_for, plan_for, strip_wake_phrase
from wake_word import SAMPLE_RATE, WakeWordError, WakeWordListener, normalize_word

DEFAULT_START_WORD = 'hey_jarvis'
DEFAULT_END_WORD = 'hey_mycroft'


class VoiceControlMixin:
    """Voice control: wake words, quiet command sessions and voice-started dictations."""

    # ------------------------ Lifecycle ------------------------

    def _start_voice_control(self):
        """Start the wake word listener when voice_control_enabled; False if it can't run."""
        if not self.config.get_setting('voice_control_enabled', False):
            return False
        reason = self._voice_control_unsupported()
        if reason:
            log(f"[VOICE] Voice control disabled: {reason}")
            self._notify_user("hyprwhspr", f"Voice control disabled: {reason}", urgency="normal")
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
        self._voice_listener = WakeWordListener(
            start_word, end_word, self._on_wake_event,
            threshold=self._get_float_setting('voice_control_threshold', 0.5),
            device=self.audio_capture.device_id,
            max_capture_seconds=self._get_float_setting('voice_control_max_command_seconds', 15.0),
        )
        return self._voice_listener.start()

    def _voice_control_unsupported(self):
        if self.config.get_setting('recording_mode', 'toggle') == 'long_form':
            return 'not available in long_form recording mode'
        backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
        if backend == 'realtime-ws':
            return 'the realtime-ws backend cannot transcribe voice commands; use a local or REST backend'
        if not self.audio_capture.input_is_shared():
            return 'the microphone is raw ALSA; the wake word stream needs PipeWire or PulseAudio'
        return None

    def _stop_voice_control(self):
        listener = getattr(self, '_voice_listener', None)
        if listener is not None:
            listener.stop()

    # ------------------------ Wake word events ------------------------

    def _on_wake_event(self, kind, detected_at):
        """Listener callback (its worker thread): decide quickly, hand slow work to a thread."""
        if kind == 'start':
            self._voice_begin_command()
        elif kind == 'end':
            if self._voice_finish_command(detected_at):
                return
            if self.is_recording:
                threading.Thread(target=self._voice_stop_recording, args=(detected_at,),
                                 daemon=True, name="VoiceStop").start()
        elif kind == 'capture_limit':
            if self._voice_cancel_command("too long - say the end word to finish a command"):
                self.audio_manager.play_error_sound()

    def _voice_begin_command(self):
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
            self._voice_listener.begin_capture()
        # The command must not be transcribed over the draft or a reply being read aloud
        self._stop_scratchpad_speech()
        self._stop_assistant_speech()
        log("[VOICE] Listening for a command")
        self._voice_cue('start')

    def _voice_finish_command(self, detected_at):
        """End a command capture and decide on it; False when no command was open."""
        with self._voice_lock:
            if self._voice_state != 'command':
                return False
            self._voice_state = 'deciding'
            generation = self._voice_generation
            audio = self._voice_listener.end_capture(self._voice_trim_seconds(detected_at))
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
            if self._voice_state == 'idle':
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

    def _run_voice_command(self, generation, audio):
        try:
            blocked = self._voice_backend_blocked()
            if blocked:
                log(f"[VOICE] Cannot take a command: {blocked}")
                self._notify_user("hyprwhspr", f"Voice command ignored: {blocked}", urgency="normal")
                return
            transcript = self.whisper_manager.transcribe_audio(audio, sample_rate=SAMPLE_RATE)
            if not self._voice_command_current(generation):
                return
            text = strip_wake_phrase((transcript or '').strip(),
                                     self._voice_start_phrase, self._voice_end_phrase)
            if not text or is_hallucination(text, self.config.get_hallucination_markers()):
                log("[VOICE] No command heard")
                self.audio_manager.play_error_sound()
                return
            log(f"[VOICE] Command: {text!r}")
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
            self._voice_execute(decision, text, generation)
        except Exception as e:
            log(f"[VOICE] Voice command failed: {e}")
        finally:
            with self._voice_lock:
                if self._voice_generation == generation and self._voice_state in ('deciding', 'acting'):
                    self._voice_state = 'idle'

    def _voice_execute(self, decision, text, generation=None):
        log(f"[VOICE] Decision: {decision.action} ({decision.confidence:.2f})"
            + (f", speak {decision.spoken_language}" if decision.spoken_language else "")
            + (f", into {decision.target_language}" if decision.target_language else ""))
        plan = plan_for(decision)
        if plan is not None:
            self._start_recording(language_override=decision.spoken_language, route=plan.route,
                                  voice_plan=plan)
            return
        if decision.action in ('paste_last', 'copy_last'):
            ok, message = self._handle_recovery(decision.action)
        elif decision.action in SCRATCHPAD_ACTIONS:
            ok, message = self._handle_scratchpad_request(decision.action)
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

    def _voice_cue(self, kind):
        if self.config.get_setting('voice_control_cues', True):
            self.audio_manager.play_cue_sound(kind)

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
