"""Long-form recording state machine and persistence orchestration."""

import threading
import time

try:
    from .hallucination import is_hallucination
    from .paths import LONGFORM_STATE_FILE
    from .segment_manager import SegmentManager
    from .text_injector import InjectionOutcome
except ImportError:
    from hallucination import is_hallucination
    from paths import LONGFORM_STATE_FILE
    from segment_manager import SegmentManager
    from text_injector import InjectionOutcome


AUTO_PAUSE_POLL_S = 0.1        # seconds between level checks
AUTO_PAUSE_KEEP_SILENCE_S = 0.5  # trailing quiet kept before a hold
AUTO_PAUSE_PREROLL_S = 1.0     # audio kept from before speech resumes a hold
PAUSE_WORD_RESUME_WINDOW_S = 0.5  # beyond the word: an auto-resume this close to it was the word


class SilenceTracker:
    """Decide when a recording goes quiet ('hold') and when speech returns ('release').

    threshold 0 calibrates from the first polls: twice their quietest level,
    the rule continuous mode uses.
    """

    CALIBRATION_POLLS = 6

    def __init__(self, pause_seconds, threshold=0.0, poll_seconds=AUTO_PAUSE_POLL_S):
        self.polls_needed = max(1, int(round(pause_seconds / poll_seconds)))
        self.threshold = threshold if threshold and threshold > 0 else None
        self._calibration = []
        self._quiet = 0

    def step(self, level, held):
        if self.threshold is None:
            self._calibration.append(level)
            if len(self._calibration) < self.CALIBRATION_POLLS:
                return None
            self.threshold = max(min(self._calibration) * 2, 2e-4)
            print(f"[LONGFORM] Auto-pause calibrated threshold={self.threshold:.5f}", flush=True)
            return None
        if level >= self.threshold:
            self._quiet = 0
            return 'release' if held else None
        if held:
            return None
        self._quiet += 1
        if self._quiet >= self.polls_needed:
            self._quiet = 0
            return 'hold'
        return None


class LongFormController:
    """Own long-form session state independently of the application shell."""

    def __init__(
        self,
        config,
        audio_capture,
        audio_manager,
        whisper_manager,
        inject_text,
        notify_capture,
        set_visualizer_state,
        show_mic_osd,
        hide_mic_osd,
        show_result_and_hide,
        write_recording_status,
        set_processing,
        hallucination_markers,
        claim_recording=None,
        release_recording=None,
        segment_manager_factory=SegmentManager,
        timer_factory=threading.Timer,
        thread_factory=threading.Thread,
        clock=time.monotonic,
        state_file=LONGFORM_STATE_FILE,
    ):
        self.config = config
        self.audio_capture = audio_capture
        self.audio_manager = audio_manager
        self.whisper_manager = whisper_manager
        self.inject_text = inject_text
        self.notify_capture = notify_capture
        self.set_visualizer_state = set_visualizer_state
        self.show_mic_osd = show_mic_osd
        self.hide_mic_osd = hide_mic_osd
        self.show_result_and_hide = show_result_and_hide
        self.write_recording_status = write_recording_status
        self.set_processing = set_processing
        self.claim_recording = claim_recording or (lambda: True)
        self.release_recording = release_recording or (lambda: None)
        self.hallucination_markers = hallucination_markers
        self.segment_manager_factory = segment_manager_factory
        self.timer_factory = timer_factory
        self.thread_factory = thread_factory
        self.clock = clock
        self.state_file = state_file

        self.state = 'IDLE'
        self.language_override = None
        self.route = 'paste'
        self.lock = threading.Lock()
        self.segment_manager = None
        self.auto_save_timer = None
        self.error_audio = None

        # Auto-pause: a monitor holds the capture buffer through silences while
        # the session stays RECORDING. The generation retires a stale monitor.
        self.auto_paused = False
        self._auto_pause_generation = 0
        self._auto_pause_stop = None
        self._auto_pause_threshold = None  # calibrated once per session
        self._auto_resumed_at = None
        self._auto_resume_preroll = 0.0

    def ensure_initialized(self):
        if self.segment_manager is not None:
            return
        max_size_mb = self.config.get_setting('long_form_temp_limit_mb', 500)
        self.segment_manager = self.segment_manager_factory(max_size_mb=max_size_mb)
        print("[LONGFORM] Segment manager initialized (lazy init)", flush=True)
        self.cleanup_temp_on_startup()

    def primary_shortcut(self, route='paste'):
        """Start, pause or resume; route applies only to a session it starts."""
        self.ensure_initialized()
        with self.lock:
            if self.state == 'IDLE':
                self.start_recording(route=route)
            elif self.state == 'RECORDING':
                self.pause_recording()
            elif self.state == 'PAUSED':
                self.resume_recording()
            elif self.state in ('PROCESSING', 'ERROR'):
                print(f"[LONGFORM] Ignoring shortcut in {self.state} state")

    def cancel_shortcut(self):
        self.ensure_initialized()
        with self.lock:
            self.cancel()

    def submit_shortcut(self):
        self.ensure_initialized()
        with self.lock:
            if self.state in ('RECORDING', 'PAUSED'):
                if self.state == 'RECORDING':
                    self.stop_auto_pause_monitor()
                    audio_data = self.audio_capture.pause_recording()
                    pending_audio = None
                    if audio_data is not None and len(audio_data) > 0:
                        if self.segment_manager.save_segment(audio_data) is None:
                            pending_audio = self.segment_manager.concatenate_readable(audio_data)
                    self.submit(audio_data=pending_audio)
                else:
                    self.submit()
            elif self.state == 'ERROR':
                print("[LONGFORM] Retrying submission")
                self.submit(retry=True)
            elif self.state == 'IDLE':
                print("[LONGFORM] Nothing to submit (IDLE state)")
            elif self.state == 'PROCESSING':
                print("[LONGFORM] Already processing, please wait")

    def request_start(self, language_override=None, route='paste'):
        """Handle an external start command using long-form state semantics."""
        self.ensure_initialized()
        lang_info = f" (language: {language_override})" if language_override else ""
        with self.lock:
            if self.state == 'IDLE':
                print(f"[CONTROL] Long-form start requested (immediate){lang_info}", flush=True)
                self.start_recording(language_override=language_override, route=route)
            elif self.state == 'PAUSED':
                print(f"[CONTROL] Long-form resume requested (immediate){lang_info}", flush=True)
                self.resume_recording()
            elif self.state == 'RECORDING':
                print("[CONTROL] Long-form already recording, ignoring start request", flush=True)
            else:
                print(f"[CONTROL] Long-form in {self.state} state, ignoring start request", flush=True)

    def request_pause(self):
        """Handle an external stop command as a long-form pause request."""
        self.ensure_initialized()
        with self.lock:
            if self.state == 'RECORDING':
                print("[CONTROL] Long-form pause requested (immediate)", flush=True)
                self.pause_recording()
            elif self.state == 'PAUSED':
                print("[CONTROL] Long-form already paused, ignoring stop request", flush=True)
            elif self.state == 'IDLE':
                print("[CONTROL] Long-form not recording, ignoring stop request", flush=True)
            else:
                print(f"[CONTROL] Long-form in {self.state} state, ignoring stop request", flush=True)

    def request_cancel(self):
        """Handle an external cancel command, including failed sessions."""
        self.ensure_initialized()
        with self.lock:
            if self.state in ('RECORDING', 'PAUSED', 'ERROR'):
                print("[CONTROL] Long-form cancel requested (immediate)", flush=True)
                self.cancel()
            else:
                print(f"[CONTROL] Long-form in {self.state} state, ignoring cancel request", flush=True)

    def pause_word(self, detected_at, trim_seconds=1.0):
        """The pause word: pause a recording session, or resume a paused one.

        The word's own audio (trim_seconds plus the time since it fired) is
        cut from the segment it ends.
        """
        self.ensure_initialized()
        with self.lock:
            if self.state == 'RECORDING':
                print("[LONGFORM] Pause word - pausing", flush=True)
                self.pause_recording(trim_tail_seconds=self._pause_word_trim(detected_at, trim_seconds))
            elif self.state == 'PAUSED':
                print("[LONGFORM] Pause word - resuming", flush=True)
                self.resume_recording()
            else:
                print(f"[LONGFORM] Pause word ignored in {self.state} state", flush=True)

    def _pause_word_trim(self, detected_at, trim_seconds):
        if self.auto_paused:
            return 0.0  # the word only reached the pre-roll, which a pause drops
        now = self.clock()
        trim = max(0.0, trim_seconds) + max(0.0, now - detected_at)
        resumed_at = self._auto_resumed_at
        if resumed_at is not None and detected_at - resumed_at <= trim_seconds + PAUSE_WORD_RESUME_WINDOW_S:
            # The word itself resumed an auto-pause: drop all of it
            trim = max(trim, self._auto_resume_preroll + (now - resumed_at))
        return trim

    def _abandon_claim(self):
        """Drop a claim whose work failed, and free anyone waiting on it."""
        self.release_recording()
        self.notify_capture("", final=True)

    def start_recording(self, language_override=None, route='paste'):
        # The backend is shared with file transcription and model loads, so a
        # session has to be claimed before any audio is captured for it
        if not self.claim_recording():
            print("[LONGFORM] Cannot start: file transcription or model load in progress", flush=True)
            # A capture client may have self-triggered this start and is now
            # waiting for a session that will never run
            self.notify_capture("", final=True)
            return

        lang_info = f" (language: {language_override})" if language_override else ""
        print(f"[LONGFORM] Starting recording session{lang_info}{f' ({route})' if route != 'paste' else ''}")
        self.language_override = language_override
        self.route = route

        started = False
        try:
            started = self.audio_capture.start_recording()
            if started:
                self.segment_manager.start_session()
        except Exception:
            # No session exists to own either, so neither the stream nor the
            # claim may outlive this call
            if started:
                try:
                    self.audio_capture.stop_recording()
                except Exception:
                    pass
            self._abandon_claim()
            raise

        if not started:
            print("[LONGFORM] Failed to start audio capture")
            self._abandon_claim()
            return

        self._auto_pause_threshold = None
        self._set_state('RECORDING', visualizer='recording')
        self.show_mic_osd()
        self.start_auto_save_timer()
        self.start_auto_pause_monitor()
        self.audio_manager.play_start_sound()

    def pause_recording(self, trim_tail_seconds=0.0):
        print("[LONGFORM] Pausing recording")
        self.stop_auto_save_timer()
        self.stop_auto_pause_monitor()
        audio_data = self.audio_capture.pause_recording()
        if audio_data is not None and trim_tail_seconds > 0:
            audio_data = audio_data[:max(0, len(audio_data) - int(trim_tail_seconds * self.audio_capture.sample_rate))]
        if audio_data is not None and len(audio_data) > 0:
            if self.segment_manager.save_segment(audio_data) is None:
                self.persistence_failed(audio_data)
                return

        self._set_state('PAUSED', visualizer='paused')
        self.audio_manager.play_stop_sound()

    def interrupt(self):
        """Pause on audio server loss, saving the buffer before recovery clears it (#282).

        Returns True when a recording session was interrupted.
        """
        with self.lock:
            if self.state != 'RECORDING':
                return False
            print("[LONGFORM] Audio server restarted - pausing session", flush=True)
            self.stop_auto_save_timer()
            self.stop_auto_pause_monitor()
            audio_data = self.audio_capture.pause_recording()
            if audio_data is not None and len(audio_data) > 0:
                if self.segment_manager.save_segment(audio_data) is None:
                    self.persistence_failed(audio_data)
                    return True
            self._set_state('PAUSED', visualizer='paused')
            self.audio_manager.play_error_sound()
            return True

    def resume_recording(self):
        print("[LONGFORM] Resuming recording")
        # PAUSED released the claim, so capturing again has to re-take it
        if not self.claim_recording():
            print("[LONGFORM] Cannot resume: file transcription or model load in progress", flush=True)
            # A capture client may have self-triggered this resume
            self.notify_capture("", final=True)
            return

        try:
            if not self.audio_capture.resume_recording():
                print("[LONGFORM] Failed to resume audio capture")
                self._set_state('ERROR', visualizer='error')
                return

            self._set_state('RECORDING', visualizer='recording')
        except Exception:
            # Capture may be unavailable (unplugged mic); the claim must not
            # outlive the failed resume
            self._abandon_claim()
            raise

        self.start_auto_save_timer()
        self.start_auto_pause_monitor()
        self.audio_manager.play_start_sound()

    def cancel(self):
        if self.state not in ('RECORDING', 'PAUSED', 'ERROR'):
            return

        print("[LONGFORM] Recording cancelled (discarded)", flush=True)
        self.notify_capture("", final=True)
        try:
            self.stop_auto_save_timer()
            self.stop_auto_pause_monitor()
            self.audio_capture.stop_recording()
            self.segment_manager.clear_session()
            self.error_audio = None
            self.language_override = None
            self.route = 'paste'
            self._set_state('IDLE')
            self.hide_mic_osd()
            self.write_recording_status(False)
            self.audio_manager.play_error_sound()
        except Exception as e:
            print(f"[ERROR] Error cancelling long-form recording: {e}", flush=True)
            try:
                self._set_state('IDLE')
                self.hide_mic_osd()
                self.write_recording_status(False)
            except Exception:
                pass

    def persistence_failed(self, audio_data):
        self.stop_auto_save_timer()
        self.stop_auto_pause_monitor()
        self.error_audio = self.segment_manager.concatenate_readable(audio_data)
        self._set_state('ERROR', visualizer='error')
        self.audio_manager.play_error_sound()

    def submit(self, retry=False, audio_data=None):
        # The caller already paused capture for a RECORDING session, but the
        # auto-save timer is still armed and must not outlive this call on
        # either the refused or the accepted path
        self.stop_auto_save_timer()
        self.stop_auto_pause_monitor()

        # Idempotent while RECORDING (already claimed); from PAUSED or ERROR it
        # re-takes the backend, which is what keeps a submit from overlapping a
        # file transcription
        if not self.claim_recording():
            print("[LONGFORM] Cannot submit: file transcription or model load in progress", flush=True)
            if self.state == 'RECORDING':
                self._set_state('PAUSED', visualizer='paused')
            self.notify_capture("", final=True)
            return

        print("[LONGFORM] Submitting for transcription")

        try:
            if audio_data is None:
                if retry and self.error_audio is not None:
                    audio_data = self.error_audio
                else:
                    audio_data = self.segment_manager.concatenate_all()

            if audio_data is None or len(audio_data) == 0:
                print("[LONGFORM] No audio data to process")
                if self.segment_manager.has_segments():
                    self.error_audio = self.segment_manager.concatenate_readable()
                    self._set_state('ERROR', visualizer='error')
                else:
                    self._set_state('IDLE')
                    self.hide_mic_osd()
                self.notify_capture("", final=True)
                return

            self._set_state('PROCESSING', visualizer='processing')
        except Exception:
            # Concatenation and the PROCESSING transition run before the
            # transcription try/except that would otherwise land in ERROR
            self._abandon_claim()
            raise

        try:
            self.set_processing(True)
            transcription = self.whisper_manager.transcribe_audio(
                audio_data,
                sample_rate=self.audio_capture.sample_rate,
                language_override=self.language_override,
            )

            if transcription and transcription.strip():
                text = transcription.strip()
                if is_hallucination(text, self.hallucination_markers):
                    print(f"[LONGFORM] Whisper hallucination detected: {text!r}")
                    self._submission_failed(audio_data)
                    return

                if self.inject_text(text, route=self.route) == InjectionOutcome.FAILED:
                    self._submission_failed(audio_data)
                    return

                self.segment_manager.clear_session()
                self.error_audio = None
                self.language_override = None
                self.route = 'paste'
                self._set_state('IDLE')
                self.show_result_and_hide(True)
            else:
                print("[LONGFORM] No transcription generated")
                self._submission_failed(audio_data)
        except Exception as e:
            print(f"[LONGFORM] Transcription error: {e}", flush=True)
            self._submission_failed(audio_data)
        finally:
            self.notify_capture("", final=True)
            self.set_processing(False)

    def _submission_failed(self, audio_data):
        self.audio_manager.play_error_sound()
        self.error_audio = audio_data
        self._set_state('ERROR', visualizer='error')

    def start_auto_save_timer(self):
        interval = self.config.get_setting('long_form_auto_save_interval', 300)
        if interval <= 0:
            return

        def auto_save_callback():
            with self.lock:
                if self.state == 'RECORDING':
                    audio_data = self.audio_capture.get_current_audio_copy()
                    if audio_data is not None and len(audio_data) > 0:
                        if self.segment_manager.save_segment(audio_data) is None:
                            frozen_audio = self.audio_capture.pause_recording()
                            if frozen_audio is None or len(frozen_audio) == 0:
                                frozen_audio = audio_data
                            self.persistence_failed(frozen_audio)
                            return
                        self.audio_capture.clear_buffer()
                        print(f"[LONGFORM] Auto-saved segment ({len(audio_data) / 16000:.1f}s)")
                    self.start_auto_save_timer()

        self.auto_save_timer = self.timer_factory(interval, auto_save_callback)
        self.auto_save_timer.daemon = True
        self.auto_save_timer.start()

    def stop_auto_save_timer(self):
        if self.auto_save_timer is not None:
            self.auto_save_timer.cancel()
            self.auto_save_timer = None

    def start_auto_pause_monitor(self):
        """Hold the buffer through silences; long_form_auto_pause_seconds 0 is off. Lock held."""
        self.stop_auto_pause_monitor()
        try:
            seconds = float(self.config.get_setting('long_form_auto_pause_seconds', 0) or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds <= 0:
            return
        self._auto_pause_generation += 1
        generation = self._auto_pause_generation
        stop_event = threading.Event()
        self._auto_pause_stop = stop_event
        thread = self.thread_factory(target=self._auto_pause_loop, args=(generation, stop_event, seconds),
                                     daemon=True, name="LongFormAutoPause")
        thread.start()

    def stop_auto_pause_monitor(self):
        """Retire the monitor. Lock held; a hold ends with the capture it belongs to."""
        self._auto_pause_generation += 1
        if self._auto_pause_stop is not None:
            self._auto_pause_stop.set()
            self._auto_pause_stop = None
        self.auto_paused = False
        self._auto_resumed_at = None

    def _auto_pause_loop(self, generation, stop_event, seconds):
        try:
            threshold = self.config.get_setting('long_form_silence_threshold', 0) or 0
            tracker = SilenceTracker(seconds, self._auto_pause_threshold or float(threshold))
            while not stop_event.wait(AUTO_PAUSE_POLL_S):
                if not self.auto_pause_step(generation, tracker, seconds):
                    return
        except Exception as e:
            print(f"[LONGFORM] Auto-pause monitor failed: {e}", flush=True)

    def auto_pause_step(self, generation, tracker, seconds):
        """One level check; False once the monitor is retired."""
        action = tracker.step(self.audio_capture.rolling_avg_level, self.auto_paused)
        with self.lock:
            if generation != self._auto_pause_generation or self.state != 'RECORDING':
                return False
            self._auto_pause_threshold = tracker.threshold
            if action == 'hold':
                trim = max(0.0, seconds - AUTO_PAUSE_KEEP_SILENCE_S)
                if self.audio_capture.hold_audio(trim_tail_seconds=trim, preroll_seconds=AUTO_PAUSE_PREROLL_S):
                    print("[LONGFORM] Silence - auto-paused", flush=True)
                    self.auto_paused = True
                    self._auto_resumed_at = None
                    self.set_visualizer_state('paused')
                    self.audio_manager.play_cue_sound('stop')
            elif action == 'release':
                self._auto_resume_preroll = self.audio_capture.release_audio()
                self._auto_resumed_at = self.clock()
                self.auto_paused = False
                print("[LONGFORM] Speech - resumed", flush=True)
                self.set_visualizer_state('recording')
                self.audio_manager.play_cue_sound('start')
        return True

    def write_state(self, state):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(state)
        except Exception as e:
            print(f"[LONGFORM] Failed to write state file: {e}", flush=True)

    def _set_state(self, state, visualizer=None):
        self.state = state
        # The claim covers only the states that hold the mic or the backend;
        # PAUSED and ERROR can persist indefinitely and must not block file
        # transcription or a model unload
        if state in ('IDLE', 'PAUSED', 'ERROR'):
            self.release_recording()
        self.write_state(state)
        if visualizer is not None:
            self.set_visualizer_state(visualizer)

    def cleanup_temp_on_startup(self):
        if self.segment_manager is None:
            return
        try:
            total_size = self.segment_manager.get_total_size()
            max_size = self.segment_manager.max_size_bytes
            if total_size > max_size:
                print(f"[LONGFORM] Temp directory over limit ({total_size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB)")
                print("[LONGFORM] Cleaning up oldest segments...")
                while self.segment_manager.cleanup_oldest():
                    if self.segment_manager.get_total_size() <= max_size:
                        break
                final_size = self.segment_manager.get_total_size()
                print(f"[LONGFORM] Cleanup complete. New size: {final_size / 1024 / 1024:.1f}MB")
            elif total_size > 0:
                print(f"[LONGFORM] Found {total_size / 1024 / 1024:.1f}MB of previous segments (limit: {max_size / 1024 / 1024:.1f}MB)")
        except Exception as e:
            print(f"[LONGFORM] Error during startup cleanup: {e}", flush=True)
