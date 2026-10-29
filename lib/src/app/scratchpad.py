"""Scratchpad: dictations that edit a draft, pasted (or read aloud) when it is ready."""

import html
import threading
import time

import llm_refiner
import speech_output
from scratchpad import ScratchpadConflict
from service_log import log
from speech_output import SpeechError, SpeechPlayer
from text_injector import InjectionOutcome, preprocess_text

# Tail of the draft shown in the notification after each change.
_PREVIEW_CHARS = 280
# How long the panel shows a paste, copy or clear before it hides.
_BRIEF_SECONDS = 1.5
# Visualizer states mirrored on the panel while a scratchpad recording runs.
_PANEL_FOLLOWS = {'recording': 'listening', 'paused': 'paused', 'processing': 'transcribing'}

_KEEP = object()  # previous=_KEEP: keep the panel's current highlight
_panel_init_lock = threading.Lock()


class ScratchpadMixin:
    """Scratchpad route, its socket requests, the paste and speak shortcuts and the preview."""

    def _scratchpad_apply(self, text):
        """Apply one dictation to the draft with the LLM; never pastes.

        CONSUMED when the draft took it. FAILED leaves the draft unchanged
        and keeps the dictation for record copy-last: a raw edit instruction
        appended to the draft would corrupt it.
        """
        dictation = preprocess_text(text, self.config).strip()
        if not dictation:
            log("[SCRATCHPAD] Nothing to add (empty after preprocessing)")
            return InjectionOutcome.CONSUMED
        get_setting = self.config.get_setting
        before = self._scratchpad.read()
        self._publish_scratchpad('updating', before)
        started = time.monotonic()
        try:
            draft = self._scratchpad.apply(
                dictation, lambda current, spoken: llm_refiner.edit_draft(current, spoken, get_setting))
        except ScratchpadConflict:
            reason = 'the draft was edited while the LLM ran'
        except llm_refiner.LLMRefineError as e:
            reason = str(e)
        except Exception as e:
            reason = type(e).__name__
        else:
            log(f"[SCRATCHPAD] Draft updated in {time.monotonic() - started:.1f}s "
                f"({len(dictation)} chars dictated, draft {len(draft)} chars)")
            # Continuous mode keeps recording into the draft after each segment
            still_recording = self.is_recording and self._scratchpad_recording()
            self._show_scratchpad(draft, previous=before, state='listening' if still_recording else 'updated')
            return InjectionOutcome.CONSUMED
        log(f"[SCRATCHPAD] Edit failed, draft unchanged: {reason}")
        self.text_injector.retain_text(dictation)
        self._publish_scratchpad('error', message=reason)
        self._notify_user(
            "hyprwhspr scratchpad",
            f"Edit failed ({reason}); the draft is unchanged. "
            "Recover the dictation with hyprwhspr record copy-last",
        )
        return InjectionOutcome.FAILED

    def _handle_scratchpad_request(self, verb, voice=None):
        """Serve a scratchpad_* socket request; returns (ok, message or draft).

        voice is the voice for one scratchpad_speak, or the new
        scratchpad_voice for scratchpad_voice.
        """
        pad = self._scratchpad
        if verb == 'scratchpad_show':
            return True, pad.read()
        if verb == 'scratchpad_clear':
            if not pad.clear():
                return True, 'The draft is already empty'
            if not self._publish_scratchpad('cleared', '', previous=None):
                self._show_scratchpad('')
            return True, 'Draft cleared (scratchpad undo restores it)'
        if verb == 'scratchpad_undo':
            before = pad.read()
            restored = pad.undo()
            if restored is None:
                return False, 'Nothing to undo'
            self._show_scratchpad(restored, previous=before)
            return True, 'Undone'
        if verb in ('scratchpad_copy', 'scratchpad_paste'):
            return self._scratchpad_deliver(verb[len('scratchpad_'):])
        if verb == 'scratchpad_speak':
            return self._scratchpad_speak(voice)
        if verb == 'scratchpad_stop':
            if self._stop_scratchpad_speech():
                return True, 'Stopped reading the draft'
            return True, 'Not reading the draft'
        if verb == 'scratchpad_voice':
            name = speech_output.normalize_voice(voice)
            if name is None:
                return False, f"Unknown voice; choose one of: {', '.join(speech_output.VOICES)}"
            self.config.set_setting('scratchpad_voice', name)
            log(f"[SCRATCHPAD] Voice set to {name}")
            return True, f'The draft is read in the voice {name}'
        return False, 'Unknown scratchpad request'

    def _scratchpad_deliver(self, action):
        """Copy or paste the draft; a paste clears it (scratchpad_clear_on_paste)."""
        draft = self._scratchpad.read()
        if not draft:
            return False, 'The draft is empty'
        ok, message = self.text_injector.deliver_prepared(draft, action)
        if not ok:
            return False, message
        log(f"[SCRATCHPAD] Draft {'pasted' if action == 'paste' else 'copied'} ({len(draft)} chars)")
        if (action == 'paste' and self.config.get_setting('scratchpad_clear_on_paste', True) is not False
                and self._scratchpad.clear(expected=draft)):
            self._close_scratchpad_notification()
            message += '; draft cleared (scratchpad undo restores it)'
        self._publish_scratchpad('pasted' if action == 'paste' else 'copied', draft, previous=None)
        return True, message

    def _on_scratchpad_paste_shortcut(self):
        """Paste key, on release: the held modifiers would otherwise join the paste chord."""
        ok, message = self._scratchpad_deliver('paste')
        if not ok:
            log(f"[SCRATCHPAD] Paste not done: {message}")
            self._notify_user("hyprwhspr scratchpad", message)

    # ------------------------ Reading the draft aloud ------------------------

    def _scratchpad_speak(self, voice=None):
        """Read the draft aloud; a second request without a voice stops it (one key toggles)."""
        reader = self._scratchpad_reader()
        if voice is None and reader.speaking:
            reader.stop()
            log("[SCRATCHPAD] Stopped reading the draft")
            return True, 'Stopped reading the draft'
        if voice is not None and speech_output.normalize_voice(voice) is None:
            return False, f"Unknown voice; choose one of: {', '.join(speech_output.VOICES)}"
        draft = self._scratchpad.read()
        if not draft:
            return False, 'The draft is empty'
        name = speech_output.normalize_voice(voice) or speech_output.configured_voice(self.config.get_setting)
        try:
            reader.speak(draft, name)
        except SpeechError as e:
            log(f"[SCRATCHPAD] Cannot read the draft aloud: {e}")
            return False, f'Cannot read the draft aloud: {e}'
        log(f"[SCRATCHPAD] Reading the draft aloud ({len(draft)} chars, voice {name})")
        return True, f'Reading the draft aloud ({name})'

    def _scratchpad_reader(self):
        reader = getattr(self, '_scratchpad_speech', None)
        if reader is None:
            get_setting = self.config.get_setting
            reader = SpeechPlayer(
                lambda part, voice: speech_output.synthesize(part, get_setting, voice),
                on_error=self._on_scratchpad_speech_error,
            )
            self._scratchpad_speech = reader
        return reader

    def _stop_scratchpad_speech(self):
        """Stop reading the draft (a recording would hear it); False if it wasn't being read."""
        reader = getattr(self, '_scratchpad_speech', None)
        return reader is not None and reader.stop()

    def _on_scratchpad_speech_error(self, message):
        log(f"[SCRATCHPAD] Reading the draft aloud failed: {message}")
        self._notify_user("hyprwhspr scratchpad", f"Reading the draft aloud failed: {message}")

    def _on_scratchpad_speak_shortcut(self):
        ok, message = self._scratchpad_speak()
        if not ok:
            self._notify_user("hyprwhspr scratchpad", message)

    # ------------------------ Preview ------------------------

    def _show_scratchpad(self, draft, previous=None, state='updated'):
        """Show the changed draft: on the panel, else in a notification."""
        if self._publish_scratchpad(state, draft, previous=previous):
            return
        self._notify_scratchpad(draft)

    def _notify_scratchpad(self, draft):
        """Show the draft's end in one notification, replaced on every change."""
        words = len(draft.split())
        if draft:
            title = f"Scratchpad · {words} word{'' if words == 1 else 's'}"
            body = draft if len(draft) <= _PREVIEW_CHARS else '…' + draft[-_PREVIEW_CHARS:].lstrip()
        else:
            title, body = "Scratchpad", "Empty"
        try:
            from desktop_notify import send_notification_with_id
            nid = send_notification_with_id(
                title, html.escape(body, quote=False),
                timeout_ms=self.config.get_setting('notification_timeout_ms', 5000),
                replaces_id=self._scratchpad_notification_id,
            )
            if nid is not None:
                self._scratchpad_notification_id = nid
        except Exception:
            pass  # Silently fail if no notification server

    def _close_scratchpad_notification(self):
        nid, self._scratchpad_notification_id = self._scratchpad_notification_id, None
        if nid is None:
            return
        try:
            from desktop_notify import close_notification
            close_notification(nid)
        except Exception:
            pass

    def _scratchpad_panel_runner(self):
        """The preview panel runner, created on first use; None where it can't run.

        Needs scratchpad_preview on and a layer-shell compositor; elsewhere
        (GNOME, no GTK4) the scratchpad uses notifications.
        """
        panel = getattr(self, '_scratchpad_panel', None)
        if panel is None:
            with _panel_init_lock:
                panel = getattr(self, '_scratchpad_panel', None)
                if panel is None:
                    panel = self._create_scratchpad_panel() or False
                    self._scratchpad_panel = panel
        if not panel or panel.disabled:
            return None
        return panel

    def _create_scratchpad_panel(self):
        if self.config.get_setting('scratchpad_preview', True) is False:
            return None
        try:
            from mic_osd import MicOSDRunner, ScratchpadPanelRunner
            # A running overlay already proved layer shell works here
            if not (isinstance(getattr(self, '_mic_osd_runner', None), MicOSDRunner)
                    or ScratchpadPanelRunner.is_supported()):
                log("[SCRATCHPAD] No layer-shell overlay; previews use notifications")
                return None
            return ScratchpadPanelRunner()
        except Exception as e:
            log(f"[SCRATCHPAD] Preview panel unavailable: {e}")
            return None

    def _publish_scratchpad(self, state, draft=None, previous=_KEEP, message=''):
        """Show state on the panel; False when there is no panel."""
        panel = self._scratchpad_panel_runner()
        if panel is None:
            return False
        if draft is None:
            draft = self._scratchpad.read()
        if state in ('updated', 'error'):
            seconds = self.config.get_setting('scratchpad_preview_seconds', 8)
            hold = float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) \
                and seconds > 0 else None
        elif state in ('pasted', 'copied', 'cleared'):
            hold = _BRIEF_SECONDS
        else:
            hold = None
        kwargs = {} if previous is _KEEP else {'previous': previous}
        try:
            return panel.publish(state, draft, message=message, hint=self._scratchpad_hint(), hold=hold, **kwargs)
        except Exception as e:
            log(f"[SCRATCHPAD] Preview update failed: {e}")
            return False

    def _scratchpad_hint(self):
        key = None
        if not self.config.get_setting('external_bindings', False):
            key = self.config.get_setting('scratchpad_paste_shortcut', None)
        paste = f"{key} pastes" if key else "hyprwhspr scratchpad paste"
        return f"{paste} · hyprwhspr scratchpad undo"

    def _scratchpad_recording(self):
        """The current recording session goes into the scratchpad."""
        if self.config.get_setting('recording_mode', 'toggle') == 'long_form':
            return getattr(getattr(self, '_longform', None), 'route', None) == 'scratchpad'
        return getattr(self, '_current_route', 'paste') == 'scratchpad'

    def _scratchpad_preview_follow(self, visualizer_state):
        """Mirror a scratchpad recording's progress on the panel."""
        try:
            status = _PANEL_FOLLOWS.get(visualizer_state)
            if status and self._scratchpad_recording():
                self._publish_scratchpad(status)
        except Exception:
            pass  # The preview must never break recording

    def _scratchpad_preview_settle(self):
        """The recording ended (OSD hidden): close a panel still waiting on it.

        A session that changed the draft (continuous mode) ends showing the
        result; one that never reached the LLM (cancelled, nothing heard)
        just hides.
        """
        panel = getattr(self, '_scratchpad_panel', None)
        try:
            if not panel or not panel.is_active():
                return
            last = panel.last
            if last.get('previous') is not None:
                self._publish_scratchpad('updated', last['draft'])
            else:
                panel.publish('hidden', last['draft'])
        except Exception:
            pass

    def _stop_scratchpad_preview(self):
        panel = getattr(self, '_scratchpad_panel', None)
        if panel:
            panel.stop()
