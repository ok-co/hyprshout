"""Scratchpad: dictations that edit a draft, pasted (or read aloud) when it is ready."""

import html
import time

import llm_refiner
import speech_output
from scratchpad import ScratchpadConflict
from service_log import log
from speech_output import SpeechError, SpeechPlayer
from text_injector import InjectionOutcome, preprocess_text

# Tail of the draft shown in the notification after each change.
_PREVIEW_CHARS = 280


class ScratchpadMixin:
    """Scratchpad route, its socket requests and the paste and speak shortcuts."""

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
            self._show_scratchpad(draft)
            return InjectionOutcome.CONSUMED
        log(f"[SCRATCHPAD] Edit failed, draft unchanged: {reason}")
        self.text_injector.retain_text(dictation)
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
            self._show_scratchpad('')
            return True, 'Draft cleared (scratchpad undo restores it)'
        if verb == 'scratchpad_undo':
            restored = pad.undo()
            if restored is None:
                return False, 'Nothing to undo'
            self._show_scratchpad(restored)
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

    def _show_scratchpad(self, draft):
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
