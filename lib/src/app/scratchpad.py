"""Scratchpad: dictations that edit a draft, pasted only when it is ready."""

import html
import time

import llm_refiner
from scratchpad import ScratchpadConflict
from service_log import log
from text_injector import InjectionOutcome, preprocess_text

# Tail of the draft shown in the notification after each change.
_PREVIEW_CHARS = 280


class ScratchpadMixin:
    """Scratchpad route, its socket requests and the paste shortcut."""

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

    def _handle_scratchpad_request(self, verb):
        """Serve a scratchpad_* socket request; returns (ok, message or draft)."""
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
