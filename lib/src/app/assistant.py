"""Voice assistant: a question from voice control answered by the LLM and read aloud.

The 'ask' voice command answers the command itself; the 'assistant' command
starts a dictation on the 'assistant' route, whose text is the question. The
reply is spoken with its own SpeechPlayer, which every recording start, the
start word and the cancel request stop, so the microphone never hears it.
"""

import html

import speech_output
import voice_assistant
from service_log import log
from speech_output import SpeechError, SpeechPlayer
from voice_assistant import AssistantError


class AssistantMixin:
    """Voice assistant: questions, spoken replies and the short conversation memory."""

    def _assistant_respond(self, question, still_wanted=None):
        """Answer question aloud; False when there is no reply to speak (the user is told why).

        still_wanted() is checked once the reply arrives: a question cancelled
        while the LLM ran is answered by nobody.
        """
        log(f"[ASSISTANT] Question: {question!r}")
        phrases = (getattr(self, '_voice_start_phrase', None), getattr(self, '_voice_end_phrase', None))
        try:
            reply = self._assistant_conversation().ask(question, self.config.get_setting, avoid_phrases=phrases)
        except AssistantError as e:
            log(f"[ASSISTANT] No answer: {e}")
            self.audio_manager.play_error_sound()
            self._notify_user("hyprwhspr assistant", f"No answer: {e}", urgency="normal")
            return False
        if still_wanted is not None and not still_wanted():
            log("[ASSISTANT] Question cancelled; reply dropped")
            return False
        log(f"[ASSISTANT] Reply ({len(reply)} chars)")
        if self.is_recording:
            # Continuous mode is still listening: a spoken reply would be transcribed
            self._show_assistant_reply(reply)
            return True
        voice = voice_assistant.configured_voice(self.config.get_setting)
        try:
            self._assistant_reader().speak(reply, voice)
        except SpeechError as e:
            log(f"[ASSISTANT] Cannot speak the reply: {e}")
            self._show_assistant_reply(reply, f"Cannot speak the reply: {e}")
            return False
        return True

    def _assistant_conversation(self):
        conversation = getattr(self, '_assistant', None)
        if conversation is None:
            conversation = self._assistant = voice_assistant.Conversation()
        return conversation

    def _assistant_reader(self):
        reader = getattr(self, '_assistant_speech', None)
        if reader is None:
            get_setting = self.config.get_setting
            reader = SpeechPlayer(
                lambda part, voice: speech_output.synthesize(part, get_setting, voice),
                on_error=self._on_assistant_speech_error, name="AssistantSpeech",
            )
            self._assistant_speech = reader
        return reader

    def _stop_assistant_speech(self):
        """Stop a reply being spoken; False if none was."""
        reader = getattr(self, '_assistant_speech', None)
        return reader is not None and reader.stop()

    def _on_assistant_speech_error(self, message):
        log(f"[ASSISTANT] Speaking the reply failed: {message}")
        self._notify_user("hyprwhspr assistant", f"Speaking the reply failed: {message}")

    def _show_assistant_reply(self, reply, problem=None):
        """The reply as a notification, when it can't be spoken."""
        body = f"{reply}\n\n({problem})" if problem else reply
        self._notify_user("hyprwhspr assistant", html.escape(body, quote=False), urgency="normal")
