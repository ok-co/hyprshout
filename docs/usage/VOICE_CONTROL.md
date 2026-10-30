# Voice control

Run hyprwhspr hands-free. Say the **start word**, speak a short command, say the **end word**. A decision model
reads the command and picks what happens next:

- **Dictate**: start a dictation, pasted as usual
- **Dictate raw**: a dictation pasted exactly as spoken, without [LLM refinement](LLM_REFINEMENT.md)
- **Translate**: a dictation [translated](LLM_REFINEMENT.md#translation) before it's pasted
- **Scratchpad**: a dictation into the [scratchpad](SCRATCHPAD.md) draft
- **Draft**: paste, copy, undo, clear or [read aloud](SCRATCHPAD.md#reading-the-draft-aloud) the scratchpad draft
- **Paste last** / **copy last**: deliver the previous dictation again
- **Ask**: a question for the [voice assistant](#voice-assistant), answered out loud

A dictation started this way ends when you say the end word. The end word also stops any recording you started
with a hotkey.

> "Hey Jarvis — translate into German — Hey Mycroft" … *speak your text* … "Hey Mycroft"

> "Hey Jarvis — how many tablespoons are in a cup? — Hey Mycroft" … *"Sixteen. Baking, or just curious?"*

## How it works

A small always-on wake word model ([openWakeWord](https://github.com/dscripka/openWakeWord)) listens on its own
microphone stream and nothing else: no audio is transcribed or sent anywhere until you say the start word.

The command session is quiet: no overlay, no media ducking, nothing pasted, only a soft ping when it opens and
closes. Its audio is transcribed by your configured backend and the text goes to a decision model on
[OpenRouter's Decisions API](https://openrouter.ai/docs/guides/community/jev). Decision models don't write text;
they pick one option from a fixed list and report how confident they are. hyprwhspr gives them only the actions
above, so a command can never run anything else.

The default model is TypeSafe's **Jev** (`typesafe/jev-1.14`). Others work too, with the same request:
`perplexity/pplx-decider-flash-v1.2` (small and inexpensive), `liquid/d1.1` or `openai/gpt-6-luna-decisions`.
Confidence scales differ between models, so retune `voice_control_min_confidence` after switching.

## Setup

1. Install openWakeWord into the service's Python. Its own requirements pin a tflite runtime that has no
   builds for current Python versions, so install the ONNX runtime it uses here and then openWakeWord itself
   without dependencies:

   ```bash
   ~/.local/share/hyprwhspr/venv/bin/python -m pip install onnxruntime scipy scikit-learn tqdm requests
   ```

   ```bash
   ~/.local/share/hyprwhspr/venv/bin/python -m pip install --no-deps openwakeword==0.6.0
   ```

2. Store an OpenRouter API key, if LLM refinement doesn't already use one:

   ```bash
   hyprwhspr config llm
   ```

3. Turn voice control on and restart the service:

   ```jsonc
   {
       "voice_control_enabled": true
   }
   ```

   ```bash
   systemctl --user restart hyprwhspr
   ```

The first start downloads the wake word models (about 5 MB) into `~/.local/share/hyprwhspr/wakeword/`. The journal
logs `[VOICE] Listening for wake words` once they're loaded.

Requirements:

- PipeWire or PulseAudio: the wake word stream shares the microphone with recordings, which raw ALSA can't do
- A local or REST [backend](../backends/OVERVIEW.md); the realtime WebSocket backend can't transcribe commands
- Any recording mode except `long_form`, which has a [pause word](RECORDING_MODES.md#pause-word) instead

## Wake words

The start and end word must be different. Pretrained words: `hey_jarvis`, `hey_mycroft`, `alexa`, `hey_rhasspy`.

```jsonc
{
    "voice_control_start_word": "hey_jarvis",
    "voice_control_end_word": "hey_mycroft"
}
```

A path to a custom `.onnx` model works too. [openWakeWord](https://github.com/dscripka/openWakeWord) can train one
for any phrase from synthetic speech.

The word that ends a dictation is cut from its audio, and any part the recognizer still heard is removed from the
text. If the last word you dictated goes missing, lower `voice_control_end_trim_seconds`; if bits of the end word
stay in, raise it.

## Commands

Say what you want in your own words; these are examples.

| Command | Action |
| --- | --- |
| "dictate", "take a message", "write an email" | dictation, pasted |
| "dictate in German" | dictation in that language (it must be in `voice_control_languages`) |
| "type exactly what I say", "raw" | dictation without LLM refinement |
| "translate into French", "in Spanish please" | dictation translated before pasting |
| "take a note", "add to the draft", "edit the draft" | dictation into the scratchpad draft |
| "send the draft" / "copy the draft" | the draft, pasted or copied |
| "undo that" / "clear the draft" | the draft's last change undone, or the draft cleared |
| "read me the draft", "what does the draft say" | the draft read aloud in `scratchpad_voice` |
| "paste that again" / "copy the last one" | the previous dictation |
| "what's a good name for a cat", "explain DNS in one sentence" | the question, answered aloud |
| "Jarvis, I need you", "I have a question" | what you say next, answered aloud |
| "never mind", "cancel" | nothing |

A command the model isn't sure about shows "Didn't understand" with what it heard.

**Translation** works like the [translate shortcut](LLM_REFINEMENT.md#translation): the LLM translates the
dictation, whether or not refinement is enabled, and the original is pasted if it fails. A command that names a
language ("translate into French") translates into that one; otherwise `llm_translate_language` applies.

**Scratchpad** dictations edit the [draft](SCRATCHPAD.md) exactly like the scratchpad shortcut: new content is
added, an instruction about the draft changes it, and nothing is pasted until you say "send the draft" (or use
the paste shortcut). Draft actions report failures, such as an empty draft, in a notification.

## Voice assistant

Ask a question instead of giving a command and the assistant answers it out loud. A command that is itself a
question is answered at once. One that only calls the assistant ("I need you", "are you there") starts a
recording: say your question, then the end word.

> "Hey Jarvis — I need you — Hey Mycroft" … *ask your question* … "Hey Mycroft"

The question goes to the [LLM refinement](LLM_REFINEMENT.md) provider and model (`assistant_model` overrides
the model), and the reply is read aloud in `assistant_voice` with the
[scratchpad's speech model](SCRATCHPAD.md#reading-the-draft-aloud). It needs the same OpenRouter key and an MP3
player (mpv, ffplay, pw-play or paplay), but not `llm_refine_enabled`.

The built-in personality is Jarvis: helpful first, brief, and dryly funny, like a well-read butler who finds you
mildly amusing. It keeps the jokes out of serious questions, answers in the language you asked in, and writes
for the ear (no lists or Markdown). It knows the current date and time, but it can't browse the web, see your
screen or act on the computer, and says so. Replace the personality with `assistant_prompt`, or keep a longer
prompt in a file with `assistant_prompt_file`, read on every question:

```jsonc
{
    "assistant_prompt_file": "~/.config/hyprwhspr/assistant.md"
}
```

Any prompt gets the current time and an instruction never to say your wake words appended, so a reply can't
trigger voice control through the speakers.

Follow-up questions keep the conversation for `assistant_memory_minutes` after the last reply ("and in
Fahrenheit?"); after that, or with `0`, every question starts fresh.

The start word, any recording, and the cancel shortcut or `hyprwhspr record cancel` cut a reply off. A question
cancelled while the LLM is still thinking is never answered. When a reply can't be spoken (no player, a speech
error, or a [continuous](RECORDING_MODES.md#continuous-mode) recording still listening), it's shown in a notification instead.

## Manual control wins

- A hotkey or `hyprwhspr record start` during a command cancels the command and starts a normal recording
- The cancel shortcut or `hyprwhspr record cancel` cancels a command that's still open or being decided
- The start word is ignored while a recording runs or is being transcribed
- A command still open after `voice_control_max_command_seconds` is dropped

Voice dictations are always delivered whole after the end word, so [live typing](../backends/REALTIME_WEBSOCKET.md)
stays off for them.

## Configuration

```jsonc
{
    "voice_control_enabled": false,
    "voice_control_start_word": "hey_jarvis",
    "voice_control_end_word": "hey_mycroft",
    "voice_control_threshold": 0.5,               // wake word score that counts as heard (0-1)
    "voice_control_end_trim_seconds": 1.0,        // audio cut so the end word isn't transcribed
    "voice_control_max_command_seconds": 15,
    "voice_control_cues": true,                   // quiet pings; follows audio_feedback
    "voice_control_decision_model": "typesafe/jev-1.14",
    "voice_control_decision_endpoint": "https://openrouter.ai/api/alpha/decisions",
    "voice_control_decision_timeout": 10,         // seconds
    "voice_control_min_confidence": 0.5,          // below this: "Didn't understand"
    "voice_control_languages": ["en", "de", "fr", "es", "it", "pt", "nl", "pl"],  // languages a command can name
    "assistant_model": null,                      // null: llm_refine_model
    "assistant_prompt": null,                     // null: the built-in Jarvis
    "assistant_prompt_file": null,                // wins over assistant_prompt
    "assistant_voice": "george",                  // any scratchpad_voice name
    "assistant_timeout": 30,                      // seconds to wait for a reply
    "assistant_memory_minutes": 5                 // follow-ups keep the conversation this long; 0 = never
}
```

Words fire by accident? Raise `voice_control_threshold`. Missed? Lower it, or speak the word a little slower. A
speaker playing near the microphone can trigger them too; a headset avoids that.

The wake word model scores 80 ms of audio in a few milliseconds, a few percent of one CPU core while the service
runs. Its stream reopens on its own after suspend or an audio server restart.
