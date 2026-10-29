# Voice control

Run hyprwhspr hands-free. Say the **start word**, speak a short command, say the **end word**. A decision model
reads the command and picks what happens next:

- **Dictate**: start a dictation, pasted as usual
- **Dictate raw**: a dictation pasted exactly as spoken, without [LLM refinement](LLM_REFINEMENT.md)
- **Translate**: a dictation translated into another language before it's pasted
- **Scratchpad**: a dictation saved to a notes file instead of pasted
- **Paste last** / **copy last**: deliver the previous dictation again

A dictation started this way ends when you say the end word. The end word also stops any recording you started
with a hotkey.

> "Hey Jarvis — translate into German — Hey Mycroft" … *speak your text* … "Hey Mycroft"

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
- Any recording mode except `long_form`

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
| "take a note", "remember this" | dictation saved to the scratchpad |
| "paste that again" / "copy the last one" | the previous dictation |
| "never mind", "cancel" | nothing |

A command the model isn't sure about shows "Didn't understand" with what it heard.

**Translation** uses your [LLM refinement](LLM_REFINEMENT.md) provider and model with a translation prompt, whether
or not refinement is enabled. A translate command that names no language uses `voice_control_translation_language`.
If the translation fails, the original text is pasted.

**The scratchpad** is `~/.local/share/hyprwhspr/scratchpad.md` unless `voice_control_scratchpad_file` says
otherwise. Each note is appended under a timestamp, after the usual [text processing](TEXT_PROCESSING.md) and LLM
refinement.

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
    "voice_control_languages": ["en", "de", "fr", "es", "it", "pt", "nl", "pl"],
    "voice_control_translation_language": "en",
    "voice_control_scratchpad_file": null         // null = ~/.local/share/hyprwhspr/scratchpad.md
}
```

Words fire by accident? Raise `voice_control_threshold`. Missed? Lower it, or speak the word a little slower. A
speaker playing near the microphone can trigger them too; a headset avoids that.

The wake word model scores 80 ms of audio in a few milliseconds, a few percent of one CPU core while the service
runs. Its stream reopens on its own after suspend or an audio server restart.
