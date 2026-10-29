# Command line

Drive the service from scripts, launchers and [any hotkey system](HOTKEYS.md#external-hotkey-systems).

## Recording

```bash
# Start recording
hyprwhspr record start

# Start recording with specific language
hyprwhspr record start --lang it    # Italian
hyprwhspr record start --lang de    # German
hyprwhspr record start --lang es    # Spanish

# Start recording whose dictation the LLM translates (llm_translate_language)
hyprwhspr record start --translate

# Start recording whose dictation goes into the scratchpad draft instead of being pasted
hyprwhspr record start --scratchpad

# Stop recording (transcribes and pastes)
hyprwhspr record stop

# Push-to-talk key-up: stop, or latch a long hold
hyprwhspr record release

# Cancel recording (discards audio, no transcription)
hyprwhspr record cancel

# Toggle recording on/off
hyprwhspr record toggle
hyprwhspr record toggle --lang it   # Toggle with language override
hyprwhspr record toggle --translate # Toggle a translated recording
hyprwhspr record toggle --scratchpad # Toggle a recording into the scratchpad

# Check current status
hyprwhspr record status

# Capture: trigger a recording and stream the transcription to stdout
# Blocks until transcription is complete. Suppresses text injection — use for scripting/piping.
# Self-triggers a recording if none is in progress; attaches to an in-flight recording if one is.
hyprwhspr record capture
hyprwhspr record capture --lang it   # Capture with language override

# Diagnose how the running daemon would process a captured transcript
hyprwhspr record capture --trace-processing
```

## Trace processing

`--trace-processing` prints one UTF-8 JSON document using the running daemon's settings. It reports exact `raw` and
`preprocessed` text, backend/model, recording and silence settings, symbol/hook/LLM refinement state, `vad_mode`, and `boundary_mode`.
It does not run hooks, append space, save the transcript, or inject text.

`vad_mode`: `none`; `silero_filter` (pre-inference filtering); `silero_segmented` (ONNX segmentation after the reported
duration gate); `server_vad`; `manual_commit`; or `provider_managed`. `boundary_mode` is `manual_stop`,
`silence_auto_stop`, or `continuous_silence`. Continuous silence flushing and realtime server VAD can turn pauses into
independently punctuated segments. Trace does not change VAD or silence defaults.

## LLM refinement

```bash
hyprwhspr config llm               # setup: provider, API key, model
hyprwhspr config llm status
hyprwhspr config llm test "text"   # or pipe text in
hyprwhspr config llm prompt        # write your own prompt
hyprwhspr config llm disable
hyprwhspr config llm forget-key
```

See [LLM refinement](LLM_REFINEMENT.md).

## Scratchpad

```bash
hyprwhspr scratchpad               # print the draft (scratchpad show)
hyprwhspr scratchpad paste         # paste it into the focused app, without auto-submit
hyprwhspr scratchpad copy          # copy it to the clipboard
hyprwhspr scratchpad undo          # undo the last change
hyprwhspr scratchpad clear
hyprwhspr scratchpad edit          # open it in $EDITOR
hyprwhspr scratchpad speak         # read it aloud (again: stop); speak NAME uses that voice once
hyprwhspr scratchpad stop          # stop reading
hyprwhspr scratchpad voice [NAME]  # list the voices, or pick the one it's read in
hyprwhspr scratchpad status
hyprwhspr config llm test --scratchpad "text"   # try an edit on a copy of the draft
```

See [Scratchpad](SCRATCHPAD.md).

## File transcription

Files in. Words out. WAV and MP3; local and REST backends.

```bash
hyprwhspr transcribe recording.mp3
hyprwhspr transcribe recording.wav -o transcript.txt
hyprwhspr transcribe recording.wav --lang fr --clean
```

Stdout by default; `-o` writes UTF-8 text. `--lang` sets the language. `--clean`
applies configured cleanup, without pasting or running hooks.

An idle service lends its loaded model. A busy service says no. No service: the
command loads the backend itself.

File transcription does not work with the realtime WebSocket backend.
