# Recording modes

How a press becomes a recording. `recording_mode` picks one.

## Toggle mode

Press to start, press again to stop. The default:

```jsonc
{
    "recording_mode": "toggle"
}
```

## Push-to-talk mode

Hold to record, release to stop:

```jsonc
{
    "recording_mode": "push_to_talk",
    "push_to_talk_lock_seconds": 3.0  // Optional: hold this long to latch hands-free. 0 (default) = off.
}
```

- Hold past `push_to_talk_lock_seconds`, let go: recording **latches on**. Press again to end it.
- Shorter holds stop on release, as always.
- External bindings latch only on key-up `release`, never on `stop` — `stop` always stops.

## Auto mode

Tap or hold; it reads your intent:

```jsonc
{
    "recording_mode": "auto"
}
```

- **Tap** (< 400ms): toggle. Tap again to stop.
- **Hold** (≥ 400ms): push-to-talk. Release to stop.

## Continuous mode

Speak; each pause pastes. Recording rolls on until you press again:

```jsonc
{
    "recording_mode": "continuous",
    "continuous_silence_seconds": 2.0,  // The pause that pastes. Default 2.0.
    "continuous_silence_threshold": 0,  // Silence level. 0 (default) = calibrate each session.
    "silence_timeout": 15               // Optional: 15s of quiet ends the session.
}
```

- The final press pastes what's left.
- Detection off? The log prints the calibrated level; start `continuous_silence_threshold` from there.

## Auto-stop on silence

Go quiet and recording ends on its own — in toggle, auto, or continuous mode:

```jsonc
{
    "silence_timeout": 2.5  // Seconds of quiet. 0 (default) = off.
}
```

- Arms only once you speak; a slow first sentence is safe.
- Hears silence the way continuous mode does, same threshold.
- In continuous mode, choose a longer quiet (say `15`): pauses still paste, the long silence ends it.
- The stop beep marks it. Press to stop sooner.

## Long-form mode

Extended recording with pause/resume support:

```jsonc
{
    "recording_mode": "long_form",
    "long_form_submit_shortcut": "SUPER+ALT+E",  // Required: no default, must be set
    "long_form_temp_limit_mb": 500,              // Optional: max temp storage (default: 500 MB)
    "long_form_auto_save_interval": 300,         // Optional: auto-save interval in seconds (default: 300 = 5 minutes)
    "external_bindings": false                   // Optional: set true when compositor bindings drive recording
}
```

- Primary shortcut toggles recording/pause/resume
- Submit shortcut processes all recorded segments and pastes transcription
- Segments are auto-saved periodically to disk for crash recovery
- Old segments are automatically cleaned up when storage limit is reached

### Auto-pause on silence

Stop keeping audio while you're quiet, so long sessions stay small and transcribe faster:

```jsonc
{
    "long_form_auto_pause_seconds": 3,  // Quiet this long pauses. 0 (default) = off.
    "long_form_silence_threshold": 0    // Silence level. 0 (default) = follow the room.
}
```

- After the quiet, the indicator switches to paused and a soft ping plays; the silence is dropped.
- Speak again and recording resumes on its own, with a soft ping. The second before you spoke is kept, so the
  first word isn't cut.
- The session stays open: the shortcut still pauses it for real, and submit sends everything recorded.
- Silence is judged against your room: three times the quietest level of the last few seconds, so it adapts
  to background noise on its own.
- The journal shows each step: `Auto-pause after 3s of silence` when recording starts, then the level and
  threshold at every pause and resume (`journalctl --user -u hyprwhspr -f`). If quiet doesn't pause, or speech
  doesn't resume, set `long_form_silence_threshold` between the levels shown there.

### Pause word

Say a word to pause, so you can talk without it being recorded. Say it again to resume:

```jsonc
{
    "long_form_pause_word": "hey_rhasspy"  // "" (default) = off
}
```

- Same as the pause shortcut: the stop sound and paused indicator, then the start sound on resume.
- Speech while paused isn't recorded or auto-resumed; the word itself is cut from the recording.
- Pretrained words are `hey_jarvis`, `hey_mycroft`, `alexa` and `hey_rhasspy`, or give the path to a custom
  `.onnx` model. Set it up like [voice control](VOICE_CONTROL.md#setup): openWakeWord installed, and PipeWire or
  PulseAudio. No API key is needed.
- `voice_control_threshold` and `voice_control_end_trim_seconds` tune it the same way they tune
  [wake words](VOICE_CONTROL.md#wake-words).
