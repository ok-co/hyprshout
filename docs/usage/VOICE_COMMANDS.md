# Voice commands

Everything you can say to hyprwhspr once [voice control](VOICE_CONTROL.md) is set up, and how to make it fit the
way you talk. Voice control has one word you need to know, the **start word** ("Hey Jarvis" by default). What it
does depends on the moment:

| When you say "Hey Jarvis" | It means |
| --- | --- |
| Nothing is recording | "Listen to my command": dictate, edit the draft, paste, ask a question |
| A dictation is recording | "About this dictation": done, cancel, into the draft, translate, keep going |
| A command is still open | "Never mind": the command is dropped |
| The draft or an answer is being read aloud | "Quiet": reading stops, and your command follows |

A command ends when you pause. The end word ("Hey Mycroft") still ends it at once, but you don't need it.

> "Hey Jarvis — send the draft" … *the draft is pasted*

> "Hey Jarvis — take a note" … *dictate* … "Hey Jarvis" … *the dictation goes into the draft*

## The recommended setup: Vosk

Any [wake word engine](VOICE_CONTROL.md#wake-words) works with everything on this page. Use
[Vosk](VOICE_CONTROL.md#other-languages-vosk) and the most common commands get faster and need no network: Vosk
listens for the command phrases themselves, right in the wake word stream. "Hey Jarvis, cancel" is recognized as
you say it, without transcribing anything or asking a model. The same model also transcribes commands that
aren't phrases when your backend can't, so voice control works with the
[realtime WebSocket backend](../backends/REALTIME_WEBSOCKET.md) or while the transcription model isn't loaded.

```jsonc
{
    "voice_control_enabled": true,
    "voice_control_engine": "vosk",
    "voice_control_vosk_model": "en-us",          // or de, pl, fr, es, ... (see Languages)
    "voice_control_start_word": "hey jarvis",
    "voice_control_end_word": "over and out"
}
```

Install Vosk into the service's Python and restart the service:

```bash
~/.local/share/hyprwhspr/venv/bin/python -m pip install vosk
```

```bash
systemctl --user restart hyprwhspr
```

With openWakeWord or sherpa-onnx, a command is transcribed by your backend first and then matched against the
same phrases, so it takes a moment longer. Commands that aren't phrases work the same with every engine.

## After the start word

### Phrases that act at once

These run as soon as they're heard, with no decision model and no API key. Say them on their own, right after
the start word, in the same breath or after a short pause. Wording around them ("okay", "please", "thanks") is
fine.

| Say | What happens |
| --- | --- |
| "dictate", "take a message" | a dictation, pasted when you finish |
| "raw", "word for word" | a dictation pasted exactly as spoken, without [LLM refinement](LLM_REFINEMENT.md) |
| "translate", "translate into German" | a dictation [translated](LLM_REFINEMENT.md#translation) before it's pasted |
| "take a note", "add to the draft" | a dictation into the [scratchpad](SCRATCHPAD.md) draft |
| "send the draft", "paste it" | the draft pasted into the focused app |
| "copy the draft" | the draft copied to the clipboard |
| "undo", "undo that" | the draft's last change undone |
| "clear the draft", "start over" | an empty draft (undo brings it back) |
| "read the draft", "read it to me" | the draft [read aloud](SCRATCHPAD.md#reading-the-draft-aloud) |
| "translate the draft", "translate the draft into Spanish" | the draft translated in place |
| "polish the draft", "clean it up" | the draft tidied into a clear prompt for a coding agent |
| "shorten the draft", "make it shorter" | the draft tightened, nothing important dropped |
| "make it a list", "bullet points" | the draft turned into a bulleted list |
| "paste again", "copy that" | the previous dictation, pasted or copied |
| "question", "I have a question" | your next sentence answered aloud by the [voice assistant](VOICE_CONTROL.md#voice-assistant) |
| "what can I say", "help" | a notification listing the phrases |
| "cancel", "never mind", "stop" | nothing |

A phrase has to be the whole command: "send the draft to Anna" isn't the "send the draft" phrase, so it goes to the
decision model like any other command.

"Translate into …" and "translate the draft into …" work for every language in `voice_control_languages`. Without
a language, `llm_translate_language` applies.

Draft edits change the draft the way a [scratchpad dictation](SCRATCHPAD.md) does: the panel shows what changed,
and "undo" or `hyprwhspr scratchpad undo` reverts it. An empty draft is reported and left alone.

### Anything else

Say it in your own words. The command is transcribed and a
[decision model](VOICE_CONTROL.md#how-it-works) picks what you meant from the same actions:

> "Hey Jarvis — turn the draft into three steps for Claude Code" … *the draft is rewritten*

> "Hey Jarvis — I'm going to dictate in German now" … *a German dictation starts*

> "Hey Jarvis — how many tablespoons are in a cup?" … *"Sixteen. Baking, or just curious?"*

An instruction about the draft ("rename the function to parse_config in the draft", "add a sentence about
tests") edits it at once. A question is answered aloud. A command the model isn't sure about shows "Didn't
understand" with what it heard.

## Inside a dictation

Say the start word while a dictation is recording, then one of these, or nothing at all:

| Say | What happens to the dictation |
| --- | --- |
| *(nothing)*, "done", "that's it" | finished and delivered |
| "Hey Jarvis" again | finished and delivered |
| "submit", "send and submit" | pasted, then Enter is pressed (even without `auto_submit`) |
| "cancel", "scrap that", "never mind" | discarded |
| "into the draft", "note it" | applied to the scratchpad draft instead of pasted |
| "translate", "translate it" | translated before it's pasted |
| "raw" | pasted exactly as spoken |
| "ask", "answer that" | sent to the voice assistant as a question |
| "keep going", "not yet" | recording goes on |

> "The login form accepts an empty password, fix the validation — Hey Jarvis — into the draft"

Everything from the start word on is cut from the dictation, so neither "Hey Jarvis" nor the command ends up in
your text. A soft ping says the start word was heard. Saying nothing finishes the dictation after
`voice_control_pause_seconds` (1.5).

**Keep going** and an accidental start word are handled the same way: if what you say after the start word isn't
a dictation phrase, recording simply continues, and the start word (with what followed it) is removed from the
transcript.

The end word still works inside a dictation and finishes it, the same as "done".

Dictation phrases don't apply:

- in [continuous mode](RECORDING_MODES.md#continuous-mode), where segments are delivered as they come; the end
  word still stops it
- while [live typing](../backends/REALTIME_WEBSOCKET.md) types words as you speak, since they're already typed
- in long-form mode, which has a [pause word](RECORDING_MODES.md#pause-word) instead
- with `"voice_control_in_dictation": false`

## How a command ends

| You | The command |
| --- | --- |
| pause after it (`voice_control_pause_seconds`, 1.5) | runs |
| say the end word | runs at once |
| say a phrase (Vosk) | runs as soon as Vosk confirms it, about half a second after you stop |
| say the start word again | is dropped |
| say nothing for `voice_control_silence_seconds` (5) | is dropped |
| keep talking past `voice_control_max_command_seconds` (15) | is dropped, with an error sound |

"Quiet" follows your room: it's measured against the background noise of the last few seconds, so a fan or
street noise doesn't keep a command open. A command that ends too soon, while you hesitate, needs a longer
`voice_control_pause_seconds`. With `"voice_control_end_on_pause": false`, only the end word runs a command, as
before.

## Hands-free: a prompt for a coding agent

1. "Hey Jarvis — take a note." Dictate the first version: *"The CSV import drops rows with quoted commas. Find
   the cause and fix it, and add a regression test."* Then "Hey Jarvis", and nothing more.
2. "Hey Jarvis — add to the draft." *"Keep the public API unchanged."* "Hey Jarvis."
3. "Hey Jarvis — read the draft." Listen to it.
4. "Hey Jarvis — mention that the parser lives in src/importer slash csv dot py." The decision model sees an
   instruction about the draft and edits it.
5. "Hey Jarvis — polish the draft." The draft becomes a tidy prompt: goal first, details as short points, paths in
   backticks.
6. Click into the agent's input. "Hey Jarvis — send the draft."

Changed your mind at any point? "Hey Jarvis — undo."

## Languages

Phrases are built in for English, German, Polish, French and Spanish:

| | Cancel | Send the draft | Done (in a dictation) |
| --- | --- | --- | --- |
| English | cancel, never mind | send the draft | done, that's it |
| Deutsch | abbrechen, vergiss es | entwurf senden | fertig |
| Polski | anuluj, nieważne | wyślij szkic | gotowe, koniec |
| Français | annuler, laisse tomber | envoie le brouillon | terminé, fini |
| Español | cancelar, olvídalo | envía el borrador | listo, terminado |

With Vosk, the phrases of the model's language are used (`voice_control_vosk_model`). Phrases with a word the
model doesn't know are left out, and the journal says how many. With other engines, a transcript can match the
phrases of any of the five languages. For another language, add your own phrases.

The decision model understands commands in any language.

## Your own phrases

`voice_control_phrases` replaces an action's built-in phrases with yours. `command` holds phrases after the start
word, `dictation` holds phrases inside a dictation. An empty list turns an action's phrases off.

```jsonc
{
    "voice_control_phrases": {
        "command": {
            "scratchpad_paste": ["ship it", "send the draft"],
            "draft_polish": ["make it a prompt"],
            "help": []
        },
        "dictation": {
            "finish": ["over", "done"],
            "submit": ["and go"]
        }
    }
}
```

Actions after the start word: `cancel`, `dictate`, `dictate_raw`, `translate`, `scratchpad`, `scratchpad_paste`,
`scratchpad_copy`, `scratchpad_undo`, `scratchpad_clear`, `scratchpad_speak`, `draft_translate`, `draft_shorten`,
`draft_polish`, `draft_list`, `paste_last`, `copy_last`, `assistant`, `help`.

Actions inside a dictation: `finish`, `submit`, `cancel`, `scratchpad`, `raw`, `translate`, `ask`, `continue`.

Good phrases are two or three words that you wouldn't say as a whole command for something else. With Vosk, write
them the way the language spells them, diacritics included, with numbers written out.

## Who transcribes a command

A command that isn't spotted as a phrase is transcribed before it's matched or sent to the decision model.
`voice_control_command_transcriber` picks what does it:

- `auto` (default): your transcription backend. With Vosk, Vosk itself when the backend can't: the realtime
  WebSocket backend, or a model that's loading or unloaded
- `backend`: always your transcription backend
- `vosk`: always Vosk. Offline, and the backend is never used for commands. Vosk's small models are less accurate
  than most backends, which matters for long free-form commands more than for phrases

Commands said inside a dictation are transcribed by Vosk when it's the engine, otherwise by the backend (never by
the realtime WebSocket backend, whose stream belongs to the dictation).

## Feedback

- **Cues**: a soft ping when the start word is heard, another when a command closes (`voice_control_cues`)
- **Announcements**: a brief notification of what happened, such as "Dictating into the draft - say "hey
  jarvis" when you're done", "Cancelled", "Into the draft" (`voice_control_announce`)
- **"What can I say"**: the phrase list, as a notification
- **Errors**: an empty draft, a failed decision or an unclear command are always reported

## Tuning

| Problem | Try |
| --- | --- |
| A command runs while I'm still thinking | raise `voice_control_pause_seconds` to 2 or 2.5 |
| A command waits too long after I stop | lower `voice_control_pause_seconds`, or say the end word |
| "Hey Jarvis" inside a dictation finishes it too soon | raise `voice_control_pause_seconds` |
| A phrase isn't recognized (Vosk) | check the journal for left-out phrases; lower `voice_control_vosk_confidence` a little |
| A phrase fires by itself | only whole phrases right after the start word count; choose a longer start word |
| The last dictated word goes missing | lower `voice_control_end_trim_seconds` |
| Bits of "Hey Jarvis" stay in the text | raise `voice_control_end_trim_seconds` |
| Too many notifications | `"voice_control_announce": false` |

The journal (`journalctl --user -u hyprwhspr -f`) logs each phrase heard (`[VOICE] Heard command phrase`), each
command transcript and each decision.

## Configuration

```jsonc
{
    "voice_control_end_on_pause": true,          // a command ends at the pause after it
    "voice_control_pause_seconds": 1.5,          // quiet that ends a command or a dictation command
    "voice_control_silence_seconds": 5,          // nothing said this long after the start word: dropped (0 = never)
    "voice_control_in_dictation": true,          // the start word inside a dictation listens for done, cancel, ...
    "voice_control_command_transcriber": "auto", // "auto" | "backend" | "vosk"
    "voice_control_announce": true,              // brief notification of what each command did
    "voice_control_phrases": {}                  // {"command": {...}, "dictation": {...}}: your own phrases
}
```

Every other voice control setting is described with [voice control](VOICE_CONTROL.md#configuration).
