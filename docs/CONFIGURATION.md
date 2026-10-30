# Configuration

Validate with ease:

```bash
hyprwhspr config validate  # Settings
hyprwhspr status --report  # Settings, installation, live state
```

Configure via `hyprwhspr setup`, the [settings editor](#settings-editor), the CLI, or by editing
`~/.config/hyprwhspr/config.json` directly.

The config file uses **sparse storage** — it only contains values you've changed from the defaults, so it stays clean and upstream default changes apply automatically on update.

There is also a `$schema` reference for IDE autocompletion and validation:

```jsonc
{
    "$schema": "https://raw.githubusercontent.com/goodroot/hyprwhspr/main/share/config.schema.json"
}
```

To view your overrides or the full resolved config:

```bash
hyprwhspr config show        # Show your overrides only
hyprwhspr config show --all  # Show all settings including defaults
```

## Settings editor

Change every setting, API key and custom prompt from one screen, without writing JSON:

```bash
hyprwhspr settings
```

`hyprwhspr config` without an action opens it too. Sections are on the left (Essentials first, then recording,
transcription, LLM refinement, scratchpad, voice control and the rest), their settings on the right, and what the
selected one does below, with its default.

| Key | Does |
| --- | --- |
| ↑ ↓ ← → | move between sections and settings |
| Space | turn a setting on or off, or step to its next choice |
| Enter | edit: pick from a list, type a value, or open your editor for prompts and structured settings |
| Backspace | back to the default |
| e | edit any setting in your text editor (`$VISUAL` or `$EDITOR`, else nano or vi) |
| / | search names and descriptions ("vosk", "pause", "prompt") |
| c | show only the settings you've changed |
| s | save |
| ? | all keys |
| q | quit; unsaved changes are offered for saving |

- **Checked before saving**: a value is checked against the schema as you set it, and the whole
  config again when you save. An invalid setting, even one written by hand, is listed instead of saved
- **Custom prompts** for [LLM refinement](usage/LLM_REFINEMENT.md) (`llm_refine_prompt`) and the
  [voice assistant](usage/VOICE_CONTROL.md#voice-assistant) (`assistant_prompt`) open in your editor with the
  built-in prompt as a starting point. Saved unchanged, or emptied, the built-in one stays in use
- **API keys** (OpenRouter, a custom LLM endpoint, Hugging Face, your REST provider) are typed hidden and kept in
  the credential store, never in `config.json`. Backspace removes one
- **Applying**: when the service is running, saving offers to restart it so the changes take effect
- The file stays sparse: only settings you changed are written, and `${VAR}` tokens are kept as you wrote them

## Minimal configuration

Only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base"
}
```

## Environment variable substitution

Both `config.json` and `credentials.json` support `${VAR}` tokens.

Tokens are stored as-is on disk and expanded at read time:

```jsonc
{
  "rest_api_key": "${OPENAI_API_KEY}"
}
```
