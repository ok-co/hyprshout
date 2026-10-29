# LLM refinement

Send each dictation through an LLM before it's pasted. The built-in prompt cleans up the text and applies what you
meant rather than what you said:

- Punctuation, capitalization and grammar
- Obvious mis-hearings, fixed from context
- Self-corrections: "Tuesday, no wait, Wednesday" becomes "Wednesday"
- Filler words, false starts, stutters and repeated phrases are removed
- Rambling or convoluted passages are rephrased concisely, keeping every fact, name and number
- Your language and tone are kept; nothing is translated, answered or added

It works with any transcription backend. [OpenRouter](https://openrouter.ai) is built in (one key, hundreds of
models); any OpenAI-compatible chat completions endpoint works too: OpenAI, Groq, or a local Ollama or llama.cpp server.

## Setup

```bash
hyprwhspr config llm
```

The wizard asks for the provider, API key, model and optional extra instructions, then sends a sample dictation to
check it works. Restart the service afterwards:

```bash
systemctl --user restart hyprwhspr
```

## API key storage

The key never goes into `config.json`. It's typed at a hidden prompt (no terminal echo, no shell history) and stored
in the credential store, `~/.local/share/hyprwhspr/credentials`, which only your user can read (mode `0600`), the
same store the cloud transcription backends use. The OpenRouter key is stored as `openrouter`; a custom endpoint's
optional key as `llm-custom`.

- The key is read on every dictation, so replacing it needs no restart
- Logs and `hyprwhspr config llm status` show a masked key only
- A key is never sent over plain `http://` to a non-local host
- `hyprwhspr config llm forget-key` deletes it

The stored value can be a `${VAR}` reference to an environment variable of the service instead of the key itself.

## Your own prompt

Three layers, from lightest to strongest:

**Extra instructions** are appended to the active prompt, so you keep the built-in cleanup:

```jsonc
{
    "llm_refine_extra_instructions": "Use British spelling. My colleagues are Siobhan and Tadhg. Format lists as bullet points."
}
```

**A prompt file** replaces the built-in prompt:

```bash
hyprwhspr config llm prompt
```

This copies the built-in prompt to `~/.config/hyprwhspr/llm_prompt.md`, points `llm_refine_prompt_file` at it and
opens it in `$EDITOR`. The file is re-read for every dictation, so later edits apply without a restart. If the file
can't be read, the dictation is pasted unrefined.

**An inline prompt**, `llm_refine_prompt`, replaces the built-in prompt from `config.json`; a prompt file takes
precedence over it.

The dictation is sent as the user message, wrapped in `<transcript>...</transcript>`. A prompt should ask for the
edited text only, without preamble; an echoed `<transcript>` wrapper is stripped.

## Configuration

```jsonc
{
    "llm_refine_enabled": true,
    "llm_refine_provider": "openrouter",          // "openrouter" | "custom"
    "llm_refine_model": null,                      // null = provider default (OpenRouter: anthropic/claude-haiku-5.5)
    "llm_refine_endpoint_url": null,               // required for "custom"
    "llm_refine_prompt": null,                     // null = built-in prompt
    "llm_refine_prompt_file": null,                // wins over llm_refine_prompt
    "llm_refine_extra_instructions": "",
    "llm_refine_temperature": null,                // null = provider default
    "llm_refine_timeout": 15                       // seconds
}
```

A local model through Ollama, no key needed:

```jsonc
{
    "llm_refine_enabled": true,
    "llm_refine_provider": "custom",
    "llm_refine_endpoint_url": "http://localhost:11434/v1/chat/completions",
    "llm_refine_model": "qwen3:8b"
}
```

Any OpenRouter model id works in `llm_refine_model`. Fast, inexpensive models without extended reasoning keep the
delay before paste short; the OSD stays in its processing state while the LLM works.

## Order and failures

Refinement runs after [word overrides, filler filtering and symbol replacements](TEXT_PROCESSING.md), and before the
[`post_transcription_hook`](TEXT_PROCESSING.md#post-transcription-hook) and the trailing space. A hook therefore
receives the refined text.

If the LLM fails for any reason (no key, network error, HTTP error, timeout, empty or truncated reply), the original
text is pasted and the reason is logged to the service journal. The log never contains the dictation.

Not refined:

- `hyprwhspr record capture`, which streams the raw transcript
- `hyprwhspr transcribe --clean`
- [Live typing](../backends/REALTIME_WEBSOCKET.md), which is turned off while refinement is enabled, because it
  needs the whole dictation

In continuous recording mode, each pasted segment is refined separately.

## Commands

```bash
hyprwhspr config llm               # setup wizard
hyprwhspr config llm status        # provider, model, masked key, prompt source
hyprwhspr config llm test "so um send it Tuesday, no, Wednesday"
echo "text" | hyprwhspr config llm test
hyprwhspr config llm prompt        # write your own prompt in $EDITOR
hyprwhspr config llm disable       # turn off; settings and key are kept
hyprwhspr config llm forget-key    # delete the stored API key
```

`test` works while refinement is disabled, so you can try a model or prompt first. `hyprwhspr config validate` warns
when refinement is enabled without a stored key.
