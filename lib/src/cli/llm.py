"""
LLM refinement commands for hyprwhspr (hyprwhspr config llm ...)
"""

import getpass
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.prompt import Prompt, Confirm

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..credential_manager import save_credential, get_credential, delete_credential, mask_api_key
except ImportError:
    from credential_manager import save_credential, get_credential, delete_credential, mask_api_key

try:
    from ..backend_utils import is_valid_http_url, redact_url
except ImportError:
    from backend_utils import is_valid_http_url, redact_url

try:
    from ..llm_refiner import (DEFAULT_PROMPT, DEFAULT_PROVIDER, LLM_PROVIDERS, LLMRefineError,
                               refine_text, resolve_target)
except ImportError:
    from llm_refiner import (DEFAULT_PROMPT, DEFAULT_PROVIDER, LLM_PROVIDERS, LLMRefineError,
                             refine_text, resolve_target)

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error

from ._shared import USER_CONFIG_DIR


LLM_ACTIONS = ('setup', 'status', 'test', 'prompt', 'disable', 'forget-key')

DEFAULT_PROMPT_FILE = USER_CONFIG_DIR / 'llm_prompt.md'

SAMPLE_TEXT = ("so um I wanted to ask if we can move the meeting to Tuesday, no wait, "
               "Wednesday at at three, and uh could you also could you also bring the the slides")

RESTART_HINT = "  systemctl --user restart hyprwhspr"


def llm_command(action: Optional[str] = None, text: Optional[str] = None):
    """Handle config llm subcommands"""
    action = action or 'setup'
    if action == 'setup':
        setup_llm()
    elif action == 'status':
        show_llm_status()
    elif action == 'test':
        if not test_llm(text):
            sys.exit(1)
    elif action == 'prompt':
        edit_llm_prompt()
    elif action == 'disable':
        disable_llm()
    elif action == 'forget-key':
        forget_llm_key()
    else:
        log_error(f"Unknown llm action: {action}")


def _provider_id(config) -> str:
    provider_id = config.get_setting('llm_refine_provider', DEFAULT_PROVIDER)
    return provider_id if provider_id in LLM_PROVIDERS else DEFAULT_PROVIDER


def _prompt_source(config) -> str:
    prompt_file = config.get_setting('llm_refine_prompt_file', None)
    if isinstance(prompt_file, str) and prompt_file.strip():
        return f"file {prompt_file}"
    prompt = config.get_setting('llm_refine_prompt', None)
    if isinstance(prompt, str) and prompt.strip():
        return "custom (llm_refine_prompt)"
    return "built-in cleanup prompt"


def _ask_api_key(provider_id: str) -> bool:
    """Prompt for the provider's key and store it; True if a key is stored afterwards."""
    provider = LLM_PROVIDERS[provider_id]
    credential = provider['credential']
    existing = get_credential(credential)
    if existing:
        print(f"\nFound stored API key: {mask_api_key(existing)}")
        if Confirm.ask("Keep the stored API key?", default=True):
            return True
    elif not provider['api_key_required']:
        if not Confirm.ask("Does this endpoint need an API key?", default=False):
            return False
    if provider_id == 'openrouter':
        print("\nCreate a key at https://openrouter.ai/keys")
    # getpass keeps the key out of the terminal and shell history
    api_key = getpass.getpass(f"Enter {provider['api_key_description']}: ").strip()
    if not api_key:
        log_warning("No API key entered")
        return bool(existing)
    if provider_id == 'openrouter' and not api_key.startswith('sk-or-'):
        log_warning("OpenRouter keys usually start with 'sk-or-'")
        if not Confirm.ask("Store it anyway?", default=False):
            return bool(existing)
    if save_credential(credential, api_key):
        log_success("API key stored in the credential store (owner-only permissions)")
        return True
    log_error("Failed to store API key")
    return bool(existing)


def setup_llm():
    """Interactive setup for LLM refinement"""
    config = ConfigManager(verbose=False)

    print("\n" + "=" * 60)
    print("LLM refinement")
    print("=" * 60)
    print("\nEach dictation is sent to an LLM that fixes punctuation and mis-hearings,")
    print("applies your self-corrections, drops repetition and tightens rambling parts")
    print("before it is pasted. If the LLM fails or times out, the original text is pasted.")
    print("\nProviders:")
    print("  [1] OpenRouter   one key, hundreds of models")
    print("  [2] Custom       any OpenAI-compatible chat completions endpoint")
    print("                   (OpenAI, Groq, Ollama, llama.cpp server, ...)")

    current = _provider_id(config)
    choice = Prompt.ask("Provider", choices=['1', '2'], default='2' if current == 'custom' else '1')
    provider_id = 'custom' if choice == '2' else 'openrouter'
    provider = LLM_PROVIDERS[provider_id]

    if provider_id == 'custom':
        current_url = config.get_setting('llm_refine_endpoint_url', None) or ''
        while True:
            url = Prompt.ask("Chat completions URL (e.g. http://localhost:11434/v1/chat/completions)",
                             default=current_url or None)
            url = (url or '').strip()
            if is_valid_http_url(url):
                break
            log_warning("Enter a full http:// or https:// URL")
        config.set_setting('llm_refine_endpoint_url', url)

    has_key = _ask_api_key(provider_id)
    if provider['api_key_required'] and not has_key:
        log_error("An API key is required for this provider; nothing changed")
        return

    current_model = config.get_setting('llm_refine_model', None) if current == provider_id else None
    default_model = current_model or provider['default_model']
    if provider_id == 'openrouter':
        print("\nAny model id from https://openrouter.ai/models works. Fast, cheap,")
        print("non-reasoning models keep the delay before paste short.")
    model = Prompt.ask("Model", default=default_model or None)
    model = (model or '').strip()
    if not model:
        log_error("A model is required; nothing changed")
        return
    config.set_setting('llm_refine_provider', provider_id)
    # Store null for the provider default so a future default upgrade applies
    config.set_setting('llm_refine_model', None if model == provider['default_model'] else model)

    print(f"\nPrompt: {_prompt_source(config)}")
    print("Use 'hyprwhspr config llm prompt' to write your own prompt in an editor.")
    extra = config.get_setting('llm_refine_extra_instructions', '') or ''
    print("\nExtra instructions are appended to the prompt, e.g.")
    print("  \"Use British spelling. My colleagues are Siobhan and Tadhg.\"")
    extra = Prompt.ask("Extra instructions (blank for none)", default=extra or '')
    config.set_setting('llm_refine_extra_instructions', (extra or '').strip())

    config.set_setting('llm_refine_enabled', True)
    if not config.save_config():
        log_error("Failed to save configuration")
        return
    log_success("LLM refinement enabled")

    if Confirm.ask("\nSend a sample dictation to test it now?", default=True):
        test_llm(None, config)

    print("\nRestart the service for the change to take effect:")
    print(RESTART_HINT)


def show_llm_status():
    """Print the LLM refinement configuration (key masked)"""
    config = ConfigManager(verbose=False)
    enabled = config.get_setting('llm_refine_enabled', False) is True
    print(f"Enabled:       {'yes' if enabled else 'no'}")
    try:
        target = resolve_target(config.get_setting)
    except LLMRefineError as e:
        print(f"Target:        not usable: {e}")
        return
    print(f"Provider:      {target['provider']['name']}")
    print(f"Endpoint:      {redact_url(target['endpoint'])}")
    print(f"Model:         {target['model']}")
    key = get_credential(target['credential'])
    if key:
        print(f"API key:       {mask_api_key(key)} (credential '{target['credential']}')")
    else:
        required = target['provider']['api_key_required']
        print(f"API key:       none stored{' (required)' if required else ''}")
    print(f"Prompt:        {_prompt_source(config)}")
    extra = config.get_setting('llm_refine_extra_instructions', '') or ''
    if extra.strip():
        print(f"Extra:         {extra.strip()}")
    print(f"Timeout:       {config.get_setting('llm_refine_timeout', 15)}s")


def test_llm(text: Optional[str] = None, config=None) -> bool:
    """Run text (argument, piped stdin, or a sample) through the configured LLM"""
    config = config or ConfigManager(verbose=False)
    if text is None and not sys.stdin.isatty():
        text = sys.stdin.read()
    text = (text or '').strip() or SAMPLE_TEXT
    print(f"\nInput:  {text}")
    started = time.monotonic()
    try:
        result = refine_text(text, config.get_setting)
    except LLMRefineError as e:
        log_error(f"LLM refinement failed: {e}")
        return False
    print(f"Output: {result}")
    log_info(f"Took {time.monotonic() - started:.1f}s")
    if config.get_setting('llm_refine_enabled', False) is not True:
        log_info("Refinement is disabled; run 'hyprwhspr config llm' to enable it")
    return True


def edit_llm_prompt():
    """Open the prompt file in $EDITOR, creating it from the active prompt"""
    config = ConfigManager(verbose=False)
    configured = config.get_setting('llm_refine_prompt_file', None)
    path = Path(configured).expanduser() if isinstance(configured, str) and configured.strip() else DEFAULT_PROMPT_FILE
    if not path.exists():
        inline = config.get_setting('llm_refine_prompt', None)
        seed = inline.strip() if isinstance(inline, str) and inline.strip() else DEFAULT_PROMPT
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(seed + '\n', encoding='utf-8')
        except OSError as e:
            log_error(f"Could not create {path}: {e}")
            return
        log_info(f"Created {path} from the {'current' if seed != DEFAULT_PROMPT else 'built-in'} prompt")
    if not (isinstance(configured, str) and configured.strip()):
        config.set_setting('llm_refine_prompt_file', str(path))
        if not config.save_config():
            log_error("Failed to save configuration")
            return
        print("\nRestart the service to start using the prompt file:")
        print(RESTART_HINT)
    print("\nThe dictation is sent as the user message inside <transcript> tags.")
    print("The file is re-read for every dictation, so later edits apply immediately.")
    editor = os.environ.get('EDITOR', 'nano')
    try:
        subprocess.run([editor, str(path)], check=True)
    except Exception as e:
        log_error(f"Failed to open editor: {e}")
        print(f"Edit the prompt at {path}")


def disable_llm():
    """Turn LLM refinement off, keeping its settings and key"""
    config = ConfigManager(verbose=False)
    config.set_setting('llm_refine_enabled', False)
    if config.save_config():
        log_success("LLM refinement disabled (settings and API key kept)")
        print("\nRestart the service for the change to take effect:")
        print(RESTART_HINT)
    else:
        log_error("Failed to save configuration")


def forget_llm_key():
    """Delete the stored API key of the configured provider"""
    config = ConfigManager(verbose=False)
    credential = LLM_PROVIDERS[_provider_id(config)]['credential']
    if delete_credential(credential):
        log_success(f"Deleted stored API key '{credential}'")
    else:
        log_info(f"No stored API key '{credential}'")
