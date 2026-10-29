"""
Scratchpad commands: show, paste, copy, clear, undo, edit and read aloud the
draft that scratchpad dictations build (hyprwhspr record start --scratchpad).

show, edit, status and voice read the draft or config directly, so they work
while the daemon is stopped; the rest go through the daemon, which owns the
undo history, the paste path and playback.
"""

import json
import os
import socket
import subprocess
import sys

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..paths import SCRATCHPAD_FILE, SOCKET_FILE
except ImportError:
    from paths import SCRATCHPAD_FILE, SOCKET_FILE

try:
    from ..scratchpad import Scratchpad
except ImportError:
    from scratchpad import Scratchpad

try:
    from ..speech_output import DEFAULT_MODEL as SPEECH_MODEL, VOICES, configured_voice, normalize_voice
except ImportError:
    from speech_output import DEFAULT_MODEL as SPEECH_MODEL, VOICES, configured_voice, normalize_voice

try:
    from ..output_control import log_info, log_success, log_error
except ImportError:
    from output_control import log_info, log_success, log_error


DAEMON_ACTIONS = ('paste', 'copy', 'clear', 'undo', 'speak', 'stop')


def scratchpad_command(action: str = 'show', value=None):
    """Handle scratchpad subcommands; exits non-zero on failure."""
    action = action or 'show'
    if value is not None and action not in ('speak', 'voice'):
        log_error(f"scratchpad {action} takes no argument")
        sys.exit(1)
    if value is not None and normalize_voice(value) is None:
        log_error(f"Unknown voice {value!r}; choose one of: {', '.join(VOICES)}")
        sys.exit(1)
    if action == 'voice':
        if not choose_voice(value):
            sys.exit(1)
    elif action == 'speak' and value is not None:
        if not request_daemon(action, voice=normalize_voice(value)):
            sys.exit(1)
    elif action == 'show':
        show_draft()
    elif action == 'status':
        show_status()
    elif action == 'edit':
        if not edit_draft():
            sys.exit(1)
    elif action in DAEMON_ACTIONS:
        if not request_daemon(action):
            sys.exit(1)
    else:
        log_error(f"Unknown scratchpad action: {action}")
        sys.exit(1)


def show_draft():
    """Print the draft to stdout, so it can be piped."""
    draft = Scratchpad(SCRATCHPAD_FILE).read()
    if draft:
        print(draft)
    else:
        print("The draft is empty", file=sys.stderr)


def show_status():
    draft = Scratchpad(SCRATCHPAD_FILE).read()
    config = ConfigManager(verbose=False)
    model = config.get_setting('scratchpad_model', None) or config.get_setting('llm_refine_model', None)
    print(f"Draft:         {len(draft.split())} words, {len(draft)} chars ({SCRATCHPAD_FILE})")
    print(f"Shortcut:      {config.get_setting('scratchpad_shortcut', None) or 'not set'}")
    print(f"Paste key:     {config.get_setting('scratchpad_paste_shortcut', None) or 'not set'}")
    print(f"Model:         {model or 'provider default'}")
    print(f"Timeout:       {config.get_setting('scratchpad_timeout', 30)}s")
    clear = config.get_setting('scratchpad_clear_on_paste', True) is not False
    print(f"After paste:   {'cleared' if clear else 'kept'}")
    print(f"Speak key:     {config.get_setting('scratchpad_speak_shortcut', None) or 'not set'}")
    print(f"Voice:         {configured_voice(config.get_setting)} "
          f"({config.get_setting('scratchpad_speech_model', None) or SPEECH_MODEL})")


def choose_voice(value=None) -> bool:
    """List the voices, or store the one the draft is read in from now on."""
    config = ConfigManager(verbose=False)
    current = configured_voice(config.get_setting)
    if value is None:
        for voice in VOICES:
            print(f"{'*' if voice == current else ' '} {voice}")
        print("Set one with: hyprwhspr scratchpad voice NAME", file=sys.stderr)
        return True
    voice = normalize_voice(value)
    config.set_setting('scratchpad_voice', voice)
    if not config.save_config():
        log_error("Could not save the configuration")
        return False
    log_success(f"Voice set to {voice}")
    # A running daemon reads its config only at start; tell it too
    if SOCKET_FILE.exists():
        return request_daemon('voice', voice=voice)
    return True


def edit_draft() -> bool:
    """Open the draft in $EDITOR; the daemon re-reads it before the next change."""
    try:
        SCRATCHPAD_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Create it private: an editor would create it with the umask.
        os.close(os.open(SCRATCHPAD_FILE, os.O_WRONLY | os.O_CREAT, 0o600))
    except OSError as e:
        log_error(f"Could not create {SCRATCHPAD_FILE}: {e}")
        return False
    editor = os.environ.get('EDITOR', 'nano')
    try:
        subprocess.run([editor, str(SCRATCHPAD_FILE)], check=True)
    except Exception as e:
        log_error(f"Failed to open editor: {e}")
        print(f"Edit the draft at {SCRATCHPAD_FILE}")
        return False
    return True


def request_daemon(action: str, voice=None) -> bool:
    """Send one scratchpad request; a paste is never retried."""
    request = {'verb': f'scratchpad_{action}'}
    if voice is not None:
        request['voice'] = voice
    request_sent = False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(15.0)
            client.connect(str(SOCKET_FILE))
            client.sendall(json.dumps(request).encode('utf-8') + b'\n')
            request_sent = True
            response = bytearray()
            while b'\n' not in response:
                chunk = client.recv(4096)
                if not chunk:
                    raise ValueError('No complete response; update or restart the daemon')
                response.extend(chunk)
                if len(response) > 65536:
                    raise ValueError('Invalid oversized daemon response')
            result = json.loads(response.split(b'\n', 1)[0])
            if not isinstance(result, dict) or not isinstance(result.get('ok'), bool):
                raise ValueError('Invalid daemon response; update or restart the daemon')
    except TimeoutError:
        log_error('The daemon did not answer in time.')
        if request_sent and action == 'paste':
            log_error('The paste may already have been dispatched; check the destination before retrying.')
        return False
    except (OSError, ValueError) as exc:
        log_error(f'Scratchpad {action} failed: {exc}. Check that the daemon is running and up to date.')
        return False
    if not result['ok']:
        message = result.get('error', 'failed')
        if action == 'undo' and message == 'Nothing to undo':
            log_info(message)
            return True
        log_error(f'Scratchpad {action}: {message}')
        return False
    log_success(result.get('message', 'Done'))
    return True
