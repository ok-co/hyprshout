"""Arch Linux support in scripts/install-deps.sh and the source-build path of scripts/install.sh.

pacman, sudo and git are replaced by stubs on PATH that log their arguments, so no
test installs packages or reads the host's /etc/os-release.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

import backend_installer  # noqa: E402

# Packaged only in the AUR; setup installs them from PyPI instead.
AUR_ONLY = ('python-sounddevice', 'python-pulsectl')

PACMAN_STUB = '''#!/usr/bin/env bash
echo "pacman $*" >> "$STUB_LOG"
case "$1" in
    -Q|-Qq)
        shift
        found=1
        for pkg in "$@"; do
            if [[ " $PACMAN_INSTALLED " == *" $pkg "* ]]; then echo "$pkg"; found=0; fi
        done
        exit $found ;;
esac
'''

SUDO_STUB = '''#!/usr/bin/env bash
exec "$@"
'''

GIT_STUB = '''#!/usr/bin/env bash
echo "git $*" >> "$STUB_LOG"
exit 1
'''


class ArchStubEnvironment:
    def __init__(self, tmp, os_release='ID=cachyos\nID_LIKE=arch\n', installed=''):
        tmp = Path(tmp)
        stubs = tmp / 'bin'
        stubs.mkdir()
        for name, body in (('pacman', PACMAN_STUB), ('sudo', SUDO_STUB), ('git', GIT_STUB)):
            (stubs / name).write_text(body)
            (stubs / name).chmod(0o755)
        (tmp / 'os-release').write_text(os_release)
        self.log = tmp / 'calls.log'
        self.log.touch()
        self.env = {
            'PATH': f'{stubs}:/usr/bin:/bin',
            'HOME': str(tmp / 'home'),
            'XDG_DATA_HOME': str(tmp / 'data'),
            'OS_RELEASE_FILE': str(tmp / 'os-release'),
            'STUB_LOG': str(self.log),
            'PACMAN_INSTALLED': installed,
        }

    def calls(self):
        return self.log.read_text().splitlines()


class InstallDepsArchTests(unittest.TestCase):
    def run_deps(self, stub, command):
        return subprocess.run(
            ['bash', '-c', f'source scripts/install-deps.sh; {command}'],
            cwd=ROOT, env=stub.env, capture_output=True, text=True,
        )

    def test_arch_derivatives_use_pacman(self):
        for os_release in ('ID=arch\n', 'ID=cachyos\nID_LIKE=arch\n', 'ID=manjaro\nID_LIKE="arch"\n'):
            with self.subTest(os_release=os_release), tempfile.TemporaryDirectory() as tmp:
                stub = ArchStubEnvironment(tmp, os_release=os_release)
                result = self.run_deps(stub, 'detect_distro; echo "$DISTRO $PKG_MANAGER"')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip().endswith('arch pacman'), result.stdout)

    def test_pacman_install_uses_official_repos_without_partial_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = ArchStubEnvironment(tmp)
            result = self.run_deps(stub, 'install_deps_pacman')
            self.assertEqual(result.returncode, 0, result.stderr)
            installs = [call for call in stub.calls() if call.startswith('pacman -S ')]
            self.assertTrue(installs)
            for call in installs:
                self.assertIn('--needed', call)
                self.assertNotIn('-Sy', call)
            packages = ' '.join(installs).split()
            for pkg in ('python-evdev', 'ydotool', 'wl-clipboard', 'xdotool', 'xorg-xprop',
                        'pipewire-pulse', 'python-pyperclip', 'gtk4-layer-shell'):
                self.assertIn(pkg, packages)
            for pkg in AUR_ONLY:
                self.assertNotIn(pkg, packages)
            self.assertIn('sounddevice and pulsectl will be installed', result.stdout)
            self.assertFalse([call for call in stub.calls() if 'yay' in call or 'paru' in call])

    def test_existing_pulseaudio_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = ArchStubEnvironment(tmp, installed='pulseaudio')
            result = self.run_deps(stub, 'install_deps_pacman')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('pipewire-pulse', ' '.join(stub.calls()))

    def test_installed_aur_package_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = ArchStubEnvironment(tmp, installed='hyprwhspr-git')
            result = self.run_deps(stub, 'install_deps_pacman')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("AUR package 'hyprwhspr-git' is installed", result.stdout)


class RocmArchTests(unittest.TestCase):
    def test_rocm_installs_from_official_repos(self):
        tools = {'rocm-smi': '/usr/bin/rocm-smi', 'pacman': '/usr/bin/pacman', 'yay': '/usr/bin/yay'}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'ROCM_PATH': str(Path(tmp) / 'rocm')}), \
                mock.patch.object(backend_installer.shutil, 'which', side_effect=tools.get), \
                mock.patch.object(backend_installer, 'run_command',
                                  return_value=subprocess.CompletedProcess([], 0)), \
                mock.patch.object(backend_installer, 'run_sudo_command') as sudo:
            self.assertFalse(backend_installer.setup_amd_support())
        sudo.assert_called_once()
        command = sudo.call_args.args[0]
        self.assertEqual(command[:2], ['pacman', '-S'])
        self.assertIn('rocm-hip-sdk', command)


class BootstrapArchTests(unittest.TestCase):
    def run_bootstrap(self, stub, **extra):
        return subprocess.run(
            ['bash', str(ROOT / 'scripts/install.sh')],
            env=dict(stub.env, **extra), capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            # No controlling terminal, so the installer's /dev/tty pre-flight fails.
            start_new_session=True,
        )

    def test_arch_defaults_to_aur_and_offers_source_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = ArchStubEnvironment(tmp)
            result = self.run_bootstrap(stub)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('yay -S hyprwhspr', result.stdout)
            self.assertIn('HYPRWHSPR_FROM_SOURCE=1', result.stdout)
            self.assertFalse((Path(tmp) / 'data/hyprwhspr/src').exists())

    def test_source_build_refuses_installed_aur_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = ArchStubEnvironment(tmp, installed='hyprwhspr')
            result = self.run_bootstrap(stub, HYPRWHSPR_FROM_SOURCE='1')
            self.assertEqual(result.returncode, 1)
            self.assertIn("AUR package 'hyprwhspr' is installed", result.stderr)
            self.assertFalse((Path(tmp) / 'data/hyprwhspr/src').exists())

    def test_source_build_proceeds_past_distro_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = ArchStubEnvironment(tmp)
            result = self.run_bootstrap(stub, HYPRWHSPR_FROM_SOURCE='1')
            self.assertIn('building from source', result.stdout)
            # Without a terminal it stops at the interactive pre-flight, before cloning.
            self.assertEqual(result.returncode, 1)
            self.assertIn('needs a terminal', result.stderr)
            self.assertFalse((Path(tmp) / 'data/hyprwhspr/src').exists())


if __name__ == '__main__':
    unittest.main()
