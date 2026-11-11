"""The kernel confinement of the transform child: even unrestricted code there can do nothing harmful.

These tests run the harness with the RestrictedPython guard disabled (a test-only request field the
tool never sets), so they check the operating-system layer alone, as if a payload had escaped.
"""

import json
import subprocess
import sys

import pytest

from ai_assist.transform_harness import confinement_supported


def run_unguarded(code: str, text: str = "x\n") -> dict:
    request = {"code": code, "text": text, "path": "f.txt", "test_unguarded": True}
    out = subprocess.run(
        [sys.executable, "-I", "-m", "ai_assist.transform_harness"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(out.stdout)


requires_landlock = pytest.mark.skipif(not confinement_supported(), reason="kernel confinement unavailable here")


# Each case prints RAN only if the harmful action succeeded.
HARMFUL = {
    "read a file": "print('RAN', open('/etc/hostname').read())",
    "write a file": "open('written.txt', 'w').write('x'); print('RAN')",
    "list a directory": "import os; print('RAN', os.listdir('/'))",
    "run a program": "import os; print('RAN' if os.system('true') == 0 else 'refused')",
    "subprocess": "import subprocess; subprocess.run(['true']); print('RAN')",
    "load socket": "import socket; print('RAN')",
    "tcp connect via ctypes": (
        "libc = ctypes.CDLL(None)\n"
        "fd = libc.socket(2, 1, 0)\n"  # AF_INET, SOCK_STREAM
        "print('RAN' if fd >= 0 else 'refused')"
    ),
    "udp socket via ctypes": (
        "libc = ctypes.CDLL(None)\n"
        "fd = libc.socket(2, 2, 0)\n"  # AF_INET, SOCK_DGRAM: Landlock does not cover UDP
        "print('RAN' if fd >= 0 else 'refused')"
    ),
    "exec via ctypes": (
        "libc = ctypes.CDLL(None)\n"
        "r = libc.execve(b'/bin/true', None, None)\n"
        "print('RAN' if r == 0 else 'refused')"
    ),
}


@requires_landlock
@pytest.mark.parametrize("code", HARMFUL.values(), ids=HARMFUL.keys())
def test_escaped_code_can_do_nothing_harmful(code):
    # ctypes is in memory (the harness used it to confine itself): reach it as an escape would
    result = run_unguarded("import sys; ctypes = sys.modules['ctypes']\n" + code)
    assert "RAN" not in result["output"], result


@requires_landlock
def test_unguarded_code_still_has_its_input_and_preloaded_modules():
    """The lockdown happens after the input is read and the modules are loaded."""
    result = run_unguarded("print(len(text), json.dumps([1]), datetime.datetime.strptime('2026', '%Y').year)")
    assert result == {"output": "2 [1] 2026\n", "error": None, "confinement": result["confinement"]}
    assert result["confinement"].startswith("landlock")


def test_confinement_is_reported():
    result = run_unguarded("print(1)")
    assert result["confinement"].startswith(("landlock", "unavailable:"))
