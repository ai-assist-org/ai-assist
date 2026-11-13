"""Child process of internal__transform: run one restricted payload over text read from stdin.

Run as `python -I -m ai_assist.transform_harness`. Reads {"code", "text", "path"} as JSON on stdin,
prints {"output", "error", "confinement"} as JSON on stdout. Never opens a file itself.

Two layers. The payload is compiled by RestrictedPython: no `_`-prefixed or frame-inspection
attributes, no str.format, no imports; modules are given as namespaces of their public names only.
And before running it, the process confines itself with the kernel (Landlock on Linux): no file
access at all, so no further module can be loaded either (no socket, no ctypes), no TCP, no new
process. The second layer is what makes a payload that escapes the first one harmless.
"""

import ast
import collections
import contextlib
import copy
import csv
import ctypes
import datetime
import functools
import io
import itertools
import json
import math
import operator
import platform
import re
import resource
import statistics
import sys
import textwrap
import traceback
import types
import warnings

import yaml
from RestrictedPython import PrintCollector, compile_restricted, safe_builtins
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)

PRELOADED = (
    "re",
    "json",
    "csv",
    "math",
    "statistics",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "textwrap",
    "io",
    "yaml",
)
OUTPUT_CAP = 10 * 1024 * 1024
FILENAME = "<transform>"

_INPLACE = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "|=": operator.ior,
    "&=": operator.iand,
    "^=": operator.ixor,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
}


def _public(module: types.ModuleType) -> types.SimpleNamespace:
    """A module's public names, without its submodules: never the module object itself."""
    names = {
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith("_") and not isinstance(getattr(module, name), types.ModuleType)
    }
    return types.SimpleNamespace(**names)


_MISSING = object()
# builtins a payload might reach for, and why they are absent (instead of a bare NameError)
_REMOVED = {
    "open": "the file content is already in `text` / `lines`",
    "eval": "write the code directly",
    "exec": "write the code directly",
    "compile": "write the code directly",
    "globals": "use the names directly",
    "locals": "use the names directly",
    "vars": "use getattr(obj, name)",
    "setattr": "assign the attribute: obj.name = value",
    "delattr": "use `del obj.name`",
    "input": "there is no terminal; the data is in `text`",
    "breakpoint": "there is no debugger",
    "memoryview": "work on str or bytes",
    "help": "see the tool description",
    "exit": "let the payload end",
    "quit": "let the payload end",
}


def _getattr(obj: object, name: str, default: object = _MISSING) -> object:
    """safer_getattr's name checks, but a missing attribute raises (safer_getattr returns None)."""
    value = safer_getattr(obj, name, _MISSING)
    if value is not _MISSING:
        return value
    if default is _MISSING:
        raise AttributeError(f"{type(obj).__name__!r} object has no attribute {name!r}")
    return default


def _hasattr(obj: object, name: str) -> bool:
    try:
        _getattr(obj, name)
    except AttributeError, TypeError, NotImplementedError:  # incl. refused `_` names
        return False
    return True


def _type(*args: object) -> type:
    """type(x) to compare types; never type(name, bases, dict) (no dynamic classes)."""
    if len(args) != 1:
        raise TypeError("type() takes 1 argument here: dynamic classes are not available")
    return type(args[0])


def _importer(modules: dict[str, types.SimpleNamespace]):
    """`import re` / `from collections import Counter` give the preloaded namespaces; nothing else."""

    def guarded_import(name: str, globals_: object = None, locals_: object = None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        # C code (e.g. datetime.strptime -> _strptime) imports through the payload's __import__. A
        # payload cannot ask for a `_` name itself (RestrictedPython refuses it in `import`), and
        # only modules already loaded before the lockdown are returned.
        if level == 0 and name.startswith("_") and name in sys.modules:
            return sys.modules[name]
        raise ImportError(f"imports are not available; use the preloaded names: {', '.join(PRELOADED)}")

    return guarded_import


def namespace(text: str, path: str) -> dict:
    modules = {
        name: _public(module)
        for name, module in (
            ("re", re),
            ("json", json),
            ("csv", csv),
            ("math", math),
            ("statistics", statistics),
            ("datetime", datetime),
            ("collections", collections),
            ("itertools", itertools),
            ("functools", functools),
            ("textwrap", textwrap),
        )
    }
    # io and yaml: only what reads/writes in memory (io.open, yaml.load could reach files or code)
    modules["io"] = types.SimpleNamespace(StringIO=io.StringIO)
    modules["yaml"] = types.SimpleNamespace(
        safe_load=yaml.safe_load, safe_load_all=yaml.safe_load_all, safe_dump=yaml.safe_dump
    )
    builtins = dict(safe_builtins)
    # RestrictedPython leaves out builtins that iterate in C because they bypass `_getiter_`, the
    # hook Zope uses for permission checks. Nothing here relies on `_getiter_` for access control
    # (attribute names are what is guarded), so they are safe to provide.
    for fn in (list, dict, set, frozenset, tuple, sorted, sum, min, max, enumerate, zip, map, filter):
        builtins[fn.__name__] = fn
    builtins.update(any=any, all=all, reversed=reversed, iter=iter, next=next, __import__=_importer(modules))
    builtins.update(getattr=_getattr, hasattr=_hasattr, type=_type)
    return {
        "__builtins__": builtins,
        "__name__": "transform",
        "__metaclass__": type,  # class statements (RestrictedPython looks it up)
        "_print_": PrintCollector,
        "_getattr_": _getattr,
        "_getitem_": operator.getitem,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        # writes go to objects the payload created (or its own copies of the namespaces), in a
        # confined process: Zope's write guard, which only lets plain dict/list through, would
        # refuse Counter and defaultdict for nothing. Attribute names are still checked.
        "_write_": lambda obj: obj,
        "_inplacevar_": lambda op, x, y: _INPLACE[op](x, y),
        **modules,
        "text": text,
        "lines": text.splitlines(),
        "path": path,
    }


def _syntax_error(exc: SyntaxError) -> str:
    """RestrictedPython reports its findings as a tuple of "Line N: ..." strings."""
    if not isinstance(exc.msg, tuple):  # plain Python syntax error
        return f"line {exc.lineno}: {type(exc).__name__}: {exc.msg}"
    first = re.sub(r"^Line (\d+): ", r"line \1: ", str(exc.msg[0]))
    return re.sub(r" at statement: .*$", "", first)


# Landlock (linux/landlock.h): syscall numbers are the same on every architecture using the
# generic table (x86_64 included); the access rights handled grow with the ABI version.
_LANDLOCK_CREATE_RULESET, _LANDLOCK_RESTRICT_SELF = 444, 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_PR_SET_NO_NEW_PRIVS = 38
_FS_ACCESS_BY_ABI = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1, 5: (1 << 16) - 1}
_NET_TCP_BIND_CONNECT = (1 << 0) | (1 << 1)  # ABI 4
_SCOPE_ABSTRACT_UNIX_SOCKET_AND_SIGNAL = (1 << 0) | (1 << 1)  # ABI 6


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


def _libc() -> ctypes.CDLL | None:
    if not sys.platform.startswith("linux"):
        return None
    return ctypes.CDLL(None, use_errno=True)


def _landlock_abi(libc: ctypes.CDLL) -> int:
    abi = libc.syscall(_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), _LANDLOCK_CREATE_RULESET_VERSION)
    return abi if abi > 0 else 0


def confinement_supported() -> bool:
    """Whether this system can confine the transform child (Landlock enabled in the kernel)."""
    libc = _libc()
    return libc is not None and _landlock_abi(libc) > 0


def _limit_resources() -> None:
    """No file growth, no new process, bounded CPU and memory (each where the platform has it)."""
    for name, value in (
        ("RLIMIT_FSIZE", 0),
        ("RLIMIT_NPROC", 0),
        ("RLIMIT_CORE", 0),
        ("RLIMIT_CPU", 60),
        ("RLIMIT_AS", 2 * 1024**3),
    ):
        limit = getattr(resource, name, None)
        if limit is not None:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(limit, (value, value))


# seccomp: refuse the syscalls Landlock does not cover. ctypes is loaded (it set up Landlock), so an
# escaped payload could call socket() or execve() directly: the kernel must refuse them.
_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER = 22, 2
_SECCOMP_RET_ALLOW, _SECCOMP_RET_EPERM = 0x7FFF0000, 0x00050000 | 1
_BPF_LD_W_ABS, _BPF_JEQ_K, _BPF_JGE_K, _BPF_RET_K = 0x20, 0x15, 0x35, 0x06
_X32_SYSCALL_BIT = 0x40000000
# arch -> (AUDIT_ARCH, syscalls denied: socket, execve, execveat, ptrace, bpf, io_uring setup/enter/register)
_SECCOMP_ARCHES = {
    "x86_64": (0xC000003E, (41, 59, 322, 101, 321, 425, 426, 427)),
    "aarch64": (0xC00000B7, (198, 221, 281, 117, 280, 425, 426, 427)),
}


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _deny_syscalls(libc: ctypes.CDLL) -> str | None:
    """Install the seccomp filter; None when done, else why not."""
    machine = platform.machine()
    if machine not in _SECCOMP_ARCHES:
        return f"no seccomp filter for {machine}: sockets are not blocked"
    audit_arch, denied = _SECCOMP_ARCHES[machine]
    deny = (_BPF_RET_K, 0, 0, _SECCOMP_RET_EPERM)
    program = [(_BPF_LD_W_ABS, 0, 0, 4), (_BPF_JEQ_K, 1, 0, audit_arch), deny, (_BPF_LD_W_ABS, 0, 0, 0)]
    if machine == "x86_64":
        program += [(_BPF_JGE_K, 0, 1, _X32_SYSCALL_BIT), deny]
    for number in denied:
        program += [(_BPF_JEQ_K, 0, 1, number), deny]
    program.append((_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW))
    filters = (_SockFilter * len(program))(*(_SockFilter(*instruction) for instruction in program))
    fprog = _SockFprog(len(program), filters)
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        return f"seccomp setup failed (errno {ctypes.get_errno()}): sockets are not blocked"
    return None


def lock_down(sandboxed: bool = False) -> str:
    """Confine this process for good; describe the confinement, or why there is none."""
    _limit_resources()
    if sandboxed:  # macOS: started under sandbox-exec by transform_tools.child_command
        return "sandbox-exec: no network, no new process, no file write, no read in the home directory"
    libc = _libc()
    if libc is None:
        return f"unavailable: no kernel confinement on {sys.platform}"
    abi = _landlock_abi(libc)
    if abi == 0:
        return f"unavailable: Landlock is not enabled in this kernel (errno {ctypes.get_errno()})"
    attr = _RulesetAttr(handled_access_fs=_FS_ACCESS_BY_ABI[max(a for a in _FS_ACCESS_BY_ABI if a <= abi)])
    size = 8
    if abi >= 4:
        attr.handled_access_net, size = _NET_TCP_BIND_CONNECT, 16
    if abi >= 6:
        attr.scoped, size = _SCOPE_ABSTRACT_UNIX_SOCKET_AND_SIGNAL, 24
    # an empty ruleset: every handled access is denied, from now on and for good
    ruleset = libc.syscall(_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(size), 0)
    if ruleset < 0 or libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        return f"unavailable: Landlock setup failed (errno {ctypes.get_errno()})"
    restricted = libc.syscall(_LANDLOCK_RESTRICT_SELF, ruleset, 0)
    libc.close(ruleset)
    if restricted != 0:
        return f"unavailable: Landlock restriction failed (errno {ctypes.get_errno()})"
    covered = "files" + (", TCP" if abi >= 4 else "") + (", abstract sockets and signals" if abi >= 6 else "")
    seccomp_problem = _deny_syscalls(libc)
    syscalls = f"; {seccomp_problem}" if seccomp_problem else "; no socket, exec or ptrace"
    return f"landlock ABI {abi}: no {covered}; no new process{syscalls}"


def _payload_line(exc: BaseException) -> int | None:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == FILENAME]
    return frames[-1].lineno if frames else None


class _ExpandAugAssign(ast.NodeTransformer):
    """`x[k] += v` -> `x[k] = x[k] + v` (same for `x.a += v`): RestrictedPython refuses the first,
    allows the second, which it then guards as usual."""

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        target = node.target
        current: ast.expr
        if isinstance(target, ast.Subscript):
            current = ast.Subscript(
                value=copy.deepcopy(target.value), slice=copy.deepcopy(target.slice), ctx=ast.Load()
            )
        elif isinstance(target, ast.Attribute):
            current = ast.Attribute(value=copy.deepcopy(target.value), attr=target.attr, ctx=ast.Load())
        else:
            return node
        current = ast.copy_location(current, target)
        value = ast.copy_location(ast.BinOp(left=current, op=node.op, right=node.value), node)
        return ast.copy_location(ast.Assign(targets=[target], value=value), node)


def run(code: str, text: str, path: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "prints, but never reads 'printed'": expected here
        try:
            tree = ast.fix_missing_locations(_ExpandAugAssign().visit(ast.parse(code, FILENAME)))
            bytecode = compile_restricted(tree, FILENAME, "exec")
        except SyntaxError as exc:
            return {"output": "", "error": _syntax_error(exc)}
    scope = namespace(text, path)
    try:
        exec(bytecode, scope)  # nosec B102 - compiled by RestrictedPython, guarded namespace
    except Exception as exc:
        line = _payload_line(exc)
        where = f"line {line}: " if line else ""
        name = getattr(exc, "name", None)
        if isinstance(exc, NameError) and name in _REMOVED:
            return {
                "output": _printed(scope),
                "error": f"{where}{name} is not available in a transform payload: {_REMOVED[name]}",
            }
        return {"output": _printed(scope), "error": f"{where}{type(exc).__name__}: {exc}"}
    return {"output": _printed(scope), "error": None}


def _printed(scope: dict) -> str:
    collector = scope.get("_print")
    output = collector() if collector else ""
    return output[:OUTPUT_CAP]


def _run_unguarded(code: str, text: str, path: str) -> dict:
    """Test-only: run plain Python under the kernel confinement alone (the tool never asks this)."""
    scope = {**namespace(text, path), "__builtins__": __builtins__}
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(code, scope)  # nosec B102 - test-only path checking the kernel confinement
    except BaseException as exc:  # noqa: BLE001 - report whatever the confinement stopped
        return {"output": out.getvalue(), "error": f"{type(exc).__name__}: {exc}"}
    return {"output": out.getvalue(), "error": None}


def main() -> None:
    request = json.loads(sys.stdin.read())
    # strptime imports _strptime, locale and calendar on first use: load them while files are readable
    datetime.datetime.strptime("2000-01-01", "%Y-%m-%d")
    # the input is read and the modules are loaded: nothing else needs the file system
    confinement = lock_down(bool(request.get("sandboxed")))
    runner = _run_unguarded if request.get("test_unguarded") else run
    result = runner(request["code"], request["text"], request["path"])
    sys.stdout.write(json.dumps({**result, "confinement": confinement}))


if __name__ == "__main__":
    main()
