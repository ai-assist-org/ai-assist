#!/usr/bin/env bash
# awl-check-config.sh — deterministic checks on ai-assist's own configuration.
#
# Usage:
#   awl-check-config.sh lint <config_dir> <signals_jsonl> <proposals_dir> <source_dir>
#   awl-check-config.sh gate <config_dir> <signals_jsonl> <proposals_dir> <source_dir>
#
# <source_dir> is ai-assist's source tree (the AWL built-in ${source_dir}).
#
# lint prints INFO lines (facts worth knowing), NOTE lines (report only:
# risky or broad entries, refusals that need a prompt fix), LINT lines
# (actionable problems), CANDIDATE lines (allowlist changes a proposal may
# make, with when the command was last refused), then config_issues=<n>
# counting LINT + CANDIDATE lines, and saves snapshots of allowed_commands.json,
# allowed_paths.json and the installed skills (name and path only) in
# <proposals_dir> (live-*.json) for the model.
#
# gate checks every <proposals_dir>/NN-allowed_commands.json and
# NN-allowed_paths.json proposal: a JSON list of strings whose additions are
# all safe CANDIDATE additions and whose removals are all junk entries.
#
# Policy (the same for lint and gate):
#   commands — junk:  contains a newline, starts with "(" or "$(", or is a
#                     refusal message fragment ("... runs", "... is not ...")
#              risky: a lone shell, interpreter or file-changing command
#                     (bash, awk, python3, rm, chmod, curl ...): allows
#                     arbitrary code — report only, never proposed
#              safe addition: refused in the logs, not allowed yet, and
#                     either an existing executable of the user's own
#                     (under $HOME, not in a hidden dir), or one of SAFE_COMMANDS
#                     — unless it is a skill's own script or was refused only
#                     inside .awl tasks: those need a skill or prompt fix (NOTE)
#   paths    — junk:  not an absolute path, or does not exist
#              broad: /, /tmp, $HOME or a directory right under it, the
#                     config dirs — report only
#
# Secrets: from the .env files ai-assist loads (<source_dir>/.env, then
# <config_dir>/.env) only the
# AI_ASSIST_MODEL*, AI_ASSIST_QUERY_TIMEOUT and AI_ASSIST_PERMISSION_MODE
# settings are read; for event sources only the type
# and broker host are shown. Nothing else is printed or copied.
set -euo pipefail

mode="${1:?usage: $0 lint|gate <config_dir> <signals_jsonl> <proposals_dir> <source_dir>}"
cfg="${2:?usage: $0 lint|gate <config_dir> <signals_jsonl> <proposals_dir> <source_dir>}"
signals="${3:?usage: $0 lint|gate <config_dir> <signals_jsonl> <proposals_dir> <source_dir>}"
pdir="${4:?usage: $0 lint|gate <config_dir> <signals_jsonl> <proposals_dir> <source_dir>}"
src="${5:-}"
# shellcheck source=ai-assist-env.sh
. "$(dirname "${BASH_SOURCE[0]}")/ai-assist-env.sh"

case "$mode" in lint|gate) ;; *) echo "ERROR: unknown mode $mode" >&2; exit 2 ;; esac
[ -f "$signals" ] || { echo "ERROR: signals file not found: $signals" >&2; exit 2; }
mkdir -p "$pdir"

# only the non-secret settings of the .env files
env_files=$(printf '%s\n' "$src/.env" "$cfg/.env")
env_lines=$(for k in AI_ASSIST_MODEL AI_ASSIST_MODEL_LOW AI_ASSIST_MODEL_MEDIUM AI_ASSIST_MODEL_HIGH AI_ASSIST_QUERY_TIMEOUT AI_ASSIST_PERMISSION_MODE; do
  v=$(ai_assist_setting "$k" "$src" "$cfg"); [ -z "$v" ] || echo "$k=$v"
done)

exec python3 -I - "$mode" "$cfg" "$signals" "$pdir" "$env_files" "$env_lines" <<'EOF'
import glob
import json
import os
import sys
from collections import Counter

mode, cfg, signals_path, pdir, env_files, env_lines = sys.argv[1:7]
HOME = os.path.expanduser("~")
RISKY = {"bash", "sh", "zsh", "dash", "awk", "gawk", "sed", "perl", "python", "python3", "node", "ruby",
         "rm", "mv", "chmod", "chown", "dd", "curl", "wget", "ssh", "scp", "sudo", "eval", "xargs", "env", "tee"}
SAFE_COMMANDS = {"mkdir"}
BROAD_PATHS = {"/", "/tmp", HOME, cfg.rstrip("/"), os.path.join(HOME, ".config")}


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def command_class(entry):
    if not entry.strip() or "\n" in entry or entry.lstrip().startswith(("(", "$(")) \
            or " runs" in entry or " is not " in entry:
        return "junk"
    if entry.split()[0] in RISKY and len(entry.split()) == 1:
        return "risky"
    return "ok"


def path_class(entry):
    if not entry.startswith("/") or not os.path.exists(entry):
        return "junk"
    norm = os.path.normpath(entry)
    if norm in BROAD_PATHS or os.path.dirname(norm) == HOME:
        return "broad"
    return "ok"


def addition_class(cmd, allowed):
    """For a command refused in the logs: already, safe, risky or review."""
    if cmd in allowed:
        return "already"
    first = cmd.split()[0] if cmd.split() else cmd
    if cmd.startswith("builtin:") or first in RISKY or "(arbitrary code)" in cmd or "<" in cmd \
            or cmd.startswith(("other:", "find -exec")):
        return "risky"
    if cmd in SAFE_COMMANDS:
        return "safe"
    path = os.path.expanduser(cmd)
    # a script of the user's own (under $HOME, not in a hidden dir like ~/.local)
    if "/" in cmd and path.startswith(HOME + "/") and "/." not in path[len(HOME):] \
            and os.path.isfile(path) and os.access(path, os.X_OK):
        return "safe"
    return "review"


allowed_cmds = load_json(os.path.join(cfg, "allowed_commands.json"), [])
allowed_paths = load_json(os.path.join(cfg, "allowed_paths.json"), [])
signals = [json.loads(l) for l in open(signals_path) if l.strip()]
refused = Counter()
last_refused = {}
refused_in = {}  # command -> {"<script>::<task>"}
for s in signals:
    if s.get("kind") == "signal" and s.get("event") == "command_refused":
        refused[s["sample"]] += s["count"]
        last_refused[s["sample"]] = max(last_refused.get(s["sample"], ""), s.get("last", ""))
        refused_in.setdefault(s["sample"], set()).add(f"{s.get('script')}::{s.get('task')}")


def prompt_fix(cmd):
    """Why a refused script needs a prompt or skill fix, not a global entry, or None.

    A refusal inside an .awl task is fixed in that task (naming the script
    in backticks approves it for that task only); a skill's own script is
    fixed by the skill listing internal__execute_skill_script.
    """
    if "/skills/" in cmd and "/scripts/" in cmd:
        return "a skill's own script: its SKILL.md must list internal__execute_skill_script in allowed-tools"
    where = refused_in.get(cmd, set())
    if where and all(w.split("::")[0].endswith(".awl") for w in where):
        return "refused only inside .awl tasks: name it in backticks there (" + ", ".join(sorted(where)) + ")"
    return None


safe_additions = {c for c in refused if addition_class(c, allowed_cmds) == "safe" and not prompt_fix(c)}
junk_cmds = {c for c in allowed_cmds if command_class(c) == "junk"}
junk_paths = {p for p in allowed_paths if path_class(p) == "junk"}

if mode == "lint":
    n = 0
    env = dict(l.split("=", 1) for l in env_lines.splitlines() if "=" in l)
    default = env.get("AI_ASSIST_MODEL", "(unset)")
    levels = {lv: env.get(f"AI_ASSIST_MODEL_{lv.upper()}") for lv in ("low", "medium", "high")}
    unset = [lv for lv, m in levels.items() if not m]
    print(f"INFO models: default={default}; " + ", ".join(f"{lv}={m or 'unset -> default'}" for lv, m in levels.items()))
    if unset:
        print(f"INFO model levels {', '.join(unset)} fall back to the default model: tasks pinned to them run on {default}")
    print(f"INFO query timeout: {env.get('AI_ASSIST_QUERY_TIMEOUT', '600 (default)')} s; permission mode: {env.get('AI_ASSIST_PERMISSION_MODE', '(default)')}")
    for env_file in [f for f in env_files.splitlines() if os.path.isfile(f)]:
        readable = [p for p in allowed_paths if path_class(p) != "junk"
                    and os.path.abspath(env_file).startswith(os.path.normpath(p) + "/")]
        if readable:
            print(f"LINT allowed_paths.json: {readable[0]} makes {env_file} (API keys) readable by tasks")
            n += 1
    for c in sorted(junk_cmds):
        print(f"LINT allowed_commands.json: junk entry {json.dumps(c)} (left by an interactive approval): remove it")
        n += 1
    for c in sorted(c for c in allowed_cmds if command_class(c) == "risky"):
        print(f"NOTE allowed_commands.json: {json.dumps(c)} allows arbitrary code")
    for p in sorted(junk_paths):
        print(f"LINT allowed_paths.json: junk entry {json.dumps(p)} (not an existing absolute path): remove it")
        n += 1
    for p in sorted(p for p in allowed_paths if path_class(p) == "broad"):
        print(f"NOTE allowed_paths.json: broad entry {json.dumps(p)}")
    for c, count in refused.most_common():
        cls = addition_class(c, allowed_cmds)
        if cls == "safe" and prompt_fix(c):
            print(f"NOTE {json.dumps(c)} refused {count} time(s), last {last_refused[c][:16]}: {prompt_fix(c)}")
        elif cls == "safe":
            print(f"CANDIDATE allowed_commands.json add {json.dumps(c)}: refused {count} time(s), last {last_refused[c][:16]}")
            n += 1
        elif cls == "risky" and count >= 3:
            print(f"NOTE {json.dumps(c)} refused {count} time(s), last {last_refused[c][:16]}: needs a prompt fix, not an allowlist entry")
    # ai-assist splits a skill's allowed-tools on whitespace: with commas,
    # "internal__execute_skill_script," never matches and the skill cannot
    # run its own scripts
    for sk in load_json(os.path.join(cfg, "installed-skills.json"), {}).get("skills", []):
        md = os.path.join(sk.get("cache_path") or "", "SKILL.md")
        try:
            head = open(md).read().split("\n---", 1)[0]
        except OSError:
            continue
        for line in head.splitlines():
            if line.startswith("allowed-tools:") and "," in line:
                print(f"LINT skill {sk.get('name')}: allowed-tools in {md} is comma-separated; ai-assist splits it "
                      "on spaces, so its entries (e.g. internal__execute_skill_script) never match: use spaces")
                n += 1
    sources = load_json(os.path.join(cfg, "event-schedules.json"), {}).get("event_sources", {})
    for kind, conf in sorted(sources.items()):
        if conf:
            host = conf.get("broker") or conf.get("host") or ""
            print(f"INFO event source {kind} configured{(' (broker ' + host + ')') if host else ''}")
    errors = sum(s["count"] for s in signals if s.get("kind") == "signal" and s.get("event") == "error_log"
                 and "mqtt" in s.get("task", ""))
    if errors:
        print(f"LINT event source mqtt: {errors} connection errors in the window: fix the broker settings or disable it")
        n += 1
    skills = [{"name": sk.get("name"), "path": sk.get("cache_path")}
              for sk in load_json(os.path.join(cfg, "installed-skills.json"), {}).get("skills", [])]
    for name, data in (("allowed_commands.json", allowed_cmds), ("allowed_paths.json", allowed_paths),
                       ("installed-skills.json", skills)):
        with open(os.path.join(pdir, "live-" + name), "w") as f:
            json.dump(data, f, indent=2)
    print(f"config_issues={n}")
    sys.exit(0)

ok = rej = 0
for kind, live, junk, additions_ok in (("allowed_commands.json", allowed_cmds, junk_cmds, safe_additions),
                                        ("allowed_paths.json", allowed_paths, junk_paths, set())):
    for p in sorted(glob.glob(os.path.join(pdir, f"[0-9][0-9]-{kind}"))):
        name = os.path.basename(p)
        data = load_json(p, None)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            print(f"REJECT {name} is not a JSON list of strings")
            rej += 1
            continue
        added, removed = set(data) - set(live), set(live) - set(data)
        bad_add, bad_rm = sorted(added - additions_ok), sorted(removed - junk)
        if not added and not removed:
            print(f"REJECT {name} changes nothing")
            rej += 1
        elif bad_add or bad_rm:
            print(f"REJECT {name}: additions not allowed by the policy {bad_add}, removals of non-junk entries {bad_rm}")
            rej += 1
        else:
            print(f"OK {name} (adds {sorted(added)}, removes {sorted(removed)})")
            ok += 1
print(f"accepted_allowlist={ok}")
print(f"rejected_allowlist={rej}")
EOF
