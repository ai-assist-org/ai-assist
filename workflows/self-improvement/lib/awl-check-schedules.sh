#!/usr/bin/env bash
# awl-check-schedules.sh — deterministic checks on ai-assist scheduled actions.
#
# Usage:
#   awl-check-schedules.sh lint <config_dir> <proposals_dir> <source_dir>
#       Lint the live file and save a snapshot of its "actions" as
#       <proposals_dir>/live-event-schedules.json (the model cannot read the
#       config dir, but may read the proposals dir). The snapshot leaves out
#       "event_sources": it holds broker credentials.
#   awl-check-schedules.sh gate <config_dir> <proposals_dir> <source_dir>
#       Check every <proposals_dir>/NN-event-schedules.json proposal against the
#       live file: valid JSON, every action accepted by ai-assist's loader,
#       same action names, at least one action changed, and the changed actions
#       pass lint. Proposals only carry "actions": event_sources stay as they
#       are in the live file.
#
# Lint rules (per action):
#   * the prompt names a .awl script but does not end in ".awl": the scheduler
#     then runs it through a query capped at 600 s instead of directly
#   * the prompt is a .awl path that does not exist
#   * the action name is used more than once
#
# <source_dir> is ai-assist's source tree (the AWL built-in ${source_dir}): its
# python (see ai-assist-env.sh; override with AI_ASSIST_PYTHON) loads the file
# exactly as the scheduler does. Prints LINT / OK / REJECT lines and
# counters; exit 0 unless the arguments are wrong.
set -euo pipefail

mode="${1:?usage: $0 lint|gate <config_dir> <proposals_dir> <source_dir>}"
# the config dir, not the file: ai-assist refuses any command naming the
# file itself (protected config), even one that only reads it
live="${2:?usage: $0 lint|gate <config_dir> <proposals_dir> <source_dir>}/event-schedules.json"
pdir="${3:?usage: $0 lint|gate <config_dir> <proposals_dir> <source_dir>}"
src="${4:-}"
# shellcheck source=ai-assist-env.sh
. "$(dirname "${BASH_SOURCE[0]}")/ai-assist-env.sh"
py=$(ai_assist_python "$src" || true)

[ -f "$live" ] || { echo "ERROR: schedules file not found: $live" >&2; exit 2; }
"${py:-false}" -E -P -c "import ai_assist.awl_executor, ai_assist.action_model" 2>/dev/null \
  || { echo "ERROR: cannot find the python running ai-assist (got '$py'): set AI_ASSIST_PYTHON" >&2; exit 2; }
case "$mode" in lint|gate) ;; *) echo "ERROR: unknown mode $mode" >&2; exit 2 ;; esac
mkdir -p "$pdir"

exec "$py" -E -P - "$mode" "$live" "$pdir" <<'EOF'
import glob
import json
import os
import re
import sys
from pathlib import Path

from ai_assist.action_model import ActionDefinition

mode, live, pdir = sys.argv[1:4]
AWL_REF = re.compile(r"\S+\.awl\b")


def lint(action):
    """Issues for one action dict (as stored in the file)."""
    issues = []
    prompt = (action.get("prompt") or "").strip()
    ref = AWL_REF.search(prompt)
    if ref and not prompt.endswith(".awl"):
        issues.append(
            "prompt names " + os.path.basename(ref.group(0)) + " but does not end in .awl: it runs "
            "through a query capped at 600 s; use the bare .awl path and move arguments to prompt_arguments"
        )
    if prompt.endswith(".awl") and not Path(os.path.expanduser(prompt)).is_file():
        issues.append("script not found: " + prompt)
    return issues


def invalid(action):
    """Why ai-assist's loader would skip this action, or None.

    ActionLoader does not fail on an invalid action: it logs a warning and
    drops it, so the action silently stops running. Same check, explicit.
    """
    try:
        ActionDefinition.from_dict(action).validate_definition()
    except (KeyError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {a["name"]: a for a in data.get("actions", [])}, data


live_actions, live_data = load(live)

if mode == "lint":
    # snapshot for the model: actions only, never event_sources (credentials)
    with open(os.path.join(pdir, "live-event-schedules.json"), "w") as f:
        json.dump({"version": live_data.get("version"), "actions": live_data.get("actions", [])}, f, indent=2)
    names = [a.get("name") for a in live_data.get("actions", [])]
    n = 0
    for name in sorted({x for x in names if names.count(x) > 1}):
        print(f"LINT {name}: action name used {names.count(name)} times")
        n += 1
    for name, action in live_actions.items():
        why = invalid(action)
        if why:
            print(f"LINT {name}: skipped by ai-assist's loader ({why}): it never runs")
            n += 1
            continue
        if not action.get("enabled", True):
            continue
        for issue in lint(action):
            print(f"LINT {name}: {issue}")
            n += 1
    print(f"lint_issues={n}")
    print(f"snapshot={pdir}/live-event-schedules.json")
    sys.exit(0)

ok = rej = 0
for p in sorted(glob.glob(os.path.join(pdir, "[0-9][0-9]-event-schedules.json"))):
    name = os.path.basename(p)
    try:
        actions, _ = load(p)
    except Exception as e:  # invalid JSON or rejected by ActionLoader
        print(f"REJECT {name} does not load: {type(e).__name__}: {str(e)[:150]}")
        rej += 1
        continue
    if set(actions) != set(live_actions):
        added, removed = sorted(set(actions) - set(live_actions)), sorted(set(live_actions) - set(actions))
        print(f"REJECT {name} adds {added} / removes {removed} actions; only edits are allowed")
        rej += 1
        continue
    dropped = [f"{n} ({w})" for n, a in actions.items() if (w := invalid(a))]
    if dropped:
        print(f"REJECT {name} has actions ai-assist would skip: {dropped}")
        rej += 1
        continue
    changed = sorted(n for n in actions if actions[n] != live_actions[n])
    if not changed:
        print(f"REJECT {name} changes nothing")
        rej += 1
        continue
    left = [f"{n}: {i}" for n in changed for i in lint(actions[n])]
    if left:
        print(f"REJECT {name} still has lint issues: {left}")
        rej += 1
        continue
    print(f"OK {name} (changed: {', '.join(changed)})")
    ok += 1
print(f"accepted_config={ok}")
print(f"rejected_config={rej}")
EOF
