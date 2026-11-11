#!/usr/bin/env bash
# awl-check-excerpts.sh — gate for the text changes proposed in the report,
# and assembly of the final report.
#
# Usage:
#   awl-check-excerpts.sh check <proposals_md> <signals_jsonl> <config_dir> <source_dir>
#   awl-check-excerpts.sh append <proposals_md> <report_md>
#
# A proposed change is a "**File**: <absolute path>" line
# followed by a ```diff block of lines starting with " " (unchanged), "-"
# (removed) or "+" (added); "@@" separates hunks. Changes are proposals in
# a report, not patches: nothing is applied.
#
# check: the file must be a workflow seen in the logs (a "path" of the
# script records in <signals_jsonl>) or a file inside an installed skill
# (<config_dir>/installed-skills.json). For every change, the "before" text of each hunk (unchanged +
# removed lines) must exist verbatim, as consecutive lines, in the current
# file, exactly once (trailing spaces ignored; when only the indentation
# differs, a common model slip, it is accepted and reported), and the hunk
# must change something. For a
# .awl file, the edited copy (made in a temp dir) must still parse with
# ai-assist's AWL parser (from <source_dir>, ai-assist's source tree: the AWL
# built-in ${source_dir}). Prints OK/REJECT per change, then
# accepted_changes= / rejected_changes=.
#
# append: appends <proposals_md> to <report_md> verbatim, so the proposals
# in the report are exactly the ones the gate checked.
set -euo pipefail

usage="usage: $0 check <proposals_md> <signals_jsonl> <config_dir> <source_dir> | append <proposals_md> <report_md>"
mode="${1:?$usage}"
pmd="${2:?$usage}"
dest="${3:?$usage}"
cfg="${4:-}"
src="${5:-}"
# shellcheck source=ai-assist-env.sh
. "$(dirname "${BASH_SOURCE[0]}")/ai-assist-env.sh"
py=$(ai_assist_python "$src" || true)

case "$mode" in
  append)
    [ -f "$pmd" ] || { echo "appended=none (no proposals)"; exit 0; }
    [ -f "$dest" ] || { echo "ERROR: report not found: $dest" >&2; exit 2; }
    { printf '\n'; cat "$pmd"; } >> "$dest"
    echo "appended=$pmd"
    exit 0 ;;
  check) ;;
  *) echo "ERROR: unknown mode $mode" >&2; exit 2 ;;
esac
if [ ! -f "$pmd" ]; then
  echo "accepted_changes=0"; echo "rejected_changes=0"; exit 0
fi
"${py:-false}" -E -P -c "import ai_assist.awl_executor, ai_assist.action_model" 2>/dev/null \
  || { echo "ERROR: cannot find the python running ai-assist (got '$py'): set AI_ASSIST_PYTHON" >&2; exit 2; }

exec "$py" -E -P - "$pmd" "$dest" "$cfg" <<'EOF'
import os
import re
import sys
import tempfile

from ai_assist.awl_executor import load_awl_workflow

import json

pmd, signals_path, cfg = sys.argv[1:4]


def allowed_targets():
    """Workflow files seen in the logs, and installed skill directories."""
    files, dirs = set(), []
    for line in open(signals_path):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("kind") == "script" and rec.get("path"):
            files.add(os.path.realpath(os.path.expanduser(rec["path"])))
    try:
        skills = json.load(open(os.path.join(cfg, "installed-skills.json"))).get("skills", [])
    except (OSError, ValueError):
        skills = []
    for sk in skills:
        if sk.get("cache_path"):
            dirs.append(os.path.realpath(os.path.expanduser(sk["cache_path"])) + "/")
    return files, dirs


ALLOWED_FILES, SKILL_DIRS = allowed_targets()


def changes(lines):
    """Yield (finding, file, [hunk...]); a hunk is a list of (marker, text)."""
    finding, path, i = "?", None, 0
    while i < len(lines):
        line = lines[i]
        if m := re.match(r"##\s+(?:Finding\s+)?(\d+)\b", line):
            finding, path = m.group(1), None
        elif m := re.match(r"\*\*File\*\*:\s*(\S+)", line):
            path = m.group(1).strip("`'\"")
        elif line.startswith("```diff"):
            hunks, cur, i = [], [], i + 1
            while i < len(lines) and not lines[i].startswith("```"):
                body = lines[i]
                if body.startswith("@@"):
                    if cur:
                        hunks.append(cur)
                    cur = []
                elif body == "":
                    cur.append((" ", ""))
                else:
                    cur.append((body[0], body[1:]))
                i += 1
            if cur:
                hunks.append(cur)
            yield finding, path, hunks
        i += 1


def find_all(haystack, needle):
    return [k for k in range(len(haystack) - len(needle) + 1) if haystack[k:k + len(needle)] == needle]


def apply(text, k, hunk):
    """The lines replacing text[k:...] for this hunk: unchanged lines keep the
    file's own text (so an indentation slip in the excerpt is not copied)."""
    out, j = [], k
    for mk, t in hunk:
        if mk == " ":
            out.append(text[j]); j += 1
        elif mk == "-":
            j += 1
        else:
            out.append(t.rstrip())
    return out, j


ok = rej = 0
for finding, path, hunks in changes(open(pmd).read().splitlines()):
    label = f"#{finding} {path or '(no **File** line)'}"
    target = os.path.realpath(os.path.expanduser(path)) if path else None
    why = None
    if not target or not os.path.isfile(target) \
            or not (target in ALLOWED_FILES or any(target.startswith(d) for d in SKILL_DIRS)):
        why = "target is neither a workflow seen in the logs nor a file of an installed skill"
    elif not hunks:
        why = "empty diff block"
    elif any(mk not in " +-" for h in hunks for mk, _ in h):
        why = "diff lines must start with ' ', '-' or '+'"
    notes = []
    if why is None:
        text = [l.rstrip() for l in open(target).read().splitlines()]
        for n, h in enumerate(hunks, 1):
            before = [t.rstrip() for mk, t in h if mk in " -"]
            after = [t.rstrip() for mk, t in h if mk in " +"]
            if before == after:
                why = f"hunk {n} changes nothing"
                break
            # models often slip on leading spaces (e.g. drop the diff marker's
            # space on unchanged lines): fall back to ignoring indentation
            where = find_all(text, before) if before else []
            fuzzy = False
            if not where and before:
                where = find_all([l.strip() for l in text], [b.strip() for b in before])
                fuzzy = bool(where)
            if not where:
                why = f"hunk {n}: the before text is not in the current file (stale or invented)"
                break
            if len(where) > 1:
                why = f"hunk {n}: the before text appears {len(where)} times: add context lines to make it unique"
                break
            if fuzzy:
                notes.append(f"hunk {n}: indentation differs from the file, check it when applying")
            new, end = apply(text, where[0], h)
            text[where[0]:end] = new
    if why is None and target.endswith(".awl"):
        with tempfile.TemporaryDirectory() as d:
            copy = os.path.join(d, os.path.basename(target))
            with open(copy, "w") as f:
                f.write("\n".join(text) + "\n")
            try:
                load_awl_workflow(copy)
            except Exception as e:
                why = f"the edited script no longer parses: {type(e).__name__}: {str(e)[:150]}"
    if why:
        print(f"REJECT {label}: {why}")
        rej += 1
    else:
        print(f"OK {label} ({len(hunks)} hunk(s){'; ' + ', '.join(notes) if notes else ''})")
        ok += 1
print(f"accepted_changes={ok}")
print(f"rejected_changes={rej}")
EOF
