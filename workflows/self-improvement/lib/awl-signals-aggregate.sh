#!/usr/bin/env bash
# awl-signals-aggregate.sh — deterministic AWL workflow-health scorecard.
#
# Usage:
#   awl-signals-aggregate.sh <days> <out_jsonl> [logs_dir] [tmp_out_tsv] [source_dir] [config_dir]
#
# source_dir / config_dir: ai-assist's source tree (the AWL built-in
# ${source_dir}) and config dir, to read AI_ASSIST_QUERY_TIMEOUT and the
# audit log (<config_dir>/audit/tool_calls.jsonl).
#
# Steps (all deterministic, no LLM):
#   1. select ai-assist.log* files modified within <days> days
#   2. extract signals with lib/extract-awl-signals.awk   -> TSV
#      and the security questions of the last <days> days from the audit
#      log (lib/audit-prompts.py, lib/extract-prompts.awk) -> same TSV
#   3. aggregate with lib/aggregate-awl-signals.awk       -> JSON lines
#   4. hash each stable signature key (sha1, 12 hex chars)
#   5. write JSONL to <out_jsonl>; print a top-10 table on stdout
#
# Exit codes: 0 ok, 2 no log files matched.
set -euo pipefail

days="${1:-7}"
# Guard against an unresolved ${days} reaching us as a literal string.
case "$days" in
  ''|*[!0-9]*) days=7 ;;
esac
[ "$days" -ge 1 ] 2>/dev/null || days=7
out="${2:?usage: $0 <days> <out_jsonl> [logs_dir] [tmp_out_tsv]}"
logs_dir="${3:-$HOME/.ai-assist/logs}"
tsv_out="${4:-/tmp/awl-signals-$$.tsv}"
src="${5:-}"
cfg="${6:-$HOME/.ai-assist}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$logs_dir" ]; then
  echo "ERROR: logs dir not found: $logs_dir" >&2
  exit 2
fi

# 1. select files. The rotating file is named ai-assist.log (no suffix) and
#    is always current, so include it explicitly. (a while-read loop:
#    macOS ships bash 3.2)
files=()
while IFS= read -r f; do files+=("$f"); done < <(
  { find "$logs_dir" -maxdepth 1 -type f -name 'ai-assist.log*' -mtime "-${days}" 2>/dev/null;
    [ -f "$logs_dir/ai-assist.log" ] && echo "$logs_dir/ai-assist.log";
  } | sort -u
)

if [ "${#files[@]}" -eq 0 ]; then
  echo "ERROR: no ai-assist.log* files matched days=$days in $logs_dir" >&2
  exit 2
fi

# 2. snapshot: the live ai-assist.log grows while we read it, and rotation
#    can move files mid-run. Copy first so extraction is self-consistent.
snap_dir=$(mktemp -d)
raw_out=""
trap 'rm -rf "$snap_dir" "$raw_out"' EXIT
for f in "${files[@]}"; do
  cp "$f" "$snap_dir/" 2>/dev/null || true
done
audit="$cfg/audit/tool_calls.jsonl"
if [ -f "$audit" ]; then cp "$audit" "$snap_dir/" 2>/dev/null || true; fi
# chronological order: dated files (ai-assist.log.YYYY-MM-DD) first, the
# current ai-assist.log last, so time-based signals (duplicate starts,
# reload bursts) never see time going backwards between files.
files=()
while IFS= read -r f; do files+=("$f"); done < <(
  find "$snap_dir" -maxdepth 1 -type f -name 'ai-assist.log.*' | sort
  [ -f "$snap_dir/ai-assist.log" ] && echo "$snap_dir/ai-assist.log"
)

# 3. extract (single awk pass over all files, state carried per PID)
# query limits for near_limit: the default 600 s and AI_ASSIST_QUERY_TIMEOUT
# from ai-assist's .env files (only that line is read: the file holds secrets)
# shellcheck source=ai-assist-env.sh
. "$here/ai-assist-env.sh"
configured=$(ai_assist_setting AI_ASSIST_QUERY_TIMEOUT "$src" "$cfg")
[[ "$configured" =~ ^[0-9]+$ ]] || configured=""
awk -v query_limits="600 ${configured}" -f "$here/signals-common.awk" -f "$here/extract-awl-signals.awk" \
  "${files[@]}" > "$tsv_out"

# security questions: the audit log records each one with its answer and the
# AWL script/task that asked
if [ -f "$snap_dir/tool_calls.jsonl" ]; then
  py=$(ai_assist_python "$src" || true)
  "${py:-python3}" -E -P "$here/audit-prompts.py" "$snap_dir/tool_calls.jsonl" "$days" \
    | awk -f "$here/signals-common.awk" -f "$here/extract-prompts.awk" >> "$tsv_out"
fi

raw_rows=$(wc -l < "$tsv_out")

# 3+4. aggregate and hash signatures
raw_out=$(mktemp)
awk -f "$here/aggregate-awl-signals.awk" "$tsv_out" > "$raw_out"

# signal records: keep awk grouping order (deterministic); hash the key
grep -v '^-' "$raw_out" |
while IFS=$'\t' read -r tag json sigkey; do
  h=$(printf '%s' "$sigkey" | sha1sum | cut -c1-12)
  printf '%s\n' "${json//@SIG@/$h}"
done > "$out"

# script rollups: awk `for (s in seen)` order is unspecified -> sort.
grep '^-' "$raw_out" | sort | cut -f2- >> "$out"

# 5. report
n_files="${#files[@]}"
n_signals=$(grep -c '"kind":"signal"' "$out" || true)
n_scripts=$(grep -c '"kind":"script"' "$out" || true)

echo "scanned_files=$n_files"
echo "log_files=$(printf '%s ' "${files[@]##*/}")"
echo "raw_signal_rows=$raw_rows"
echo "distinct_signals=$n_signals"
echo "scripts_seen=$n_scripts"
echo "output=$out"
echo "tsv=$tsv_out"
echo
echo "=== top signals by occurrences ==="
grep '"kind":"signal"' "$out" \
  | sed -E 's/.*"signature":"([^"]*)".*"script":"([^"]*)","task":"([^"]*)","event":"([^"]*)","count":([0-9]+).*"mechanical":(true|false).*/\5\t\2\t\3\t\4\t\6\t\1/' \
  | sort -rn -k1,1 \
  | head -10 \
  | awk -F'\t' 'BEGIN{printf "%-6s %-24s %-22s %-17s %-6s %s\n","COUNT","SCRIPT","TASK","EVENT","MECH","SIG"}{printf "%-6s %-24s %-22s %-17s %-6s %s\n",$1,$2,$3,$4,$5,$6}'
