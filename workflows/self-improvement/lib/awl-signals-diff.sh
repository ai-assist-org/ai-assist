#!/usr/bin/env bash
# awl-signals-diff.sh — keep only findings not already seen.
#
# Usage:
#   awl-signals-diff.sh <signals_jsonl> <state_jsonl> [--record-state]
#
# <signals_jsonl> : output of awl-signals-aggregate.sh
# <state_jsonl>   : ledger of already-reported signatures (created if absent)
# --record-state  : append the NEW signatures to <state_jsonl> (deterministic;
#                   the AWL task does not have to copy JSON lines itself)
#
# stdout: summary counters + the new signal objects (JSONL) when NEW_ONLY=1
#         (default 0: only counters, kept for a readable report section)
#
# A signal is new when its signature is absent from the ledger, or when it
# occurred again after the latest occurrence the ledger recorded for it (a
# problem that came back, e.g. after a fix that did not work).
#
# Idempotency contract: running twice with unchanged inputs yields
# new_signals=0 on the second run.

set -euo pipefail

signals="${1:?usage: $0 <signals_jsonl> <state_jsonl> [--record-state]}"
state="${2:?usage: $0 <signals_jsonl> <state_jsonl> [--record-state]}"
record="${3:-}"

[ -f "$signals" ] || { echo "ERROR: signals file not found: $signals" >&2; exit 2; }
touch "$state"

# known signatures with their latest recorded occurrence: "sig<TAB>last"
known=$(mktemp)
trap 'rm -f "$known"' EXIT
jq -r 'select(.signature) | [.signature, (.last // "")] | @tsv' "$state" 2>/dev/null \
  | sort -t$'\t' -k1,1 -k2,2r | sort -t$'\t' -u -k1,1 > "$known" || true
known_count=$(wc -l < "$known")

total=$(jq -r 'select(.kind=="signal")|.signature' "$signals" | wc -l)

# new = absent from the ledger, or seen again after its recorded last
new_file=$(mktemp)
jq -c 'select(.kind=="signal")' "$signals" |
while IFS= read -r obj; do
  sig=$(printf '%s' "$obj" | jq -r '.signature')
  last=$(printf '%s' "$obj" | jq -r '.last // ""')
  recorded=$(awk -F'\t' -v s="$sig" '$1 == s { print $2; found = 1 } END { if (!found) print "ABSENT" }' "$known")
  if [ "$recorded" = "ABSENT" ] || [[ "$last" > "$recorded" ]]; then
    printf '%s\n' "$obj"
  fi
done > "$new_file"

new_count=$(wc -l < "$new_file")

if [ "$record" = "--record-state" ] && [ "$new_count" -gt 0 ]; then
  now=$(date -Iseconds)
  while IFS= read -r obj; do
    printf '%s' "$obj" | jq -c --arg now "$now" \
      '{signature, script, task, event, count, first, last, sample,
        first_seen: $now, report_file: null, accepted: "unknown"}' >> "$state"
  done < "$new_file"
fi

echo "known_signatures=$known_count"
echo "total_signals=$total"
echo "new_signals=$new_count"
echo "state_file=$state"
echo
echo "=== new signals ==="
if [ "$new_count" -eq 0 ]; then
  echo "(none — nothing changed since the last run)"
else
  jq -r '[.count, .script, .task, .event, .signature] | @tsv' "$new_file" |
    sort -rn -k1,1 |
    awk -F'\t' 'BEGIN{printf "%-6s %-24s %-22s %-17s %s\n","COUNT","SCRIPT","TASK","EVENT","SIG"}{printf "%-6s %-24s %-22s %-17s %s\n",$1,$2,$3,$4,$5}'
fi

rm -f "$new_file"
