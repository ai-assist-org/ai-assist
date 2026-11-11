#!/usr/bin/env bash
# awl-signals-diff.sh — keep only findings not already seen or decided.
#
# Usage:
#   awl-signals-diff.sh <signals_jsonl> <state_jsonl> [--record-state]
#                       [--verdicts <verdicts_jsonl>] [--review-out <review_jsonl>]
#
# <signals_jsonl> : output of awl-signals-aggregate.sh
# <state_jsonl>   : ledger of already-reported signatures (created if absent)
# --record-state  : append the NEW signatures to <state_jsonl> (deterministic;
#                   the AWL task does not have to copy JSON lines itself)
# --verdicts      : reviewers' verdicts (see awl-verdict.sh); the latest per
#                   signature wins. Missing file = no verdicts.
# --review-out    : write the records the classifier must review: the new
#                   signals (kind "signal"), the rolled-up records (kinds
#                   "shell", "prompt" and "pipe"), each with "regression" and
#                   "prior_verdict", and up to 10 past dismissals with their
#                   notes (kind "counter_example")
#
# A signal is new when its signature is absent from the ledger, or when it
# occurred again after the latest occurrence the ledger recorded for it (a
# problem that came back, e.g. after a fix that did not work). Then the
# latest verdict decides:
#   none                     -> new
#   false_positive, wont_fix -> suppressed
#   fixed                    -> regression (new) if it occurred after the
#                               verdict, else suppressed
#   real                     -> open: listed, never re-reviewed
#
# Rolled-up records (kind "shell": shell forms an internal tool could run,
# kind "prompt": security questions, kind "pipe": what command outputs are
# piped into) have signatures too. They never count as
# new signals: all of them go to the review file, except the dismissed ones
# (suppressed), the confirmed ones (open), and the fixed ones not seen since
# the verdict; a fixed one seen again is a regression. --record-state adds
# the ones absent from the ledger, so a reviewer can record a verdict on them.
#
# Idempotency contract: running twice with unchanged inputs yields
# new_signals=0 on the second run.

set -euo pipefail

usage="usage: $0 <signals_jsonl> <state_jsonl> [--record-state] [--verdicts FILE] [--review-out FILE]"
signals="${1:?$usage}"
state="${2:?$usage}"
shift 2
record=""
verdicts=""
review_out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --record-state) record=1 ;;
    --verdicts) verdicts="${2:?--verdicts needs a file}"; shift ;;
    --review-out) review_out="${2:?--review-out needs a file}"; shift ;;
    *) echo "ERROR: unknown option $1 ($usage)" >&2; exit 2 ;;
  esac
  shift
done

[ -f "$signals" ] || { echo "ERROR: signals file not found: $signals" >&2; exit 2; }
touch "$state"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# known signatures with their latest recorded occurrence: "sig<TAB>last"
jq -r 'select(.signature) | [.signature, (.last // "")] | @tsv' "$state" 2>/dev/null \
  | sort -t$'\t' -k1,1 -k2,2r | sort -t$'\t' -u -k1,1 > "$tmp/known" || true
known_count=$(wc -l < "$tmp/known")

# latest verdict per signature: "sig<TAB>verdict<TAB>at<TAB>note"
touch "$tmp/verdicts"
if [ -n "$verdicts" ] && [ -f "$verdicts" ]; then
  jq -r 'select(.signature and .verdict) | [.signature, .verdict, (.at // ""), (.note // "")] | @tsv' \
    "$verdicts" 2>/dev/null \
    | awk -F'\t' '{ v[$1] = $0 } END { for (s in v) print v[s] }' | sort > "$tmp/verdicts" || true
fi

total=$(jq -r 'select(.kind=="signal")|.signature' "$signals" | wc -l)

lookup() {  # lookup <file> <signature> -> the line's other fields, or ABSENT
  awk -F'\t' -v s="$2" '$1 == s { sub(/^[^\t]*\t/, ""); print; found = 1 } END { if (!found) print "ABSENT" }' "$1"
}

regressions=0; open=0; suppressed=0
: > "$tmp/new"; : > "$tmp/open"; : > "$tmp/rollups"; : > "$tmp/rollups_unknown"
while IFS= read -r obj; do
  IFS=$'\t' read -r sig last kind < <(printf '%s' "$obj" | jq -r '[.signature, (.last // "-"), .kind] | @tsv')
  [ "$last" != "-" ] || last=""
  recorded=$(lookup "$tmp/known" "$sig")
  is_new=""
  if [ "$recorded" = "ABSENT" ] || [[ "$last" > "$recorded" ]]; then is_new=1; fi
  v=$(lookup "$tmp/verdicts" "$sig")
  verdict=""; at=""; note=""
  if [ "$v" != "ABSENT" ]; then IFS=$'\t' read -r verdict at note <<< "$v"; fi

  if [ "$kind" != "signal" ]; then  # a rolled-up record
    [ "$recorded" != "ABSENT" ] || printf '%s\n' "$obj" >> "$tmp/rollups_unknown"
    case "$verdict" in
      real)
        open=$((open + 1)); printf '%s\t%s\n' "$obj" "$note" >> "$tmp/open" ;;
      false_positive|wont_fix)
        suppressed=$((suppressed + 1)) ;;
      fixed)
        if [[ "${last:0:19}" > "${at:0:19}" ]]; then
          regressions=$((regressions + 1))
          printf '%s' "$obj" | jq -c '. + {regression: true, prior_verdict: "fixed"}' >> "$tmp/rollups"
        else
          suppressed=$((suppressed + 1))
        fi ;;
      *)
        printf '%s' "$obj" | jq -c '. + {regression: false, prior_verdict: null}' >> "$tmp/rollups" ;;
    esac
    continue
  fi

  case "$verdict" in
    real)
      open=$((open + 1))
      printf '%s\t%s\n' "$obj" "$note" >> "$tmp/open" ;;
    false_positive|wont_fix)
      [ -z "$is_new" ] || suppressed=$((suppressed + 1)) ;;
    fixed)
      if [ -n "$is_new" ] && [[ "${last:0:19}" > "${at:0:19}" ]]; then
        regressions=$((regressions + 1))
        printf '%s' "$obj" | jq -c '. + {regression: true, prior_verdict: "fixed"}' >> "$tmp/new"
      elif [ -n "$is_new" ]; then
        suppressed=$((suppressed + 1))
      fi ;;
    *)
      [ -z "$is_new" ] || printf '%s' "$obj" | jq -c '. + {regression: false, prior_verdict: null}' >> "$tmp/new" ;;
  esac
done < <(jq -c 'select(.signature and (.kind == "signal" or .kind == "shell" or .kind == "prompt" or .kind == "pipe"))' "$signals")

new_count=$(wc -l < "$tmp/new")

if [ "$record" = "1" ]; then
  # an explicit format: BSD date (macOS) has no -I option
  now=$(date '+%Y-%m-%dT%H:%M:%S%z')
  jq -c --arg now "$now" \
    '{signature, kind, script, task, event, count, first, last, sample,
      first_seen: $now, report_file: null, accepted: "unknown"}' "$tmp/new" "$tmp/rollups_unknown" >> "$state"
fi

# one event per signature, from the ledger then the current signals
{ jq -r 'select(.signature and .event) | [.signature, .event, (.script // ""), (.task // ""), (.sample // "")] | @tsv' \
    "$state" 2>/dev/null || true
  jq -r 'select(.signature and .event) | [.signature, .event, .script, .task, (.sample // "")] | @tsv' "$signals"
} | awk -F'\t' '!seen[$1]++' | sort > "$tmp/events"

if [ -n "$review_out" ]; then
  mkdir -p "$(dirname "$review_out")"
  {
    cat "$tmp/new" "$tmp/rollups"
    # the 10 most recent dismissals, as counter-examples for the classifier
    join -t$'\t' "$tmp/verdicts" "$tmp/events" \
      | awk -F'\t' '$2 == "false_positive" || $2 == "wont_fix"' \
      | sort -t$'\t' -k3,3r | head -10 \
      | jq -R -c 'split("\t") | {kind: "counter_example", signature: .[0], verdict: .[1], at: .[2],
                  note: .[3], event: .[4], script: .[5], task: .[6], sample: .[7]}'
  } > "$review_out"
fi

echo "known_signatures=$known_count"
echo "total_signals=$total"
echo "new_signals=$new_count"
echo "regressions=$regressions"
echo "open=$open"
echo "suppressed=$suppressed"
echo "state_file=$state"
echo
echo "=== new signals ==="
if [ "$new_count" -eq 0 ]; then
  echo "(none — nothing changed since the last run)"
else
  jq -r '[.count, .script, .task, (.event + (if .regression then " (REGRESSION)" else "" end)), .signature] | @tsv' \
    "$tmp/new" |
    sort -rn -k1,1 |
    awk -F'\t' 'BEGIN{printf "%-6s %-24s %-22s %-17s %s\n","COUNT","SCRIPT","TASK","EVENT","SIG"}{printf "%-6s %-24s %-22s %-17s %s\n",$1,$2,$3,$4,$5}'
fi

echo
echo "=== open (confirmed real, not fixed) ==="
if [ "$open" -eq 0 ]; then
  echo "(none)"
else
  while IFS=$'\t' read -r obj note; do
    printf '%s' "$obj" | jq -r --arg note "$note" '"\(.signature) \(.script)::\(.task) \(.event) count=\(.count) last=\(.last) — \($note)"'
  done < "$tmp/open"
fi

echo
echo "=== precision by event (reviewed findings: real = real + fixed + wont_fix) ==="
join -t$'\t' "$tmp/verdicts" "$tmp/events" \
  | awk -F'\t' '{ ev = $5; n[ev]++; if ($2 == "false_positive") fp[ev]++; else real[ev]++ }
                END { for (e in n) printf "PRECISION %s reviewed=%d real=%d false_positive=%d\n", e, n[e], real[e], fp[e] }' \
  | sort
