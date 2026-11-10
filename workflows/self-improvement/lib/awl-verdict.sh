#!/usr/bin/env bash
# awl-verdict.sh — record a reviewer's verdict on a self-improvement finding.
#
# Usage:
#   awl-verdict.sh <state_jsonl> <verdicts_jsonl> <signature> <verdict> <note> [reviewer]
#
# <signature> : the finding's signature (12 hex characters, shown in the
#               report and stored in the ledger <state_jsonl>)
# <verdict>   : real            a real problem, not fixed yet
#               fixed           the fix is in (a later occurrence is a regression)
#               wont_fix        real but deliberate or accepted
#               false_positive  should never have been a finding
# <note>      : why (required: later runs show it to the classifier)
# [reviewer]  : who decided, e.g. "human:alice" or "agent:iris" (default: unspecified)
#
# Appends one JSON line {signature, verdict, reviewer, note, at} to
# <verdicts_jsonl>; the latest line for a signature wins. Never rewrites it.
# Exit codes: 0 recorded, 2 invalid input.

set -euo pipefail

usage="usage: $0 <state_jsonl> <verdicts_jsonl> <signature> <verdict> <note> [reviewer]"
state="${1:?$usage}"
verdicts="${2:?$usage}"
signature="${3:?$usage}"
verdict="${4:?$usage}"
note="${5:-}"
reviewer="${6:-unspecified}"

fail() { echo "ERROR: $*" >&2; exit 2; }

case "$verdict" in
  real|fixed|wont_fix|false_positive) ;;
  *) fail "verdict must be one of: real, fixed, wont_fix, false_positive (got '$verdict')" ;;
esac
[ -n "$note" ] || fail "a note explaining the verdict is required"
[ -f "$state" ] || fail "ledger not found: $state"
jq -e --arg s "$signature" 'select(.signature == $s)' "$state" > /dev/null 2>&1 \
  || fail "signature $signature is not in the ledger $state"

# local wall-clock time, comparable with the "last" occurrence of signals
jq -cn --arg s "$signature" --arg v "$verdict" --arg r "$reviewer" --arg n "$note" \
  --arg at "$(date '+%Y-%m-%dT%H:%M:%S')" \
  '{signature: $s, verdict: $v, reviewer: $r, note: $n, at: $at}' >> "$verdicts"
echo "recorded $verdict for $signature"
