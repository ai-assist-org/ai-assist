---
name: self-improvement-review
description: Record a review verdict on a finding of the self-improvement report (real, fixed, wont_fix or false_positive), so the next runs stop re-reporting dismissed findings, flag regressions and measure which detection rules are noisy. Use when the user (or a reviewing agent) says a finding of a self-improvement report "is not a real problem", "is fixed", "is intended", "is a false positive", "is confirmed", or asks to review the report's findings.
argument-hint: <finding id or signature> <verdict> <why>
---

# Self-improvement review

The self-improvement workflow (`workflows/self-improvement/self-improvement.awl`)
reports findings mined from ai-assist's logs. A verdict tells the next runs what
a reviewer concluded about one of them. Verdicts are only ever written by the
workflow's `awl-verdict.sh` script: never edit the JSONL files yourself.

## 1. Find the finding

- The report is `self-improvement-report-<date>.md` in the reports directory
  (the latest one unless the user names another). Read it. The reports
  directory is `~/ai-assist/reports`, unless `AI_ASSIST_REPORTS_DIR` is set in
  the environment or in the instance's `.env` file (e.g. `~/.ai-assist/.env`);
  when several ai-assist instances exist, ask the user which one.
- Each finding in its "Ranked fixes" table has an id (1..n, only valid in that
  report) and a **signature** (12 hex characters, stable across runs). A
  verdict is recorded on the signature: map the id the user gives to it.
- Rows of the "Shell commands instead of internal tools" and "Security
  questions" tables have their own signature (one per shell form or question
  form, e.g. `grep -i` replaced by `internal__search_in_file`). A finding built
  from several of them lists their signatures: record a verdict on each.
- The report's "## Review" section gives the exact command to run, with this
  installation's paths. Use it as is; only fill in the placeholders.

## 2. Choose the verdict

| Verdict | When |
|---|---|
| `real` | A real problem that is not fixed yet. It stays on the report's "Open" list and is not re-classified. |
| `fixed` | The fix is in (a commit, a config or `.awl` change). A later occurrence is reported as a regression. |
| `wont_fix` | Real, but deliberate or accepted (e.g. a refusal the user wants). Suppressed from now on. |
| `false_positive` | Should never have been a finding (the detection rule is wrong for it). Suppressed, and counted against its rule. |

- Reviewing as an agent: check the evidence before deciding (read the log
  sample, the `.awl` task or the code). Never record `fixed` without pointing
  to the change; never guess `false_positive` because a finding looks odd.
- The note is required and is shown to the next runs' classifier: write the
  reason in one sentence ("refusal is intended: the task must not run curl").
- When the user's intent is ambiguous (e.g. "this one is fine" could be
  `wont_fix` or `false_positive`), ask before recording.

## 3. Record it

Run the command from the report's "## Review" section:

```
bash <lib>/awl-verdict.sh <ledger> <verdicts> <signature> <verdict> "<why>" <reviewer>
```

- `<reviewer>`: `human:<name>` when recording what the user decided,
  `agent:<your name>` for your own review.
- It refuses an unknown signature, an unknown verdict or an empty note: fix
  the input instead of working around it.
- Recording again for the same signature replaces the verdict (the latest
  wins); the history is kept.

Confirm to the user what was recorded: finding id, signature, verdict, note.
