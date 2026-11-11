# Self-Improvement Workflow

`workflows/self-improvement/self-improvement.awl` reads ai-assist's own logs
and configuration and proposes fixes for your AWL workflows, skills, scheduled
actions, allowlists and ai-assist itself. It never changes anything: every
proposal waits for you to review and apply it.

## Running it

```bash
ai-assist /run <ai-assist source tree>/workflows/self-improvement/self-improvement.awl
ai-assist /run <ai-assist source tree>/workflows/self-improvement/self-improvement.awl days=14
```

`days` is the lookback window (default 7). To run it regularly, add a scheduled
action whose prompt is the bare `.awl` path.

## What it looks at

- **Workflows**: failed tasks, unresolved variables, refused commands and paths,
  tasks that run out of turns or never report, conditions that are never true,
  runs that hit the query time limit.
- **Skills**: skills that cannot run their own scripts (for example an
  `allowed-tools` field written with commas instead of spaces), descriptions
  over the size limit.
- **Scheduled actions**: failures, actions started twice, configuration
  mistakes (a prompt that names a `.awl` script but does not end with it runs
  through a time-limited query instead of directly).
- **Configuration**: leftover junk in `allowed_commands.json` and
  `allowed_paths.json`, entries that allow arbitrary code or expose `.env`,
  commands refused often enough to deserve a fix, failing event sources.
- **ai-assist**: repeated errors and tracebacks, written up as issue drafts.
- **Internal tools**: shell commands (`grep`, `sed`, `ls`, `jq`...) run where an
  internal tool exists. Using the tools' own descriptions, it tells habit (the
  tool already does it) from a missing feature, written up as a feature request.
- **Pipe tax**: what command outputs are piped into (`| head -20`,
  `| cut -c1-80`, `2>&1 | tail`…), whatever the command, and pipelines whose
  exit code 0 may hide a failure (a pipe reports its last command's exit code),
  read from the audit log. The model judges the candidates: a search for errors
  is not a hidden failure.
- **Security questions**: the commands and paths you had to approve, and the
  ones refused in workflows where nobody could answer. Each kind of question
  is classified as avoidable (an internal tool could have done it), a false
  positive (e.g. a `sed` program read as a path), a candidate for the
  allowlists, or legitimate.

The log extraction, counting and every check below are deterministic scripts;
the model only classifies findings and writes the proposals.

## Output

In `~/ai-assist/reports`:

| File | Content |
|------|---------|
| `self-improvement-report-<date>.md` | The report, ending with the full proposals |
| `self-improvement-<date>/` | Proposed `allowed_commands.json`, `allowed_paths.json` and `event-schedules.json` copies, ai-assist issue drafts |
| `self-improvement-signals-<date>.jsonl` | The counted log signals |
| `self-improvement-state.jsonl` | Signals already reported: the next run only reports new ones |
| `self-improvement-verdicts.jsonl` | Reviewers' verdicts on findings (see below) |

Text and configuration proposals are checked before they reach the report:

- a text change must quote text that exists, once, in the current file, and an
  edited workflow must still parse;
- a scheduled-actions copy must load in ai-assist and only edit existing actions;
- an allowlist copy may only remove junk entries and add commands the policy
  accepts.

## Applying proposals

- Text changes: edit the file as shown in the report.
- Issue drafts: review them before filing on the public ai-assist project.
- Allowlists: compare the proposed copy with the file in `~/.ai-assist`, copy it
  over and restart ai-assist.
- Scheduled actions: compare and copy over `~/.ai-assist/event-schedules.json`;
  the scheduler reloads it.

## Review and feedback

Each finding in the report has a signature (12 hex characters, stable across
runs). A reviewer, human or AI, records what a finding really was:

| Verdict | Meaning | Next runs |
|---------|---------|-----------|
| `real` | real, not fixed yet | listed under "Open findings", not reviewed again |
| `fixed` | the fix is in | a later occurrence is reported as a regression, ranked first |
| `wont_fix` | real but deliberate or accepted | no longer reported |
| `false_positive` | should never have been a finding | no longer reported; shown to the classifier as a counter-example |

Record a verdict with the command printed in the report's "Review" section:

```bash
bash workflows/self-improvement/lib/awl-verdict.sh \
  ~/ai-assist/reports/self-improvement-state.jsonl \
  ~/ai-assist/reports/self-improvement-verdicts.jsonl \
  4436a75fbc3e false_positive "the refusal is intended" human:alice
```

or install the `self-improvement-review` skill and tell the agent ("finding 3
is intended, not a problem"):

```
/skill/install ./skills/self-improvement-review@main
```

Shell forms an internal tool could run (e.g. `grep -i` instead of
`internal__search_in_file`) and kinds of security questions have signatures
too, shown in their report tables: marking one `fixed` after improving a tool
tells you on the next runs whether agents really stopped using the shell.

Verdicts are appended, never rewritten: the latest one for a signature wins and
the history stays. The report's precision table (reviewed / real / false
positive per kind of finding) shows which detection rules are noisy.
