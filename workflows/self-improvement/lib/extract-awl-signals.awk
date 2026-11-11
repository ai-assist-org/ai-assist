# extract-awl-signals.awk — deterministic extraction of AWL workflow health
# signals from ~/.ai-assist/logs/ai-assist.log*.
#
# Output: TSV, one row per signal:
#   date \t time \t pid \t script \t task \t event \t detail
#
# Besides AWL scripts, two pseudo-scripts group signals that are not tied to
# a .awl file: "scheduler" (scheduled actions; task = action name) and
# "ai-assist" (ai-assist's own errors; task = exception or component).
#
# Attribution model: log lines from nested AWL executions nest their lock
# windows (parent.awl -> child.awl -> parent.awl), so we keep a
# per-PID STACK of scripts. The script for a task line is the top of that
# PID's stack. Task lines themselves never carry the script name.
#
# Giant lines (audit records embedding whole reports, up to ~22 KB) are
# skipped BEFORE any pattern matching.
#
# This file is pure extraction: no judgement, no LLM. Safe to re-run.
# Needs signals-common.awk (awk -f signals-common.awk -f extract-awl-signals.awk).

function base(p,  n, a) { n = split(p, a, "/"); return a[n] }

function emit(date, time, pid, script, task, event, detail) {
  detail = redact(detail)
  if (length(detail) > 200) detail = substr(detail, 1, 200)
  gsub(/\t/, " ", detail)
  if (script == "") script = "unknown"
  if (task == "") task = "-"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", date, time, pid, script, task, event, detail
}

# Pull the JSON-ish value of a key from the line, e.g. "resolved_path": "..."
# Pull a bare numeric value: "duration_s": 7.18
function jsonnum(line, key,  re, s) {
  re = "\"" key "\": *"
  if (!match(line, re)) return ""
  s = substr(line, RSTART + RLENGTH)
  if (!match(s, /^[0-9.]+/)) return ""
  return substr(s, 1, RLENGTH)
}

function jsonstr(line, key,  re, s) {
  re = "\"" key "\": \""
  if (!match(line, re)) return ""
  s = substr(line, RSTART + RLENGTH)
  if (!match(s, /^[^"]*/)) return ""
  return substr(s, 1, RLENGTH)
}

# seconds since 1970-01-01 for "YYYY-MM-DD" "HH:MM:SS,mmm", wall-clock time
# (only differences of a few seconds are used). Plain arithmetic instead of
# gawk's mktime(): BSD awk (macOS) and mawk (Debian/Ubuntu) do not have it.
# Days from the civil date: H. Hinnant's days_from_civil, years >= 0.
function tosecs(d, t,  a, b, y, m, era, yoe, doy, doe) {
  split(d, a, "-"); split(substr(t, 1, 8), b, ":")
  y = a[1] + 0; m = a[2] + 0
  if (m <= 2) y--
  era = int(y / 400); yoe = y - era * 400
  doy = int((153 * (m > 2 ? m - 3 : m + 9) + 2) / 5) + a[3] - 1
  doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
  return ((era * 146097 + doe - 719468) * 24 + b[1]) * 3600 + b[2] * 60 + b[3]
}

# "ai_assist.event_source_mqtt" -> "event_source_mqtt"
function component(lg) { sub(/^ai_assist\./, "", lg); return lg }

# query_limits: space-separated limits in seconds (the wrapper passes the
# default 600 and the configured AI_ASSIST_QUERY_TIMEOUT)
BEGIN { OFS = "\t"; if (query_limits == "") query_limits = "600"; nlim = split(query_limits, LIMITS, " ") }

# ── 0. Python tracebacks ────────────────────────────────────────────
# Continuation lines carry no log prefix: remember the last prefixed line's
# date/time/pid, the last "File ..., in <function>" frame, and emit one
# signal when the exception line ("Type: message") arrives.
/^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] / { lastdate = $1; lasttime = $2; lastpid = $4; in_tb = 0 }
/^Traceback \(most recent call last\):/ { in_tb = 1; tb_frame = "?"; next }
in_tb && /^  File "/ {
  f = $0; sub(/^  File "/, "", f)
  fpath = f; sub(/".*/, "", fpath)
  fn = f; sub(/.*, in /, "", fn)
  tb_frame = base(fpath) ":" fn
  next
}
in_tb && /^[A-Za-z_][A-Za-z0-9_.]*(:|$)/ {
  exc = $0; sub(/:.*/, "", exc)
  m = $0; if (!sub(/^[^:]*: */, "", m)) m = ""
  short = exc; sub(/.*\./, "", short)
  emit(lastdate, lasttime, lastpid, "ai-assist", short, "traceback", exc " @ " tb_frame ": " substr(m, 1, 120))
  in_tb = 0
  next
}
in_tb { next }

# ── 0b. refusals listed under "N security rejection(s):" ───────────
# Indented "[!] <tool>: Error: ..." lines, again without a log prefix.
/^    \[!\] / {
  r = $0; sub(/^    \[!\] [a-z_]+: (Error: )?/, "", r)
  who = owner(lastpid); t = curtask[lastpid]
  if (match(r, /^Command '[^']*' is not in the allowed commands list/)) {
    c = r; sub(/^Command '/, "", c); sub(/' is not in the allowed commands list.*/, "", c)
    emit(lastdate, lasttime, lastpid, who, t, "command_refused", c)
  } else if (r ~ /^Shell builtin\(s\) .* are not allowed/) {
    c = r; sub(/^Shell builtin\(s\) /, "", c); sub(/ are not allowed.*/, "", c)
    emit(lastdate, lasttime, lastpid, who, t, "command_refused", "builtin: " c)
  } else if (r ~ /^[a-z0-9]+ runs arbitrary code/) {
    c = r; sub(/ runs arbitrary code.*/, "", c)
    emit(lastdate, lasttime, lastpid, who, t, "command_refused", c " (arbitrary code)")
  } else if (r ~ /^[a-z0-9.]+ <[a-z-]+> execution is not allowed/) {
    c = r; sub(/ execution is not allowed.*/, "", c)
    emit(lastdate, lasttime, lastpid, who, t, "command_refused", c)
  } else if (r ~ /^find with -exec/) {
    emit(lastdate, lasttime, lastpid, who, t, "command_refused", "find -exec")
  } else if (match(r, /^Path '[^']*' is outside allowed directories/)) {
    c = r; sub(/^Path '/, "", c); sub(/' is outside allowed directories.*/, "", c)
    emit(lastdate, lasttime, lastpid, who, t, "path_refused", c)
  } else if (match(r, /^Skill '[^']*' not allowed to execute scripts/)) {
    c = r; sub(/^Skill '/, "", c); sub(/' not allowed to execute scripts.*/, "", c)
    emit(lastdate, lasttime, lastpid, who, t, "skill_refused", c)
  } else {
    emit(lastdate, lasttime, lastpid, who, t, "command_refused", "other: " substr(r, 1, 120))
  }
  next
}

# ── hard guards ─────────────────────────────────────────────────────
# Audit lines can embed whole reports (up to ~22 KB) and can also contain
# log patterns verbatim inside a logged command (e.g. a grep for
# "AWL task '.*' failed"). So: (a) truncate to 600 chars, and (b) only
# accept a line from the logger that actually emits it — every rule below
# is gated on `logger`.
{ logger = $6; level = $8; $0 = substr($0, 1, 600) }
/ai_assist\.(awl_executor|awl_runtime|agent|action_scheduler|event_sources)/ == 0 && /ai_assist\.audit - INFO - Tool call: internal__execute_command/ == 0 && /Tool description warning/ == 0 && /Skill content warning/ == 0 && !(level == "ERROR" && logger ~ /^ai_assist\./) { next }

{
  date = $1
  time = $2
  pid  = $4
  msg  = $0
  sub(/^[^ ]+ [^ ]+ - [0-9]+ - [a-z_.]+ - [A-Z]+ - /, "", msg)
}

# ── 1. lock acquired: push script on this PID's stack ────────────────
/AWL script lock acquired:/ {
  if (logger != "ai_assist.awl_executor") next
  p = jsonstr($0, "resolved_path")
  if (p == "") p = jsonstr($0, "script")
  script = base(p)
  depth[pid]++
  stack[pid, depth[pid]] = script
  started[pid, script] = date "T" substr(time, 1, 8)
  emit(date, time, pid, script, "-", "run_start", p)
  next
}

# ── 2. lock released: pop, record duration ───────────────────────────
/AWL script lock released:/ {
  if (logger != "ai_assist.awl_executor") next
  p = jsonstr($0, "resolved_path")
  if (p == "") p = jsonstr($0, "script")
  script = base(p)
  d = jsonnum($0, "duration_s")
  # pop matching entry (defensive: pop top if it matches)
  if (depth[pid] > 0) {
    if (stack[pid, depth[pid]] == script) depth[pid]--
    else {
      for (i = depth[pid]; i >= 1; i--) {
        if (stack[pid, i] == script) {
          for (j = i; j < depth[pid]; j++) stack[pid, j] = stack[pid, j + 1]
          depth[pid]--
          break
        }
      }
    }
  }
  emit(date, time, pid, script, "-", "run_end", "duration_s=" (d == "" ? "?" : d))
  for (li = 1; li <= nlim; li++)
    if (d != "" && d + 0 >= LIMITS[li] * 0.95 && d + 0 <= LIMITS[li] + 5)
      emit(date, time, pid, script, "-", "near_limit", "duration_s=" d " (query limit " LIMITS[li] " s)")
  next
}

# ── 2b. a start refused because the script still holds its lock ──────
/AWL script lock rejected:/ {
  if (logger != "ai_assist.awl_executor") next
  p = jsonstr($0, "resolved_path")
  if (p == "") p = jsonstr($0, "script")
  emit(date, time, pid, base(p), "-", "lock_rejected", "start refused: already running")
  next
}

# ── helper: script owning the current line ──────────────────────────
# (top of this PID's stack)
function owner(pid, script) {
  if (depth[pid] > 0) return stack[pid, depth[pid]]
  return "unknown"
}

# ── 3. task lifecycle ───────────────────────────────────────────────
/AWL task '/ {
  if (logger != "ai_assist.awl_runtime") next
  # task name = text between the first pair of single quotes
  t = $0
  sub(/.*AWL task '/, "", t)
  sub(/'.*/, "", t)

  if (/ starting$/) {
    curtask[pid] = t
    emit(date, time, pid, owner(pid), t, "task_start", "")
    next
  }
  if (/ succeeded/) {
    # two wordings: "succeeded, exposed: {...}" and
    # "succeeded (no exposed vars)" — the latter is itself a finding.
    if (/ succeeded \(no exposed vars\)/) {
      emit(date, time, pid, owner(pid), t, "task_ok", "NO_EXPOSED_VARS")
    } else {
      e = $0
      sub(/.*succeeded, exposed: /, "", e)
      sub(/, methods:.*/, "", e)
      emit(date, time, pid, owner(pid), t, "task_ok", e)
    }
    next
  }
  if (/ failed/) {
    emit(date, time, pid, owner(pid), t, "task_fail", msg)
    next
  }
  if (/using model level '/) {
    lvl = $0
    sub(/.*using model level '/, "", lvl); sub(/'.*/, "", lvl)
    emit(date, time, pid, owner(pid), t, "model_level", lvl)
    next
  }
  if (/unresolved variable\(s\) in prompt:/) {
    v = $0
    sub(/.*unresolved variable\(s\) in prompt: */, "", v)
    emit(date, time, pid, owner(pid), t, "unresolved_var", v)
    next
  }
  if (/security rejection\(s\)/) {
    n = $0
    sub(/.*AWL task '[^']*': /, "", n)
    emit(date, time, pid, owner(pid), t, "security_reject", n)
    next
  }
  next
}

# ── 4. agent-level signals (task name must be inferred) ─────────────
/nudging to continue/ {
  if (logger != "ai_assist.agent") next
  emit(date, time, pid, owner(pid), curtask[pid], "nudge_finish", msg)
  next
}
/Turn budget of / {
  if (logger != "ai_assist.agent") next
  emit(date, time, pid, owner(pid), curtask[pid], "budget_turn", msg)
  next
}
/Extended turn budget used up/ {
  if (logger != "ai_assist.agent") next
  emit(date, time, pid, owner(pid), curtask[pid], "budget_extended", msg)
  next
}
/Validation error for internal__/ {
  if (logger != "ai_assist.agent") next
  emit(date, time, pid, owner(pid), curtask[pid], "tool_validation_error", msg)
  next
}
/Tool loop detected/ {
  if (logger != "ai_assist.agent") next
  emit(date, time, pid, owner(pid), curtask[pid], "tool_loop", msg)
  next
}
/not installed/ {
  if (logger != "ai_assist.agent" && logger != "ai_assist.event_sources") next
  emit(date, time, pid, "ai-assist", component(logger), "dependency_missing", msg)
  next
}

# ── 5. AWL-level structural signals ─────────────────────────────────
/AWL @if '/ {
  if (logger != "ai_assist.awl_runtime") next
  verdict = "?"
  if (/evaluated to True/) verdict = "True"
  else if (/evaluated to False/) verdict = "False"
  expr = $0
  sub(/.*AWL @if '/, "", expr)
  sub(/' evaluated to.*/, "", expr)
  emit(date, time, pid, owner(pid), curtask[pid], "if_eval", expr " => " verdict)
  next
}

# ── 6. prompt/skill hygiene ─────────────────────────────────────────
/Tool description warning for / {
  if (logger != "ai_assist.agent") next
  w = $0
  sub(/.*Tool description warning for /, "", w)
  emit(date, time, pid, "global", curtask[pid], "tool_desc_warn", w)
  next
}
/Skill content warning for / {
  if (logger != "ai_assist.agent" && logger != "ai_assist.skills_loader") next
  w = $0
  sub(/.*Skill content warning for /, "", w)
  emit(date, time, pid, "global", curtask[pid], "skill_warn", w)
  next
}

# ── 7. scheduled actions ────────────────────────────────────────────
logger == "ai_assist.action_scheduler" && msg ~ /^Running .*\.\.\.$/ {
  a = msg; sub(/^Running /, "", a); sub(/\.\.\.$/, "", a)
  secs = tosecs(date, time)
  # keyed by action only: a second scheduler process is caught too
  if ((a in last_start) && secs - last_start[a] >= 0 && secs - last_start[a] <= 5)
    emit(date, time, pid, "scheduler", a, "action_duplicate", "started again within " (secs - last_start[a]) " s")
  last_start[a] = secs
  emit(date, time, pid, "scheduler", a, "action_start", "")
  next
}
logger == "ai_assist.action_scheduler" && msg ~ /: failed - / {
  a = msg; sub(/: failed - .*/, "", a)
  d = msg; sub(/^[^:]*: failed - /, "", d)
  emit(date, time, pid, "scheduler", a, "action_fail", d)
  next
}
logger == "ai_assist.action_scheduler" && msg ~ /: completed$/ {
  a = msg; sub(/: completed$/, "", a)
  emit(date, time, pid, "scheduler", a, "action_ok", "")
  next
}
logger == "ai_assist.action_scheduler" && msg ~ /^Reloading actions/ {
  secs = tosecs(date, time)
  if ((pid in last_reload) && secs - last_reload[pid] >= 0 && secs - last_reload[pid] <= 2)
    emit(date, time, pid, "scheduler", "-", "reload_burst", "reload again within " (secs - last_reload[pid]) " s")
  last_reload[pid] = secs
  next
}

# ── 8. any other ERROR from ai-assist (e.g. a reconnect loop) ────────
# The scheduled action failure itself is already an action_fail signal.
level == "ERROR" && logger ~ /^ai_assist\./ {
  if (logger == "ai_assist.action_engine" && msg ~ /^Action '.*' failed$/) next
  emit(date, time, pid, "ai-assist", component(logger), "error_log", msg)
  next
}

# ── 9a. pipe tax: what any command is piped into ─────────────────────
# One "pipe_reducer" event per reducer of a pipeline (`| head -20`,
# `| cut -c1-80`, `| tail -5`...), whatever the first command is. Detail:
# producer \037 reducer form (numbers as #) \037 1 when stderr was merged
# into the pipe (`2>&1 |`). Only known reducers count: a `|` inside quotes
# splits wrongly and must not invent one.
function pipe_reducers(c, date, time, pid,   n, st, i, w, prod, red, opt, merged, seen_red) {
  sub(/"(, "[a-z_]+": .*)?\}\) -> [A-Za-z]+$/, "", c)
  gsub(/\|\|/, "\001", c)
  n = split(c, st, /[ ]*\|[ ]*/)
  if (n < 2) return
  split(st[1], w, " "); prod = w[1]; sub(/.*\//, "", prod)
  merged = (st[1] ~ /2>&1/) ? 1 : 0
  for (i = 2; i <= n; i++) {
    sub(/[;&)].*/, "", st[i])  # `| head -5; echo` -> the reducer is `head -5`
    split(st[i], w, " "); red = w[1]; sub(/.*\//, "", red)
    if (red !~ /^(head|tail|cut|grep|egrep|jq|sed|awk|wc|sort|uniq|tr|column|fold)$/) continue
    opt = (w[2] ~ /^-/) ? w[2] : ""
    sub(/=.*/, "", opt)
    red = red (opt != "" ? " " opt : "")
    gsub(/[0-9]+/, "#", red)
    if (red in seen_red) continue
    seen_red[red] = 1
    emit(date, time, pid, owner(pid), curtask[pid], "pipe_reducer", prod "\037" red "\037" merged)
  }
}

# ── 9. shell commands an internal tool could run instead ────────────
# "<command and options> -> <internal tool>"; a pipeline is marked "| ...":
# it usually means the tool cannot do the whole job.
logger == "ai_assist.audit" && msg ~ /^Tool call: internal__execute_command\(\{"command": "/ {
  c = msg
  sub(/^Tool call: internal__execute_command\(\{"command": "/, "", c)
  sub(/^cd [^;&|]*(&&|;) */, "", c)
  n = split(c, cw, " ")
  cmd = cw[1]; sub(/.*\//, "", cmd)
  form = cmd; opts = 0; piped = 0
  for (i = 2; i <= n; i++) {
    if (cw[i] == "|" || cw[i] ~ /^\|/) { piped = 1; break }
    if (cw[i] == "&&" || cw[i] == ";" || cw[i] ~ /;$/) break
    if (cw[i] ~ /^-[a-zA-Z]/ && opts < 4) { o = cw[i]; sub(/=.*/, "", o); form = form " " o; opts++ }
  }
  pipe_reducers(c, date, time, pid)
  tool = internal_tool(cmd, c)
  if (tool == "") next
  if (cmd == "sed" && c ~ /^sed -n '?[0-9]+,[0-9]+p/) form = "sed -n N,Mp"
  else if (cmd == "sed" && c ~ /^sed -n '?\//) form = "sed -n /re/,/re/p"
  if (piped) form = form " | ..."
  emit(date, time, pid, owner(pid), curtask[pid], "shell_instead_of_tool", form " -> " tool)
  next
}
