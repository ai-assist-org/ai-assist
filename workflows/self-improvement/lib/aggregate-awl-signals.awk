# aggregate-awl-signals.awk — turn extract-awl-signals.tsv into a
# deterministic scorecard.
#
# Input  (stdin or ARGV): TSV from extract-awl-signals.awk
# Output (stdout): two record kinds, JSON, one per line.
#
#   signal records — "@SIG@\t{json}" where @SIG@ is a stable signature key
#     that the caller hashes (see awl-signals-aggregate.sh):
#     {"kind":"signal","signature":"@SIG@","script","task","event",
#      "count","first","last","sample","mechanical"}
#
#   script records — "-" as first column, per-script rollup:
#     {"kind":"script","script","path","runs","run_seconds","tasks_started",
#      "tasks_ok","task_fail","unresolved_var","security_reject",
#      "security_reject_total","budget","nudge","if_false"}
#
#   action records — "-" as first column, per scheduled action:
#     {"kind":"action","action","runs","ok","fail","duplicate"}
#
#   shell records — "@ROLLUP@" as first column (hashed like signals), per shell command and single
#   option (or "| " for a pipe) that an internal tool could replace (rolled
#   up, not signals: one per task would flood the findings and the ledger):
#     {"kind":"shell","tool","form","count","top_task","top_task_count"}
#
#   pipe records — "@ROLLUP@" (hashed), per form a command output is piped
#   into (`head -#`, `cut -c#-#`...), any command:
#     {"kind":"pipe","reducer","count","stderr_merged","last","top_producer","top_task"}
#
#   prompt records — "@ROLLUP@" as first column (hashed), per security question form (see
#   section 10 of the extractor), rolled up like the shell records:
#     {"kind":"prompt","prompt":"command|path","form","tool","count",
#      "approved","denied","auto_denied","top_task","top_task_count","sample"}
#
# The pseudo-scripts "scheduler" and "ai-assist" (see the extractor) get no
# script record: they are not .awl files.
#
# Determinism: grouping is sorted by (script, task, event, detail); output
# ordering is therefore reproducible. No randomness, no time-dependence.

# Replace every character c in s by r. Not gsub(): backslashes in its
# replacement string are read differently by gawk, BSD awk and mawk.
function replace_char(s, c, r,  parts, n, i, out) {
  n = split(s, parts, c)
  out = parts[1]
  for (i = 2; i <= n; i++) out = out r parts[i]
  return out
}

function jesc(s) {
  s = replace_char(s, "\\", "\\\\")
  s = replace_char(s, "\"", "\\\"")
  gsub(/[\r\n]/, " ", s)
  return s
}

# Normalize a detail so cosmetic variation (numbers, whitespace, PIDs,
# timestamps) does not create a brand-new signature every run.
function norm(d) {
  gsub(/[0-9]+/, "#", d)
  gsub(/[[:space:]]+/, " ", d)
  return d
}

# Static baseline classification. Returns 1 if mechanically fixable, so the
# report can rank mechanical fixes first. The classify task may override
# these, but this table keeps re-runs stable and reproducible.
function is_mechanical(ev) {
  if (ev == "unresolved_var")  return 1   # typo / missing Expose:
  if (ev == "tool_desc_warn")  return 1   # trim tool description text
  if (ev == "skill_warn")      return 1   # trim SKILL.md text
  return 0                                # everything else needs judgement
}

BEGIN { FS = "\t" }

{
  date = $1; time = $2; pid = $3; script = $4; task = $5; event = $6; detail = $7

  # ── per-action rollup (scheduled actions) ──────────────────────────
  if (script == "scheduler" && task != "-") {
    actions[task] = 1
    if (event == "action_start")     a_runs[task]++
    if (event == "action_ok")        a_ok[task]++
    if (event == "action_fail")      a_fail[task]++
    if (event == "action_duplicate") a_dup[task]++
  }

  # ── per-script rollup (counted before any skip) ────────────────────
  # (security questions have their own rollup: one asked outside any workflow
  # must not create an "unknown" script)
  if (script != "scheduler" && script != "ai-assist" && event != "security_prompt") seen[script] = 1
  if (event == "run_start")       { runs[script]++; path[script] = detail }
  if (event == "run_end") {
    d = detail; sub(/^duration_s=/, "", d)
    if (d ~ /^[0-9.]+$/) runsec[script] += d
  }
  if (event == "task_start")      { starts[script]++ }
  if (event == "task_ok")         { oks[script]++ }
  if (event == "task_fail")       { fails[script]++ }
  if (event == "unresolved_var")  { unres[script]++ }
  if (event == "security_reject") {
    rejects[script]++
    n = detail; sub(/[^0-9].*/, "", n); if (n == "") n = 1
    rejects_total[script] += n
  }
  if (event == "budget_turn" || event == "budget_extended") { budget[script]++ }
  if (event == "nudge_finish")    { nudge[script]++ }
  if (event == "if_eval" && detail ~ /=> False$/) { iffalse[script]++ }

  # ── shell commands an internal tool could run: rolled up ───────────
  # One record per command and single option ("grep -rn" counts for
  # "grep -r" and "grep -n"; a pipe counts as "grep |"): a capability is
  # then counted whole instead of split across option combinations.
  if (event == "shell_instead_of_tool") {
    f = detail; sub(/ -> .*/, "", f); gsub(/[0-9]+/, "#", f)
    t = detail; sub(/.* -> /, "", t)
    nf = split(f, ft, " "); cmd = ft[1]; shk = 0
    if (f ~ /^sed -n /) { keys[++shk] = f; sub(/ \| \.\.\.$/, "", keys[shk]) }
    else for (j = 2; j <= nf; j++) {
      if (ft[j] ~ /^--/) keys[++shk] = cmd " " ft[j]
      else if (cmd == "find" && ft[j] ~ /^-[a-z]+$/) keys[++shk] = cmd " " ft[j]
      else if (ft[j] ~ /^-[a-zA-Z#]+$/) {
        letters = substr(ft[j], 2); gsub(/#/, "", letters)
        for (m = 1; m <= length(letters); m++) keys[++shk] = cmd " -" substr(letters, m, 1)
      }
    }
    if (f ~ / \| \.\.\.$/) keys[++shk] = cmd " |"
    if (shk == 0) keys[++shk] = cmd
    for (j = 1; j <= shk; j++) {
      sk = t SUBSEP keys[j]
      if ((sk, NR) in seen_line) continue
      seen_line[sk, NR] = 1
      sh_count[sk]++
      if (date "T" time > sh_last[sk]) sh_last[sk] = date "T" time
      sh_task[sk, script "::" task]++
      if (sh_task[sk, script "::" task] > sh_top_n[sk]) { sh_top_n[sk] = sh_task[sk, script "::" task]; sh_top[sk] = script "::" task }
    }
    next
  }

  # ── pipe tax: rolled up per reducer form ─────────────────────────
  if (event == "pipe_reducer") {
    split(detail, pp, "\037")
    rk = pp[2]
    pi_count[rk]++
    if (pp[3] == "1") pi_merged[rk]++
    if (date "T" time > pi_last[rk]) pi_last[rk] = date "T" time
    pi_prod[rk, pp[1]]++
    if (pi_prod[rk, pp[1]] > pi_prod_n[rk]) { pi_prod_n[rk] = pi_prod[rk, pp[1]]; pi_top_prod[rk] = pp[1] }
    pi_task[rk, script "::" task]++
    if (pi_task[rk, script "::" task] > pi_top_n[rk]) { pi_top_n[rk] = pi_task[rk, script "::" task]; pi_top[rk] = script "::" task }
    next
  }

  # ── security questions: rolled up per form ───────────────────────
  if (event == "security_prompt") {
    split(detail, pf, "\037")
    pk = pf[1] SUBSEP pf[3] SUBSEP pf[4]
    if (!(pk in pr_count)) pr_sample[pk] = pf[5]
    pr_count[pk]++
    if (date "T" time > pr_last[pk]) pr_last[pk] = date "T" time
    if (pf[2] == "approved") pr_ok[pk]++
    else if (pf[2] == "denied") pr_no[pk]++
    else pr_auto[pk]++
    pr_task[pk, script "::" task]++
    if (pr_task[pk, script "::" task] > pr_top_n[pk]) { pr_top_n[pk] = pr_task[pk, script "::" task]; pr_top[pk] = script "::" task }
    next
  }

  # ── signal grouping: skip pure plumbing, keep findings ────────────
  if (event == "run_start" || event == "run_end" || event == "model_level") next
  if (event == "task_start" || event == "action_start" || event == "action_ok") next
  if (event == "task_ok") {
    # only interesting when the task exposed nothing useful
    if (detail != "NO_EXPOSED_VARS" && detail !~ /^\{('[^']+': '\[\]', )*'[^']+': '\[\]'\}$/) next
  }

  nd = norm(detail)
  key = script SUBSEP task SUBSEP event SUBSEP nd
  if (!(key in cnt)) {
    order[++nk] = key
    k_script[key] = script; k_task[key] = task; k_event[key] = event
    k_detail[key] = detail; k_nd[key] = nd
  }
  cnt[key]++
  ts = date "T" time
  if (!(key in k_first) || ts < k_first[key]) k_first[key] = ts
  if (!(key in k_last)  || ts > k_last[key])  k_last[key]  = ts
}

END {
  # ── signal records (deterministic order) ───────────────────────────
  for (i = 1; i <= nk; i++) {
    key = order[i]
    sigkey = k_script[key] "|" k_task[key] "|" k_event[key] "|" k_nd[key]
    printf("@SIG@\t{\"kind\":\"signal\",\"signature\":\"@SIG@\",\"script\":\"%s\",\"task\":\"%s\",\"event\":\"%s\",\"count\":%d,\"first\":\"%s\",\"last\":\"%s\",\"sample\":\"%s\",\"mechanical\":%s}\t%s\n",
      jesc(k_script[key]), jesc(k_task[key]), jesc(k_event[key]),
      cnt[key], k_first[key], k_last[key], jesc(substr(k_detail[key], 1, 200)),
      is_mechanical(k_event[key]) ? "true" : "false", jesc(sigkey))
  }

  # ── per-script rollup ──────────────────────────────────────────────
  for (s in seen) {
    printf("-\t{\"kind\":\"script\",\"script\":\"%s\",\"path\":\"%s\",\"runs\":%d,\"run_seconds\":%.1f,\"tasks_started\":%d,\"tasks_ok\":%d,\"task_fail\":%d,\"unresolved_var\":%d,\"security_reject\":%d,\"security_reject_total\":%d,\"budget\":%d,\"nudge\":%d,\"if_false\":%d}\n",
      jesc(s), jesc(path[s]), runs[s], runsec[s], starts[s], oks[s], fails[s], unres[s],
      rejects[s], rejects_total[s], budget[s], nudge[s], iffalse[s])
  }

  # ── per shell command form ─────────────────────────────────────────
  # "@ROLLUP@" lines get a signature like signal records (the caller hashes
  # the key in the last column), so a reviewer can record a verdict on them.
  for (sk in sh_count) {
    split(sk, parts, SUBSEP)
    printf("@ROLLUP@\t{\"kind\":\"shell\",\"signature\":\"@SIG@\",\"script\":\"ai-assist\",\"task\":\"%s\",\"event\":\"shell_instead_of_tool\",\"tool\":\"%s\",\"form\":\"%s\",\"sample\":\"%s\",\"count\":%d,\"last\":\"%s\",\"top_task\":\"%s\",\"top_task_count\":%d}\t%s\n",
      jesc(parts[1]), jesc(parts[1]), jesc(parts[2]), jesc(parts[2]), sh_count[sk], sh_last[sk],
      jesc(sh_top[sk]), sh_top_n[sk], jesc("shell|" parts[1] "|" parts[2]))
  }

  # ── per security question form ────────────────────────────────────
  for (pk in pr_count) {
    split(pk, parts, SUBSEP)
    printf("@ROLLUP@\t{\"kind\":\"prompt\",\"signature\":\"@SIG@\",\"script\":\"ai-assist\",\"task\":\"%s\",\"event\":\"security_prompt\",\"prompt\":\"%s\",\"form\":\"%s\",\"tool\":\"%s\",\"count\":%d,\"last\":\"%s\",\"approved\":%d,\"denied\":%d,\"auto_denied\":%d,\"top_task\":\"%s\",\"top_task_count\":%d,\"sample\":\"%s\"}\t%s\n",
      jesc(parts[2]), jesc(parts[1]), jesc(parts[2]), jesc(parts[3]), pr_count[pk], pr_last[pk],
      pr_ok[pk], pr_no[pk], pr_auto[pk], jesc(pr_top[pk]), pr_top_n[pk], jesc(pr_sample[pk]),
      jesc("prompt|" parts[1] "|" parts[2] "|" parts[3]))
  }

  # ── per pipe reducer form ─────────────────────────────────────────
  for (rk in pi_count) {
    printf("@ROLLUP@\t{\"kind\":\"pipe\",\"signature\":\"@SIG@\",\"script\":\"ai-assist\",\"task\":\"execute_command\",\"event\":\"pipe_reducer\",\"reducer\":\"%s\",\"sample\":\"| %s\",\"count\":%d,\"stderr_merged\":%d,\"last\":\"%s\",\"top_producer\":\"%s\",\"top_task\":\"%s\",\"top_task_count\":%d}\t%s\n",
      jesc(rk), jesc(rk), pi_count[rk], pi_merged[rk], pi_last[rk], jesc(pi_top_prod[rk]),
      jesc(pi_top[rk]), pi_top_n[rk], jesc("pipe|" rk))
  }

  # ── per-action rollup ──────────────────────────────────────────────
  for (a in actions) {
    printf("-\t{\"kind\":\"action\",\"action\":\"%s\",\"runs\":%d,\"ok\":%d,\"fail\":%d,\"duplicate\":%d}\n",
      jesc(a), a_runs[a], a_ok[a], a_fail[a], a_dup[a])
  }
}
