# extract-prompts.awk — classify the security questions asked to the user.
#
# Input: TSV from the audit log (see awl-signals-aggregate.sh), one row per
# question: date \t time \t pid \t script \t task \t tool \t kind \t decision
# \t question, newlines in the question replaced by \036.
# Output: extract-awl-signals.awk's TSV, event "security_prompt", detail:
# kind, decision, form, the internal tool that could have done it ("-" if
# none) and the question, separated by \037. Forms: "<cmd> (arbitrary
# code)", "python3 stdin execution", "<cmd> [subcommands]" for a command
# outside the allowlist, "path for <cmd>" for a path outside the allowed ones, or the
# tool name when the question is not about a shell command (e.g. scheduling
# an action).
#
# Needs signals-common.awk (awk -f signals-common.awk -f extract-prompts.awk).

BEGIN { FS = "\t"; OFS = "\t" }

# The command a question is about: its name, plus the subcommands of tools that
# have them (`git worktree list`, `gh pr view`), so a reviewer can tell which
# ones could be approved without asking. Leading `cd ... &&`, redirections,
# env assignments and grouping are skipped.
function command_form(q,   c, n, w, i, form, subs) {
  c = q
  sub(/^cd [^;&|]*(&&|;) */, "", c)
  gsub(/[0-9]*>&[0-9]+/, "", c)  # 2>&1 is not a command separator
  if (c ~ /^[ (]*(for|while|until|if|case|select|mapfile|readarray)([ ;]|$)/) return "(shell construct)"
  sub(/[;&|].*/, "", c)
  n = split(c, w, " ")
  for (i = 1; i <= n; i++) {
    if (w[i] ~ /^[({]+$/ || w[i] ~ /^[0-9]*[<>]/ || w[i] ~ /^[A-Za-z_][A-Za-z0-9_]*=/) continue
    if (w[i] ~ /^(env|nice|nohup|stdbuf)$/) continue  # wrappers: the command follows
    if (w[i] == "timeout") { i++; continue }        # and its duration
    break
  }
  if (i > n) return "(unparsed)"
  form = w[i]; sub(/^[({]+/, "", form); sub(/.*\//, "", form)
  if (form ~ /^(git|gh|podman|docker|oc|kubectl|systemctl|journalctl|uv|dnf|apt|npm|pip|cargo|go|helm)$/) {
    subs = 0
    for (i++; i <= n && subs < 2; i++) {
      if (w[i] ~ /^-[Cc]$/) { i++; continue }  # git -C <dir>, git -c <key=value>
      if (w[i] ~ /^-/ && subs == 0) continue   # other global options before the subcommand
      if (w[i] !~ /^[a-z][a-z0-9-]*$/) break
      form = form " " w[i]; subs++
    }
  }
  return form
}

{
  ptool = $6; kind = $7; decision = $8; q = $9
  stool = "-"
  if (kind == "path") {
    form = "path"
    if (match(q, /\036Reason: /)) {
      reason = substr(q, RSTART + RLENGTH)
      sub(/^cd [^;&|]*(&&|;) */, "", reason); sub(/^[({ ]+/, "", reason)
      split(reason, rw, " "); pc = rw[1]; sub(/.*\//, "", pc)
      if (pc != "") form = "path for " pc
    }
  } else if (ptool != "internal__execute_command") {
    form = ptool
  } else if (q ~ /^[A-Za-z0-9._-]+ runs arbitrary code/) {
    form = q; sub(/ runs arbitrary code.*/, "", form); form = form " (arbitrary code)"
  } else if (q ~ /^[A-Za-z0-9._-]+ [^ ]+ execution: /) {
    form = q; sub(/: .*/, "", form)
  } else {
    form = command_form(q)
    split(form, fw, " ")
    t = internal_tool(fw[1], q); if (t != "") stool = t
  }
  gsub(/\036/, " ", q)
  q = substr(redact(q), 1, 200)
  print $1, $2, $3, $4, $5, "security_prompt", kind "\037" decision "\037" form "\037" stool "\037" q
}
