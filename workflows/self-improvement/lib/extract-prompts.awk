# extract-prompts.awk — classify the security questions asked to the user.
#
# Input: TSV from the audit log (see awl-signals-aggregate.sh), one row per
# question: date \t time \t pid \t script \t task \t tool \t kind \t decision
# \t question, newlines in the question replaced by \036.
# Output: extract-awl-signals.awk's TSV, event "security_prompt", detail:
# kind, decision, form, the internal tool that could have done it ("-" if
# none) and the question, separated by \037. Forms: "<cmd> (arbitrary
# code)", "python3 stdin execution", "<cmd>" for a command outside the
# allowlist, "path for <cmd>" for a path outside the allowed ones, or the
# tool name when the question is not about a shell command (e.g. scheduling
# an action).
#
# Needs signals-common.awk (awk -f signals-common.awk -f extract-prompts.awk).

BEGIN { FS = "\t"; OFS = "\t" }

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
    c = q; sub(/^cd [^;&|]*(&&|;) */, "", c)
    split(c, cw, " "); form = cw[1]; sub(/.*\//, "", form)
    t = internal_tool(form, c); if (t != "") stool = t
  }
  gsub(/\036/, " ", q)
  q = substr(redact(q), 1, 200)
  print $1, $2, $3, $4, $5, "security_prompt", kind "\037" decision "\037" form "\037" stool "\037" q
}
