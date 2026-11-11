# signals-common.awk — functions shared by extract-awl-signals.awk and
# extract-prompts.awk (load it first: awk -f signals-common.awk -f ...).

function redact(s) {
  gsub(/[Pp]assword[=:][^ ,}]*/, "password=<redacted>", s)
  gsub(/[Tt]oken[=:][^ ,}]*/, "token=<redacted>", s)
  gsub(/[Ss]ecret[=:][^ ,}]*/, "secret=<redacted>", s)
  gsub(/api[_-]?[Kk]ey[=:][^ ,}]*/, "apikey=<redacted>", s)
  return s
}

# The internal tool that does what shell command <cmd> (full text <c>) does,
# or "" when there is none.
function internal_tool(cmd, c) {
  if (cmd ~ /^(grep|egrep|rg)$/) return "internal__search_in_file"
  if (cmd ~ /^(cat|head|tail)$/) return "internal__read_file"
  if (cmd == "sed" && c ~ /^sed -i/) return "internal__edit_file"
  if (cmd == "sed" && c ~ /^sed -n '?([0-9]+,[0-9]+p|\/)/) return "internal__read_file"
  if (cmd == "ls" || (cmd == "find" && c !~ / -exec/)) return "internal__list_directory"
  if (cmd == "jq") return "internal__json_query"
  if (cmd == "mkdir") return "internal__create_directory"
  return ""
}
