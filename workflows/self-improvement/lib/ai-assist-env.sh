# ai-assist-env.sh — sourced by the helpers in this directory to find the
# ai-assist installation, given its source tree (the AWL built-in
# ${source_dir}) — nothing depends on where it is checked out.
#
#   ai_assist_python SRC     the python to import ai-assist with: the
#                            AI_ASSIST_PYTHON override, else SRC/.venv's, else
#                            the interpreter of the ai-assist launcher on PATH
#   ai_assist_setting K SRC CFG
#                            the value of setting K from the .env files
#                            ai-assist loads — SRC/.env, then CFG/.env, which
#                            wins — empty if unset. Callers ask only for
#                            non-secret keys: the files also hold API keys.

ai_assist_python() {
  if [ -n "${AI_ASSIST_PYTHON:-}" ]; then
    echo "$AI_ASSIST_PYTHON"
  elif [ -n "${1:-}" ] && [ -x "$1/.venv/bin/python" ]; then
    echo "$1/.venv/bin/python"
  else
    local launcher
    launcher=$(command -v ai-assist 2>/dev/null) || return 1
    sed -n '1s/^#![[:space:]]*//p' "$launcher"
  fi
}

ai_assist_setting() {
  local key="$1" f line value=""
  for f in "${2:-}/.env" "${3:-$HOME/.ai-assist}/.env"; do
    [ -f "$f" ] || continue
    line=$(grep -E "^$key=" "$f" | tail -1 || true)
    if [ -n "$line" ]; then value=${line#*=}; fi
  done
  value=${value%\"}; value=${value#\"}; value=${value%\'}; value=${value#\'}
  echo "$value"
}
