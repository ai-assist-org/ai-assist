# Query diagnostics

ai-assist records a lightweight timeline for every query at
`~/.ai-assist/traces/query_events.jsonl`. Unlike the final query trace, this
file is updated while the query runs, so it remains useful after Ctrl-C or a
provider failure.

Events contain only operational metadata: query ID, elapsed time, turn,
phase, and tool name. Prompts, tool arguments, and tool results are not saved
there.

Use `/debug` during an interactive query to see its current phase and elapsed
time. Useful phases include `model_stream_opened`, `model_first_event`,
`tool_started`, and `tool_completed`.

Model streams time out if the first event or a later event takes too long:

```bash
export AI_ASSIST_MODEL_FIRST_EVENT_TIMEOUT=120
export AI_ASSIST_MODEL_IDLE_TIMEOUT=120
```

Both values are seconds and default to 120. On timeout, the query returns an
error rather than waiting indefinitely; increase them for deliberately slow
models or providers.

Security approval prompts are also cancelled with their query. If an older
process left a terminal in raw mode after an interruption, run `stty sane` in
that terminal before restarting ai-assist.
