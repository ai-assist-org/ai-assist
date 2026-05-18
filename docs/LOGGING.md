# Logging

ai-assist maintains detailed logs to help with debugging and monitoring agent behavior.

## Log Files

Logs are automatically written to:

```
~/.ai-assist/logs/ai-assist-YYYY-MM-DD.log
```

A new log file is created each day. For example:
- `ai-assist-2026-03-31.log`
- `ai-assist-2026-04-01.log`

## Log Levels

ai-assist uses Python's standard logging levels:

| Level | Description | Typical Use |
|-------|-------------|-------------|
| `DEBUG` | Detailed diagnostic information | Troubleshooting, development |
| `INFO` | General informational messages | Normal operation tracking |
| `WARNING` | Warning messages (non-critical issues) | Potential problems |
| `ERROR` | Error messages (failures) | Operation failures |
| `CRITICAL` | Critical failures | Severe errors |

## Configuration

Logging is configured separately for **file output** and **console output**.

### Environment Variables

**`AI_ASSIST_LOG_LEVEL`** - Controls what gets written to log files
- **Default:** `INFO`
- **Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Recommendation:** Keep at `INFO` or `DEBUG` for comprehensive logs

**`AI_ASSIST_CONSOLE_LOG_LEVEL`** - Controls what appears in the terminal
- **Default:** `WARNING`
- **Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Recommendation:** Keep at `WARNING` to avoid cluttering terminal output

### Default Behavior

By default (no environment variables set):
- **File logs:** Everything at `INFO` level and above
- **Console output:** Only `WARNING` and `ERROR` messages

This keeps your terminal clean while maintaining detailed logs for troubleshooting.

## Usage Examples

### Standard Operation (Default)

```bash
ai-assist
```

Result:
- File: INFO, WARNING, ERROR messages
- Console: WARNING, ERROR messages

### Debug Mode (Verbose)

```bash
export AI_ASSIST_LOG_LEVEL=DEBUG
export AI_ASSIST_CONSOLE_LOG_LEVEL=INFO
ai-assist
```

Result:
- File: All messages including DEBUG
- Console: INFO, WARNING, ERROR messages

### Quiet Mode (Errors Only)

```bash
export AI_ASSIST_LOG_LEVEL=ERROR
export AI_ASSIST_CONSOLE_LOG_LEVEL=ERROR
ai-assist
```

Result:
- File: Only ERROR and CRITICAL messages
- Console: Only ERROR and CRITICAL messages

### One-Time Debug Run

No need to export - just set for this command:

```bash
AI_ASSIST_LOG_LEVEL=DEBUG AI_ASSIST_CONSOLE_LOG_LEVEL=DEBUG ai-assist
```

### Permanent Configuration

Add to your `~/.bashrc`, `~/.zshrc`, or `~/.profile`:

```bash
# ai-assist logging configuration
export AI_ASSIST_LOG_LEVEL=INFO
export AI_ASSIST_CONSOLE_LOG_LEVEL=WARNING
```

## Viewing Logs

### Tail Current Log

```bash
tail -f ~/.ai-assist/logs/ai-assist-$(date +%Y-%m-%d).log
```

### View All Logs

```bash
ls -lh ~/.ai-assist/logs/
```

### Search Logs

```bash
# Find all errors
grep "ERROR" ~/.ai-assist/logs/*.log

# Find specific message
grep "context overflow" ~/.ai-assist/logs/*.log

# Find messages from specific module
grep "ai_assist.agent" ~/.ai-assist/logs/*.log
```

### View Last 100 Lines

```bash
tail -100 ~/.ai-assist/logs/ai-assist-$(date +%Y-%m-%d).log
```

### Follow Logs in Real-Time with Filtering

```bash
# Only show warnings and errors
tail -f ~/.ai-assist/logs/ai-assist-$(date +%Y-%m-%d).log | grep -E "WARNING|ERROR"

# Only show specific module
tail -f ~/.ai-assist/logs/ai-assist-$(date +%Y-%m-%d).log | grep "ai_assist.agent"
```

## Log Format

Each log entry contains:

```
TIMESTAMP - MODULE - LEVEL - MESSAGE
```

Example:

```
2026-03-31 17:15:33,456 - ai_assist.filesystem_tools - WARNING - Command 'tar' requires confirmation
2026-03-31 17:15:34,789 - ai_assist.mcp_client - ERROR - MCP server connection failed: timeout
```

## What Gets Logged

### File Logs (Default: INFO level)

- Agent initialization and configuration
- MCP server connections and disconnections
- Tool executions and results
- API calls and token usage
- Knowledge graph operations
- Scheduled action executions
- Cache hits and misses
- Configuration loading
- Errors and exceptions with stack traces

### Console Output (Default: WARNING level)

- Warnings about potential issues
- Error messages
- Critical failures

**Note:** Normal agent output (responses, tool use notifications) is **not** part of logging - that goes directly to the terminal/TUI.

## Troubleshooting

### No Log Files Created

Check that the config directory exists:

```bash
ls -la ~/.ai-assist/logs/
```

If it doesn't exist, ai-assist will create it on startup.

### Logs Too Verbose

Reduce the log level:

```bash
export AI_ASSIST_LOG_LEVEL=WARNING
export AI_ASSIST_CONSOLE_LOG_LEVEL=ERROR
```

### Need More Detail for Debugging

Enable DEBUG level:

```bash
export AI_ASSIST_LOG_LEVEL=DEBUG
export AI_ASSIST_CONSOLE_LOG_LEVEL=DEBUG
```

### Terminal Cluttered with Log Messages

The console log level is too low. Increase it:

```bash
export AI_ASSIST_CONSOLE_LOG_LEVEL=WARNING  # or ERROR
```

### Want Logs But Clean Terminal

This is the default behavior! Just use:

```bash
# File gets detailed logs, console stays clean
export AI_ASSIST_LOG_LEVEL=INFO
export AI_ASSIST_CONSOLE_LOG_LEVEL=WARNING
```

## Log Rotation

Currently, logs are **not automatically rotated** - each day gets a new file, but old files are kept indefinitely.

To manually clean old logs:

```bash
# Delete logs older than 30 days
find ~/.ai-assist/logs/ -name "*.log" -mtime +30 -delete

# Archive logs older than 7 days
find ~/.ai-assist/logs/ -name "*.log" -mtime +7 -exec gzip {} \;
```

You can add this to a cron job for automatic cleanup:

```bash
# Clean logs older than 30 days daily at 2am
0 2 * * * find ~/.ai-assist/logs/ -name "*.log" -mtime +30 -delete
```

## Best Practices

1. **Keep file logging at INFO or DEBUG** - disk space is cheap, missing logs are expensive
2. **Keep console logging at WARNING** - reduces terminal clutter
3. **Check logs when debugging issues** - they contain valuable context
4. **Use grep/tail for log analysis** - don't try to read the whole file
5. **Set DEBUG mode only when troubleshooting** - it generates a lot of output

## Related

- [Configuration](../README.md#configuration) - General configuration options
- [Quick Start](../README.md#quick-start) - Getting started guide
