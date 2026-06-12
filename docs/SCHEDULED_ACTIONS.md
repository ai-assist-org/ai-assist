# Scheduled Actions

Schedule one-shot future actions that execute automatically with notifications.

## Usage

In interactive mode, ask the agent to remind you:

```bash
ai-assist /interactive
You: Remind me in 2 hours to check the DCI job status for job-456
```

The agent schedules the action → `/monitor` process executes it → You get notified.

## Time Formats

- `in 2 hours`, `in 30 minutes`, `in 1 day`
- `in 3h`, `in 15m`, `in 2d` (short forms)
- `tomorrow at 9am`
- `next monday 10:00`

## Notification Channels

Configured automatically by the agent based on your request:

- **desktop** - System notifications (notify-send on Linux) - truncated to 500 chars
- **file** - Append to `~/.ai-assist/notifications.log` - full result
- **console** - Display in `/monitor` output - full result
- **TUI** - Auto-displayed in interactive mode - full result with Rich formatting

The agent decides the best execution strategy:
- Simple reminders → Just sends notification (no agent query)
- Data queries → Executes via agent, then notifies with results
- Reports → Executes via agent, saves to file (optional notification)

## How Actions Execute

**Reminder-only actions** (`notify=True`, `create_report=False`):
- Send notification directly with your reminder message
- No agent query executed (efficient for simple reminders)
- Example: "Remind me in 1 hour to watch TV" → Shows "⏰ Time to watch TV!"

**Query/Report actions** (`create_report=True` or mixed):
- Execute the prompt via the agent
- Results sent to notification channels and/or saved as report
- Example: "Check failed jobs in 30 minutes and notify me" → Agent queries, notifies with results

## Managing Actions

**View scheduled actions:**
```bash
cat ~/.ai-assist/scheduled-actions.json
```

**View notification history:**
```bash
tail -f ~/.ai-assist/notifications.log
```

## File Maintenance

Scheduled actions are automatically cleaned up:
- **Pending actions**: Kept indefinitely until executed
- **Completed/failed actions**: Kept for 7 days, then archived
- **Archive location**: `~/.ai-assist/scheduled-actions-archive.jsonl`

The archive uses JSONL format (one action per line) for efficient append-only storage.

**Manual cleanup:**
```bash
ai-assist /cleanup-actions
```

**View archived actions:**
```bash
# See all archived actions
cat ~/.ai-assist/scheduled-actions-archive.jsonl | jq

# Count archived actions
wc -l ~/.ai-assist/scheduled-actions-archive.jsonl
```

## Requirements

**The `/monitor` process must be running:**
```bash
ai-assist /monitor &
```

Or in tmux:
```bash
tmux new -d -s ai-assist-monitor 'ai-assist /monitor'
```

**For desktop notifications (Linux):**
```bash
sudo dnf install libnotify  # Fedora/RHEL
sudo apt install libnotify-bin  # Ubuntu/Debian
```

## Periodic Task Notifications

Add notifications to actions in `~/.ai-assist/event-schedules.json`:

```json
{
  "monitors": [
    {
      "name": "critical-failures",
      "prompt": "Check for critical DCI failures",
      "interval": "1h",
      "notify": true,
      "notification_channels": ["desktop", "file"]
    }
  ]
}
```

**Fields:**
- `notify` (boolean) - Enable notifications (default: false)
- `notification_channels` (array) - List of channels (default: ["console"])

## Troubleshooting

**Actions not executing:**
Check `/monitor` is running:
```bash
ps aux | grep "ai-assist /monitor"
```

**Desktop notifications not appearing:**
```bash
which notify-send
notify-send "Test" "Test notification"
```

**Action failed:**
Check the result in scheduled-actions.json:
```bash
jq '.actions[] | select(.status == "failed")' ~/.ai-assist/scheduled-actions.json
```
