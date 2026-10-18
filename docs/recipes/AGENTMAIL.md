# AgentMail: react to new email

Use AgentMail with ai-assist to react to an incoming email without exposing an
inbound webhook endpoint. A local AgentMail MCP server holds an outbound
connection to AgentMail, emits an MCP event, and ai-assist runs an action in
response.

This recipe uses the event bridge in the [AgentMail MCP pull request #91](https://github.com/agentmail-to/agentmail-mcp/pull/91). Use a local checkout that includes that change.

## How it works

```text
AgentMail event stream → local AgentMail MCP server → ai-assist /monitor
                                                     → action → notification
```

The local server connects outward to AgentMail. ai-assist does not need a
public URL, so this also works behind a firewall or NAT.

## Configure the local MCP server

In `~/.ai-assist/mcp_servers.yaml`, configure AgentMail to run from your local
checkout rather than through a hosted or transient package runner:

```yaml
agentmail:
  command: uv
  args:
    - run
    - --directory
    - /path/to/agentmail-mcp/python/stdio-bridge
    - agentmail-mcp
  env:
    AGENTMAIL_API_KEY: ${AGENTMAIL_API_KEY}
```

Set `AGENTMAIL_API_KEY` in the environment used by ai-assist. The bridge
connects to AgentMail and expands an unfiltered email subscription to the
inboxes available to that API key.

## Create the email action

Add the MCP source and an action to `~/.ai-assist/event-schedules.json`:

```json
{
  "event_sources": { "mcp": {} },
  "actions": [
    {
      "name": "New AgentMail email",
      "trigger": {
        "type": "mcp",
        "server": "agentmail",
        "event": "message.received",
        "arguments": {}
      },
      "prompt": "Fetch the referenced new AgentMail email, read it, and identify any action needed.",
      "enabled": true,
      "notify": true,
      "notification_channels": ["desktop", "console", "file"]
    }
  ]
}
```

If your file already has `event_sources` or `actions`, merge these entries
instead of replacing it.

`arguments` is empty to receive events from every inbox accessible to the API
key. A bridge that supports filtering can instead accept values such as an
inbox ID in `arguments`.

## Receive the result interactively

Run the monitor process:

```bash
ai-assist /monitor
```

Keep `"file"` in `notification_channels`. It writes the action result to the
shared notification stream, which an open interactive ai-assist session shows
automatically. `desktop` sends an operating-system notification; `console`
writes to the monitor output.

## Choose the level of automation

The action prompt controls what happens after an email arrives. Start with
triage or summarization. For example, change the prompt to:

```text
Fetch the referenced email. Summarize it, identify required action, and notify
me. Do not draft or send a reply.
```

Only use a prompt that sends replies once you are comfortable granting that
authority to an automated event-triggered action. An event notification carries
identifiers and metadata; the action should fetch the message through the
normal AgentMail MCP tools before deciding what to do.
