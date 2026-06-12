# Sandbox Container Images

Container images for running ai-assist instances in isolated podman-compose stacks.

## Build

```bash
# From the ai-assist repo root:
podman build -t ai-assist-sandbox -f sandbox/ai-assist/Dockerfile .

# From the dci-mcp-server repo:
podman build -t dci-mcp-server -f sandbox/dci-mcp-server/Dockerfile /path/to/dci-mcp-server/
```

## Usage

```bash
# Create an instance
ai-assist /sandbox init my-instance                        # all features
ai-assist /sandbox init my-instance --features=ssh,git     # only ssh and git

# Configure credentials
cp ~/.ai-assist-instances/my-instance/.env.example ~/.ai-assist-instances/my-instance/.env
# Edit .env with your credentials

# Edit identity
$EDITOR ~/.ai-assist-instances/my-instance/sandbox/.ai-assist/identity.yaml

# Run
ai-assist /sandbox run my-instance /query "hello"
ai-assist /sandbox run my-instance /monitor
ai-assist /sandbox run my-instance /run workflow.awl

# Manage
ai-assist /sandbox list
ai-assist /sandbox stop my-instance
ai-assist /sandbox delete my-instance
```

## Architecture

Each instance is a podman-compose stack with per-service secret isolation:

```
instance-dir/
  compose.yaml    <- host-only (orchestration + credential mapping)
  .env            <- host-only (all secrets, never mounted)
  sandbox/        <- bind-mounted as /workspace in ai-assist container
    .ai-assist/   <- config dir (no credentials)
    reports/      <- output reports
    workspace/    <- agent working area
```

Credentials reach each container exclusively via compose environment injection.
The ai-assist container never sees MCP server credentials.
