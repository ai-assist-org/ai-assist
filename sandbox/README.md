# Sandbox Container Images

Container images for running ai-assist instances in isolated podman-compose stacks.

## Build

```bash
# Or use make targets:
make sandbox-build          # base image + dci-mcp-server
make sandbox-build-dev      # dev profile (Go, Ansible, uv, shellcheck, yamllint)

# Manual build:
podman build -t ai-assist-sandbox -f sandbox/ai-assist/Dockerfile .
podman build -t dci-mcp-server -f /path/to/dci-mcp-server/Containerfile.sse /path/to/dci-mcp-server/
podman build -t ai-assist-dev -f sandbox/profiles/dev/Dockerfile .
```

## Usage

```bash
# Create an instance
ai-assist /sandbox init my-instance                        # all features, base image
ai-assist /sandbox init my-instance --features=ssh,git     # only ssh and git
ai-assist /sandbox init my-instance --image=ai-assist-dev  # use dev image

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

# Install as a persistent systemd service (runs /monitor)
ai-assist /sandbox service my-instance install
ai-assist /sandbox service my-instance status
ai-assist /sandbox service my-instance logs -f
ai-assist /sandbox service my-instance remove
```

**Note:** Systemd user services only run while you're logged in. To start at boot:
```bash
sudo loginctl enable-linger $USER
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

## Image Profiles

The base image (`ai-assist-sandbox`) contains only runtime essentials. Custom profiles
in `sandbox/profiles/` extend it with workload-specific tools:

| Profile | Image | Tools |
|---------|-------|-------|
| base | `ai-assist-sandbox` | jq, git, ssh, gpg, gh |
| dev | `ai-assist-dev` | + Go, Ansible, uv, shellcheck, yamllint, gcc, make, cmake |

Create your own profile: add a `sandbox/profiles/<name>/Dockerfile` starting with
`FROM ai-assist-sandbox:latest`. Do not set `USER` — the sandbox uses `userns_mode: keep-id`.

## Testing

```bash
make test-integration    # builds images and runs integration tests
```
