"""MCP prompt reference parsing utility"""


def is_mcp_prompt(prompt: str) -> bool:
    return prompt.startswith("mcp://")


def parse_mcp_prompt(prompt: str) -> tuple[str, str]:
    """Parse 'mcp://server/prompt' into (server, prompt).

    Raises:
        ValueError: If format is invalid
    """
    if not is_mcp_prompt(prompt):
        raise ValueError("Not an MCP prompt reference")
    ref = prompt[6:]
    if "/" not in ref:
        raise ValueError("MCP prompt must be 'mcp://server/prompt'")
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("MCP prompt must be 'mcp://server/prompt'")
    return parts[0], parts[1]
