# mcp-copilotcli-history

An MCP server that provides tools for searching through GitHub Copilot's conversation history stored in `~/.copilot/session-state/`.

## Features

- **Search Sessions**: Full-text search across all Copilot conversations
- **List Recent Sessions**: View recent sessions with titles extracted from first user message
- **Session Statistics**: Get aggregate stats about your Copilot usage
- **View Conversations**: Read the full conversation from any session
- **Search by File**: Find sessions that referenced specific files
- **Search Tool Usage**: Find examples of how tools were used
- **Find Empty Sessions**: Identify empty, abandoned, or trivially short sessions for cleanup
- **Delete Sessions**: Safely remove low-value session directories (dry-run by default)

## Installation

### Using uvx (recommended)

When using [`uv`](https://docs.astral.sh/uv/) no specific installation is needed:

```bash
uvx mcp-copilotcli-history
```

### Using pip

```bash
pip install mcp-copilotcli-history
```

After installation, run as a module:

```bash
python -m mcp_copilotcli_history
```

## Configuration

### Configure for Claude Desktop

Add to your `claude_desktop_config.json`:

<details>
<summary>Using uvx</summary>

```json
{
  "mcpServers": {
    "copilot-history": {
      "command": "uvx",
      "args": ["mcp-copilotcli-history"]
    }
  }
}
```
</details>

<details>
<summary>Using pip installation</summary>

```json
{
  "mcpServers": {
    "copilot-history": {
      "command": "python",
      "args": ["-m", "mcp_copilotcli_history"]
    }
  }
}
```
</details>

### Configure for VS Code

Add the configuration to your user-level MCP configuration file. Open the Command Palette (`Ctrl + Shift + P`) and run `MCP: Open User Configuration`.

<details>
<summary>Using uvx</summary>

```json
{
  "servers": {
    "copilot-history": {
      "command": "uvx",
      "args": ["mcp-copilotcli-history"]
    }
  }
}
```
</details>

<details>
<summary>Using pip installation</summary>

```json
{
  "servers": {
    "copilot-history": {
      "command": "python",
      "args": ["-m", "mcp_copilotcli_history"]
    }
  }
}
```
</details>

### Configure for Zed

Add to your Zed `settings.json`:

```json
"context_servers": {
  "copilot-history": {
    "command": "uvx",
    "args": ["mcp-copilotcli-history"]
  }
}
```

## Available Tools

### search_sessions

Search through all Copilot session history for a pattern.

**Arguments:**
- `query` (required): Search term or regex pattern
- `event_type` (optional): Filter by event type (user.message, assistant.message, etc.)
- `max_results` (optional): Maximum results to return (default: 20)
- `case_sensitive` (optional): Case-sensitive matching (default: false)

### list_recent_sessions

List the most recent Copilot sessions with their titles.

**Arguments:**
- `limit` (optional): Maximum sessions to return (default: 10)

### get_session_stats

Get statistics about all Copilot session history.

### get_session_conversation

Get the conversation from a specific session.

**Arguments:**
- `session_id` (required): Session ID (full or partial)
- `include_tool_calls` (optional): Include tool call details (default: false)
- `max_messages` (optional): Maximum messages to return (default: 50)

### search_by_file_path

Find sessions that referenced a specific file or path pattern.

**Arguments:**
- `file_pattern` (required): File path or pattern to search for
- `max_results` (optional): Maximum results (default: 20)

### search_tool_usage

Find sessions where specific tools were used.

**Arguments:**
- `tool_name` (optional): Tool name to filter by
- `max_results` (optional): Maximum results (default: 20)

### find_empty_sessions

Scan all session history and identify sessions that are empty, abandoned, or trivially short.

**Arguments:**
- `include_trivial` (optional): Include trivially short sessions (default: true)
- `max_content_length` (optional): Character threshold for "trivial" classification (default: 200)

### delete_sessions

Delete session directories from Copilot history. Performs a dry run by default — set `confirm` to true to actually delete.

**Arguments:**
- `session_ids` (required): List of session IDs to delete
- `confirm` (optional): Set to true to actually delete; false for dry run (default: false)

## Example Use Cases

Once configured, you can ask your AI assistant questions like:

- "Search my Copilot history for discussions about terraform"
- "What sessions did I have this week?"
- "Find conversations where I worked on main.py"
- "How did I use the create_file tool before?"
- "Show me the conversation from session abc123"

### Session Cleanup Workflow

You can use the session cleanup tools to keep your history tidy:

1. **Find low-value sessions**: Ask "Find any empty or abandoned sessions in my Copilot history"
2. **Review the results**: The tool returns each session's category (`empty`, `abandoned`, or `trivial`), a reason, and the session title so you can decide what to keep
3. **Delete what you don't need**: Ask "Delete sessions aaa111, bbb222, ccc333" — this does a dry run first showing what would be deleted and bytes freed
4. **Confirm deletion**: Ask "Delete those sessions with confirm set to true" to actually remove them

## Debugging

Use the MCP inspector to debug:

```bash
npx @modelcontextprotocol/inspector uvx mcp-copilotcli-history
```

## License

MIT
