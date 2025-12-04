"""
MCP Server for searching GitHub Copilot conversation history.

This server provides tools for searching through GitHub Copilot's session
history stored in ~/.copilot/session-state/ JSONL files.
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    name="Copilot Session History",
    instructions="""
    This server provides tools to search and explore GitHub Copilot's conversation history.
    Use these tools when you need to:
    - Find past conversations about specific topics
    - Look up how something was implemented before
    - Review previous discussions or decisions
    - Get statistics about Copilot usage patterns
    """,
)


# ============================================================================
# Configuration and Helpers
# ============================================================================


def get_session_state_dir() -> Path:
    """Get the Copilot session state directory path."""
    return Path.home() / ".copilot" / "session-state"


def list_session_files(session_dir: Optional[Path] = None) -> list[Path]:
    """List all JSONL session files, sorted by modification time (newest first)."""
    if session_dir is None:
        session_dir = get_session_state_dir()
    if not session_dir.exists():
        return []
    files = list(session_dir.glob("*.jsonl"))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


# Cache for session titles to avoid re-reading files
_session_titles_cache: dict[str, str] = {}


def get_session_title(file_path: Path, max_length: int = 80) -> str:
    """Extract the first user message as the session title."""
    session_id = file_path.stem

    if session_id in _session_titles_cache:
        return _session_titles_cache[session_id]

    title = "(no user message)"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "user.message":
                        content = entry.get("data", {}).get("content", "")
                        content = re.sub(
                            r"<current_datetime>.*?</current_datetime>", "", content
                        )
                        content = " ".join(content.split())
                        if content:
                            title = (
                                content[:max_length] + "..."
                                if len(content) > max_length
                                else content
                            )
                            break
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    _session_titles_cache[session_id] = title
    return title


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return ts[:16] if ts else "unknown"


def extract_searchable_content(entry: dict) -> str:
    """Extract searchable text content from a session entry."""
    content_parts = []

    entry_type = entry.get("type", "")
    data = entry.get("data", {})

    if entry_type == "user.message":
        content_parts.append(data.get("content", ""))
        for attachment in data.get("attachments", []):
            content_parts.append(attachment.get("displayName", ""))
            content_parts.append(attachment.get("path", ""))

    elif entry_type == "assistant.message":
        content_parts.append(data.get("content", ""))
        for tool_req in data.get("toolRequests", []):
            content_parts.append(tool_req.get("name", ""))
            args = tool_req.get("arguments", {})
            if isinstance(args, dict):
                content_parts.extend(str(v) for v in args.values())

    elif entry_type == "tool.result":
        result = data.get("result", {})
        if isinstance(result, dict):
            content_parts.append(str(result.get("content", "")))
        elif isinstance(result, str):
            content_parts.append(result)

    elif entry_type == "session.start":
        content_parts.append(data.get("sessionId", ""))
        content_parts.append(data.get("selectedModel", ""))

    if not content_parts:
        content_parts.append(json.dumps(data))

    return " ".join(filter(None, content_parts))


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
def search_sessions(
    query: str,
    event_type: Optional[str] = None,
    max_results: int = 20,
    case_sensitive: bool = False,
) -> list[dict]:
    """
    Search through all Copilot session history for a pattern.

    This tool searches across all stored Copilot conversations to find
    messages, tool calls, and other events matching your query. Use this
    to find past discussions, code snippets, or decisions.

    Args:
        query: The search term or regex pattern to find
        event_type: Optional filter for event type (user.message, assistant.message, tool.result)
        max_results: Maximum number of results to return (default: 20)
        case_sensitive: Whether to perform case-sensitive matching (default: False)

    Returns:
        List of matching entries with session context and content snippets
    """
    session_dir = get_session_state_dir()
    if not session_dir.exists():
        return [{"error": f"Session directory not found: {session_dir}"}]

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error as e:
        return [{"error": f"Invalid regex pattern: {e}"}]

    files = list_session_files(session_dir)
    results = []
    event_types = [event_type] if event_type else None

    for file_path in files:
        if len(results) >= max_results:
            break

        session_id = file_path.stem
        session_title = get_session_title(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(results) >= max_results:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    entry_type = entry.get("type", "")

                    if event_types and entry_type not in event_types:
                        continue

                    content = extract_searchable_content(entry)
                    match = pattern.search(content)

                    if match:
                        # Create a content snippet around the match
                        match_pos = match.start()
                        start = max(0, match_pos - 100)
                        end = min(len(content), match_pos + len(match.group()) + 100)
                        snippet = content[start:end]
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."

                        results.append(
                            {
                                "session_id": session_id[:8] + "...",
                                "session_title": session_title,
                                "event_type": entry_type,
                                "timestamp": format_timestamp(
                                    entry.get("timestamp", "")
                                ),
                                "matched_text": match.group(),
                                "content_snippet": snippet,
                            }
                        )
        except Exception:
            continue

    if not results:
        return [{"message": f"No results found for '{query}'"}]

    return results


@mcp.tool()
def list_recent_sessions(limit: int = 10) -> list[dict]:
    """
    List the most recent Copilot sessions with their titles.

    Use this to get an overview of recent conversations and find
    sessions to explore further.

    Args:
        limit: Maximum number of sessions to return (default: 10)

    Returns:
        List of recent sessions with metadata (id, title, date, model, size)
    """
    session_dir = get_session_state_dir()
    if not session_dir.exists():
        return [{"error": f"Session directory not found: {session_dir}"}]

    files = list_session_files(session_dir)[:limit]
    sessions = []

    for file_path in files:
        session_id = file_path.stem
        size_kb = file_path.stat().st_size / 1024

        start_time = "unknown"
        model = "default"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    entry = json.loads(first_line)
                    start_time = format_timestamp(entry.get("timestamp", ""))
                    model = entry.get("data", {}).get("selectedModel", "default")
        except Exception:
            pass

        title = get_session_title(file_path)

        sessions.append(
            {
                "session_id": session_id,
                "title": title,
                "started": start_time,
                "model": model,
                "size_kb": round(size_kb, 1),
            }
        )

    return sessions


@mcp.tool()
def get_session_stats() -> dict:
    """
    Get statistics about all Copilot session history.

    Returns aggregate information about stored sessions including
    total count, size, date range, event types, and models used.

    Returns:
        Dictionary with session statistics
    """
    session_dir = get_session_state_dir()
    if not session_dir.exists():
        return {"error": f"Session directory not found: {session_dir}"}

    files = list_session_files(session_dir)

    stats: dict = {
        "total_sessions": len(files),
        "total_size_mb": 0.0,
        "total_entries": 0,
        "event_types": defaultdict(int),
        "models_used": defaultdict(int),
        "date_range": {"oldest": None, "newest": None},
    }

    for file_path in files:
        stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        stats["total_entries"] += 1

                        event_type = entry.get("type", "unknown")
                        stats["event_types"][event_type] += 1

                        if event_type == "session.start":
                            model = entry.get("data", {}).get(
                                "selectedModel", "default"
                            )
                            stats["models_used"][model] += 1

                        ts = entry.get("timestamp")
                        if ts:
                            if (
                                stats["date_range"]["oldest"] is None
                                or ts < stats["date_range"]["oldest"]
                            ):
                                stats["date_range"]["oldest"] = ts
                            if (
                                stats["date_range"]["newest"] is None
                                or ts > stats["date_range"]["newest"]
                            ):
                                stats["date_range"]["newest"] = ts

                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    # Format for output
    return {
        "total_sessions": stats["total_sessions"],
        "total_size_mb": round(stats["total_size_mb"], 2),
        "total_entries": stats["total_entries"],
        "date_range": {
            "oldest": format_timestamp(stats["date_range"]["oldest"])
            if stats["date_range"]["oldest"]
            else None,
            "newest": format_timestamp(stats["date_range"]["newest"])
            if stats["date_range"]["newest"]
            else None,
        },
        "event_types": dict(
            sorted(stats["event_types"].items(), key=lambda x: -x[1])
        ),
        "models_used": dict(sorted(stats["models_used"].items(), key=lambda x: -x[1])),
    }


@mcp.tool()
def get_session_conversation(
    session_id: str,
    include_tool_calls: bool = False,
    max_messages: int = 50,
) -> list[dict]:
    """
    Get the conversation from a specific session.

    Retrieves the user and assistant messages from a session in
    chronological order, allowing you to review a past conversation.

    Args:
        session_id: The session ID (full or partial) to retrieve
        include_tool_calls: Whether to include tool call details (default: False)
        max_messages: Maximum number of messages to return (default: 50)

    Returns:
        List of messages from the session in chronological order
    """
    session_dir = get_session_state_dir()
    if not session_dir.exists():
        return [{"error": f"Session directory not found: {session_dir}"}]

    # Find the session file (supports partial ID matching)
    files = list(session_dir.glob(f"{session_id}*.jsonl"))
    if not files:
        return [{"error": f"Session not found: {session_id}"}]

    file_path = files[0]
    messages = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(messages) >= max_messages:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    event_type = entry.get("type", "")
                    timestamp = format_timestamp(entry.get("timestamp", ""))
                    data = entry.get("data", {})

                    if event_type == "user.message":
                        content = data.get("content", "")
                        # Clean up content
                        content = re.sub(
                            r"<current_datetime>.*?</current_datetime>", "", content
                        )
                        content = content.strip()

                        msg: dict = {
                            "role": "user",
                            "timestamp": timestamp,
                            "content": content,
                        }

                        # Include attachment info if present
                        attachments = data.get("attachments", [])
                        if attachments:
                            msg["attachments"] = [
                                a.get("displayName", a.get("path", "unknown"))
                                for a in attachments
                            ]

                        messages.append(msg)

                    elif event_type == "assistant.message":
                        content = data.get("content", "")
                        if content:
                            msg = {
                                "role": "assistant",
                                "timestamp": timestamp,
                                "content": content,
                            }

                            if include_tool_calls:
                                tools = data.get("toolRequests", [])
                                if tools:
                                    msg["tool_calls"] = [
                                        {"name": t.get("name", "unknown")}
                                        for t in tools
                                    ]

                            messages.append(msg)

                    elif event_type == "session.start":
                        model = data.get("selectedModel", "default")
                        messages.append(
                            {
                                "role": "system",
                                "timestamp": timestamp,
                                "content": f"Session started with model: {model}",
                            }
                        )

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        return [{"error": f"Error reading session: {e}"}]

    return messages


@mcp.tool()
def search_by_file_path(
    file_pattern: str,
    max_results: int = 20,
) -> list[dict]:
    """
    Find sessions that referenced a specific file or path pattern.

    Searches through session history for mentions of file paths,
    useful for finding past work on specific files or directories.

    Args:
        file_pattern: File path or pattern to search for (e.g., "main.py", "src/")
        max_results: Maximum number of results to return (default: 20)

    Returns:
        List of sessions and entries that referenced the file pattern
    """
    return search_sessions(
        query=re.escape(file_pattern),
        max_results=max_results,
        case_sensitive=False,
    )


@mcp.tool()
def search_tool_usage(
    tool_name: Optional[str] = None,
    max_results: int = 20,
) -> list[dict]:
    """
    Find sessions where specific tools were used.

    Search for tool invocations in past sessions, optionally filtered
    by tool name. Useful for finding examples of how tools were used.

    Args:
        tool_name: Optional tool name to filter by (e.g., "create_file", "run_in_terminal")
        max_results: Maximum number of results to return (default: 20)

    Returns:
        List of tool usage instances with context
    """
    session_dir = get_session_state_dir()
    if not session_dir.exists():
        return [{"error": f"Session directory not found: {session_dir}"}]

    files = list_session_files(session_dir)
    results = []

    for file_path in files:
        if len(results) >= max_results:
            break

        session_id = file_path.stem
        session_title = get_session_title(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(results) >= max_results:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if entry.get("type") != "assistant.message":
                        continue

                    data = entry.get("data", {})
                    tool_requests = data.get("toolRequests", [])

                    for tool_req in tool_requests:
                        name = tool_req.get("name", "")
                        if tool_name is None or tool_name.lower() in name.lower():
                            args = tool_req.get("arguments", {})
                            # Summarize arguments
                            args_summary = {}
                            for k, v in (
                                args if isinstance(args, dict) else {}
                            ).items():
                                v_str = str(v)
                                args_summary[k] = (
                                    v_str[:100] + "..." if len(v_str) > 100 else v_str
                                )

                            results.append(
                                {
                                    "session_id": session_id[:8] + "...",
                                    "session_title": session_title,
                                    "timestamp": format_timestamp(
                                        entry.get("timestamp", "")
                                    ),
                                    "tool_name": name,
                                    "arguments": args_summary,
                                }
                            )

                            if len(results) >= max_results:
                                break

        except Exception:
            continue

    if not results:
        msg = "No tool usage found"
        if tool_name:
            msg += f" for tool '{tool_name}'"
        return [{"message": msg}]

    return results


# ============================================================================
# Main entry point
# ============================================================================


def main():
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
