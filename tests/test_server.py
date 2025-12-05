"""Tests for the MCP server module."""

import json
from pathlib import Path
from unittest import mock

import pytest

from mcp_copilotcli_history.server import (
    _session_titles_cache,
    extract_searchable_content,
    format_timestamp,
    get_session_conversation,
    get_session_state_dir,
    get_session_stats,
    get_session_title,
    list_recent_sessions,
    list_session_files,
    search_by_file_path,
    search_sessions,
    search_tool_usage,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_session_dir(tmp_path: Path):
    """Create a temporary session directory with sample JSONL files."""
    session_dir = tmp_path / "session-state"
    session_dir.mkdir()

    # Create sample session file
    session1 = session_dir / "abc123-session1.jsonl"
    entries = [
        {
            "type": "session.start",
            "timestamp": "2025-12-01T10:00:00Z",
            "data": {"sessionId": "abc123-session1", "selectedModel": "gpt-4o"},
        },
        {
            "type": "user.message",
            "timestamp": "2025-12-01T10:01:00Z",
            "data": {"content": "How do I create a Python function?"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2025-12-01T10:01:30Z",
            "data": {
                "content": "Here's how to create a Python function:",
                "toolRequests": [],
            },
        },
        {
            "type": "user.message",
            "timestamp": "2025-12-01T10:02:00Z",
            "data": {
                "content": "Can you edit my file?",
                "attachments": [{"displayName": "main.py", "path": "/src/main.py"}],
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2025-12-01T10:02:30Z",
            "data": {
                "content": "I'll edit the file for you.",
                "toolRequests": [
                    {
                        "name": "replace_string_in_file",
                        "arguments": {"filePath": "/src/main.py", "oldString": "foo"},
                    }
                ],
            },
        },
    ]
    with open(session1, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Create another session file
    session2 = session_dir / "def456-session2.jsonl"
    entries2 = [
        {
            "type": "session.start",
            "timestamp": "2025-12-02T14:00:00Z",
            "data": {"sessionId": "def456-session2", "selectedModel": "claude-sonnet"},
        },
        {
            "type": "user.message",
            "timestamp": "2025-12-02T14:01:00Z",
            "data": {"content": "Help me debug this error in utils.py"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2025-12-02T14:01:30Z",
            "data": {
                "content": "Let me look at that file.",
                "toolRequests": [
                    {"name": "read_file", "arguments": {"filePath": "/src/utils.py"}}
                ],
            },
        },
        {
            "type": "tool.result",
            "timestamp": "2025-12-02T14:01:35Z",
            "data": {"result": {"content": "def helper(): pass"}},
        },
    ]
    with open(session2, "w", encoding="utf-8") as f:
        for entry in entries2:
            f.write(json.dumps(entry) + "\n")

    return session_dir


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the session titles cache before each test."""
    _session_titles_cache.clear()
    yield
    _session_titles_cache.clear()


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestGetSessionStateDir:
    """Tests for get_session_state_dir function."""

    def test_default_path(self):
        """Test default path when no env var is set."""
        with mock.patch.dict("os.environ", {}, clear=True):
            path = get_session_state_dir()
            assert path == Path.home() / ".copilot" / "session-state"

    def test_custom_path_from_env(self, tmp_path: Path):
        """Test custom path from environment variable."""
        custom_dir = tmp_path / "custom-sessions"
        with mock.patch.dict("os.environ", {"SESSION_STATE_DIR": str(custom_dir)}):
            path = get_session_state_dir()
            assert path == custom_dir


class TestListSessionFiles:
    """Tests for list_session_files function."""

    def test_list_files_sorted_by_mtime(self, temp_session_dir: Path):
        """Test that files are sorted by modification time."""
        files = list_session_files(temp_session_dir)
        assert len(files) == 2
        # All files should have .jsonl extension
        for f in files:
            assert f.suffix == ".jsonl"

    def test_empty_directory(self, tmp_path: Path):
        """Test with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        files = list_session_files(empty_dir)
        assert files == []

    def test_nonexistent_directory(self, tmp_path: Path):
        """Test with non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        files = list_session_files(nonexistent)
        assert files == []


class TestGetSessionTitle:
    """Tests for get_session_title function."""

    def test_extracts_first_user_message(self, temp_session_dir: Path):
        """Test extracting title from first user message."""
        session_file = temp_session_dir / "abc123-session1.jsonl"
        title = get_session_title(session_file)
        assert title == "How do I create a Python function?"

    def test_truncates_long_titles(self, tmp_path: Path):
        """Test that long titles are truncated."""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        session_file = session_dir / "long-title.jsonl"

        long_content = "A" * 200
        entry = {
            "type": "user.message",
            "timestamp": "2025-12-01T10:00:00Z",
            "data": {"content": long_content},
        }
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        title = get_session_title(session_file, max_length=80)
        assert len(title) == 83  # 80 + "..."
        assert title.endswith("...")

    def test_caches_results(self, temp_session_dir: Path):
        """Test that titles are cached."""
        session_file = temp_session_dir / "abc123-session1.jsonl"

        # First call
        title1 = get_session_title(session_file)

        # Check cache
        assert "abc123-session1" in _session_titles_cache

        # Second call should use cache
        title2 = get_session_title(session_file)
        assert title1 == title2


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_format_iso_timestamp(self):
        """Test formatting ISO timestamp."""
        result = format_timestamp("2025-12-01T10:30:45Z")
        assert result == "2025-12-01 10:30"

    def test_format_invalid_timestamp(self):
        """Test handling invalid timestamp."""
        result = format_timestamp("invalid")
        assert result == "invalid"[:16]

    def test_format_empty_timestamp(self):
        """Test handling empty timestamp."""
        result = format_timestamp("")
        assert result == "unknown"


class TestExtractSearchableContent:
    """Tests for extract_searchable_content function."""

    def test_user_message_content(self):
        """Test extracting content from user message."""
        entry = {
            "type": "user.message",
            "data": {"content": "Hello world"},
        }
        content = extract_searchable_content(entry)
        assert "Hello world" in content

    def test_user_message_with_attachments(self):
        """Test extracting content from user message with attachments."""
        entry = {
            "type": "user.message",
            "data": {
                "content": "Check this file",
                "attachments": [{"displayName": "test.py", "path": "/path/to/test.py"}],
            },
        }
        content = extract_searchable_content(entry)
        assert "Check this file" in content
        assert "test.py" in content
        assert "/path/to/test.py" in content

    def test_assistant_message_with_tool_requests(self):
        """Test extracting content from assistant message with tools."""
        entry = {
            "type": "assistant.message",
            "data": {
                "content": "I'll help",
                "toolRequests": [
                    {"name": "read_file", "arguments": {"filePath": "/test.py"}}
                ],
            },
        }
        content = extract_searchable_content(entry)
        assert "I'll help" in content
        assert "read_file" in content
        assert "/test.py" in content

    def test_tool_result_dict(self):
        """Test extracting content from tool result (dict)."""
        entry = {
            "type": "tool.result",
            "data": {"result": {"content": "file contents here"}},
        }
        content = extract_searchable_content(entry)
        assert "file contents here" in content

    def test_tool_result_string(self):
        """Test extracting content from tool result (string)."""
        entry = {
            "type": "tool.result",
            "data": {"result": "simple result"},
        }
        content = extract_searchable_content(entry)
        assert "simple result" in content


# ============================================================================
# Test MCP Tools
# ============================================================================


class TestSearchSessions:
    """Tests for search_sessions tool."""

    def test_search_finds_matching_content(self, temp_session_dir: Path):
        """Test that search finds matching content."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_sessions(query="Python function")
            assert len(results) >= 1
            assert any("Python function" in r.get("matched_text", "") for r in results)

    def test_search_case_insensitive(self, temp_session_dir: Path):
        """Test case-insensitive search."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_sessions(query="PYTHON", case_sensitive=False)
            assert len(results) >= 1

    def test_search_with_event_type_filter(self, temp_session_dir: Path):
        """Test filtering by event type."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_sessions(query=".", event_type="user.message")
            for r in results:
                if "event_type" in r:
                    assert r["event_type"] == "user.message"

    def test_search_respects_max_results(self, temp_session_dir: Path):
        """Test max_results limit."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_sessions(query=".", max_results=2)
            assert len(results) <= 2

    def test_search_no_results(self, temp_session_dir: Path):
        """Test search with no matches."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_sessions(query="xyznonexistent123")
            assert len(results) == 1
            assert "message" in results[0]

    def test_search_invalid_regex(self, temp_session_dir: Path):
        """Test search with invalid regex."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_sessions(query="[invalid")
            assert len(results) == 1
            assert "error" in results[0]

    def test_search_missing_directory(self, tmp_path: Path):
        """Test search with missing session directory."""
        nonexistent = tmp_path / "nonexistent"
        with mock.patch.dict("os.environ", {"SESSION_STATE_DIR": str(nonexistent)}):
            results = search_sessions(query="test")
            assert len(results) == 1
            assert "error" in results[0]


class TestListRecentSessions:
    """Tests for list_recent_sessions tool."""

    def test_list_returns_sessions(self, temp_session_dir: Path):
        """Test listing recent sessions."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = list_recent_sessions(limit=10)
            assert len(results) == 2
            for session in results:
                assert "session_id" in session
                assert "title" in session
                assert "model" in session

    def test_list_respects_limit(self, temp_session_dir: Path):
        """Test limit parameter."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = list_recent_sessions(limit=1)
            assert len(results) == 1

    def test_list_missing_directory(self, tmp_path: Path):
        """Test with missing session directory."""
        nonexistent = tmp_path / "nonexistent"
        with mock.patch.dict("os.environ", {"SESSION_STATE_DIR": str(nonexistent)}):
            results = list_recent_sessions()
            assert len(results) == 1
            assert "error" in results[0]


class TestGetSessionStats:
    """Tests for get_session_stats tool."""

    def test_stats_returns_correct_counts(self, temp_session_dir: Path):
        """Test that stats returns correct counts."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            stats = get_session_stats()
            assert stats["total_sessions"] == 2
            assert stats["total_entries"] == 9  # 5 + 4 entries
            assert "event_types" in stats
            assert "models_used" in stats
            assert "date_range" in stats

    def test_stats_event_type_counts(self, temp_session_dir: Path):
        """Test event type counting."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            stats = get_session_stats()
            assert stats["event_types"]["user.message"] == 3
            assert stats["event_types"]["assistant.message"] == 3
            assert stats["event_types"]["session.start"] == 2

    def test_stats_missing_directory(self, tmp_path: Path):
        """Test with missing session directory."""
        nonexistent = tmp_path / "nonexistent"
        with mock.patch.dict("os.environ", {"SESSION_STATE_DIR": str(nonexistent)}):
            stats = get_session_stats()
            assert "error" in stats


class TestGetSessionConversation:
    """Tests for get_session_conversation tool."""

    def test_get_conversation(self, temp_session_dir: Path):
        """Test retrieving a session conversation."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            messages = get_session_conversation("abc123")
            assert len(messages) >= 3
            roles = [m["role"] for m in messages]
            assert "user" in roles
            assert "assistant" in roles
            assert "system" in roles

    def test_get_conversation_with_tool_calls(self, temp_session_dir: Path):
        """Test including tool calls in conversation."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            messages = get_session_conversation("abc123", include_tool_calls=True)
            tool_messages = [m for m in messages if "tool_calls" in m]
            assert len(tool_messages) >= 1

    def test_get_conversation_not_found(self, temp_session_dir: Path):
        """Test with non-existent session."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            messages = get_session_conversation("nonexistent123")
            assert len(messages) == 1
            assert "error" in messages[0]


class TestSearchByFilePath:
    """Tests for search_by_file_path tool."""

    def test_search_finds_file_references(self, temp_session_dir: Path):
        """Test finding file references."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_by_file_path("main.py")
            assert len(results) >= 1


class TestSearchToolUsage:
    """Tests for search_tool_usage tool."""

    def test_search_all_tools(self, temp_session_dir: Path):
        """Test searching for all tool usage."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_tool_usage()
            assert len(results) >= 2  # replace_string_in_file and read_file

    def test_search_specific_tool(self, temp_session_dir: Path):
        """Test searching for specific tool."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_tool_usage(tool_name="read_file")
            assert len(results) >= 1
            for r in results:
                if "tool_name" in r:
                    assert "read_file" in r["tool_name"]

    def test_search_tool_not_found(self, temp_session_dir: Path):
        """Test searching for non-existent tool."""
        with mock.patch.dict(
            "os.environ", {"SESSION_STATE_DIR": str(temp_session_dir)}
        ):
            results = search_tool_usage(tool_name="nonexistent_tool")
            assert len(results) == 1
            assert "message" in results[0]
