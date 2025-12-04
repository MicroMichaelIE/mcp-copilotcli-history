# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-12-04

### Added

- Support for `SESSION_STATE_DIR` environment variable to configure session directory path
- Enables scoped filesystem access when used with MCP inputs

## [0.1.0] - 2025-12-04

### Added

- Initial release of mcp-copilotcli-history MCP server
- `search_sessions` tool - Full-text search across all Copilot conversations
- `list_recent_sessions` tool - View recent sessions with titles
- `get_session_stats` tool - Aggregate statistics about Copilot usage
- `get_session_conversation` tool - Read full conversation from any session
- `search_by_file_path` tool - Find sessions that referenced specific files
- `search_tool_usage` tool - Find examples of tool usage in past sessions
- Configuration examples for Claude Desktop, VS Code, and Zed
- Dockerfile for containerized deployment
- GitHub Actions for CI/CD (PyPI and Docker publishing)

[Unreleased]: https://github.com/MicroMichaelIE/mcp-copilotcli-history/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MicroMichaelIE/mcp-copilotcli-history/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MicroMichaelIE/mcp-copilotcli-history/releases/tag/v0.1.0
