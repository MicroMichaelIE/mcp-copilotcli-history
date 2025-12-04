# Copilot Instructions

## Project Overview

This is an MCP (Model Context Protocol) server that provides tools for searching GitHub Copilot CLI conversation history stored in `~/.copilot/session-state/` JSONL files.

## Tech Stack

- Python 3.10+
- FastMCP (`mcp` package) for MCP server implementation
- Hatchling for build system

## Key Commands

- Install: `uv pip install -e .`
- Run server: `mcp-copilotcli-history` or `python -m mcp_copilotcli_history`

## Code Guidelines

- Use type hints for all function signatures
- Keep functions focused and single-purpose
- Handle file I/O errors gracefully with try/except
- Use `Path` from `pathlib` for file operations
- Return meaningful error dictionaries instead of raising exceptions in MCP tools
