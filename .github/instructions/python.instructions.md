---
applyTo: "**/*.py"
---

# Python Code Standards

## Type Hints

- Use `Optional[T]` for nullable parameters
- Use `list[T]`, `dict[K, V]` (lowercase) for Python 3.10+ style
- Import types from `typing` module when needed

## MCP Tools

- Decorate tools with `@mcp.tool()`
- Include comprehensive docstrings with Args/Returns sections
- Return `list[dict]` or `dict` for tool responses
- Include error info in return value: `{"error": "message"}`

## Style (PEP 8)

- Use double quotes for strings
- 4 spaces for indentation (no tabs)
- Max line length: 88 characters (Black formatter)
- Two blank lines before top-level functions/classes
- One blank line between methods

## Imports (PEP 8)

- Group imports: stdlib, third-party, local (blank line between groups)
- Sort alphabetically within groups
- Use absolute imports over relative imports
- Avoid wildcard imports (`from x import *`)

## Naming (PEP 8)

- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Prefix private items with underscore (`_private_func`)

## Best Practices

- Add docstrings to all public functions
- Use `defaultdict` for counting/grouping operations
- Prefer `pathlib.Path` over `os.path`
- Use context managers (`with`) for file operations
- Handle exceptions specifically, not bare `except:`
