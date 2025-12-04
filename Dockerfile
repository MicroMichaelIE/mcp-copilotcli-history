# syntax=docker/dockerfile:1

# Build stage
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy source files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Build the wheel
RUN python -m build --wheel

# Runtime stage
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="mcp-copilotcli-history"
LABEL org.opencontainers.image.description="MCP server for searching GitHub Copilot conversation history"
LABEL org.opencontainers.image.source="https://github.com/MicroMichaelIE/mcp-copilotcli-history"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcp

WORKDIR /app

# Copy the built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the package
RUN pip install --no-cache-dir *.whl && rm -f *.whl

# Switch to non-root user
USER mcp

# The MCP server communicates via stdio
ENTRYPOINT ["mcp-copilotcli-history"]
