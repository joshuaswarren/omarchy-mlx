#!/usr/bin/env python3
"""Minimal MCP stdio server exposing one `echo` tool (mcp 2.x API, which oMLX
pins as mcp>=2,<3). Used by api_parity.sh section a18 via --mcp-config."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("parity-echo")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back, prefixed with echo:"""
    return f"echo:{text}"


if __name__ == "__main__":
    mcp.run()
