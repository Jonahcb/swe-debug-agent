"""MCP (Model Context Protocol) client for connecting to various MCP servers."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


# Pre-configured MCP servers
MCP_SERVERS = {
    "filesystem": MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/"],
    ),
    "terminal": MCPServerConfig(
        name="terminal",
        command="npx",
        args=["-y", "@anthropic/server-terminal"],
    ),
    "github": MCPServerConfig(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": ""},  # Set via environment
    ),
    "python-debug": MCPServerConfig(
        name="python-debug",
        command="uvx",
        args=["python-debug-mcp-server"],
    ),
    "gpt-researcher": MCPServerConfig(
        name="gpt-researcher",
        command="uvx",
        args=["gpt-researcher-mcp"],
    ),
    "brave-search": MCPServerConfig(
        name="brave-search",
        command="npx",
        args=["-y", "@anthropic/mcp-server-brave-search"],
        env={"BRAVE_API_KEY": ""},  # Set via environment
    ),
}


class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._tools_cache: dict[str, list[dict]] = {}

    @asynccontextmanager
    async def connect(self, server_name: str, config: MCPServerConfig | None = None):
        """Connect to an MCP server."""
        if config is None:
            config = MCP_SERVERS.get(server_name)
            if config is None:
                raise ValueError(f"Unknown server: {server_name}")

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._sessions[server_name] = session
                try:
                    yield session
                finally:
                    self._sessions.pop(server_name, None)

    async def list_tools(self, session: ClientSession) -> list[dict]:
        """List available tools from a connected server."""
        result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def call_tool(
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Call a tool on a connected server."""
        result = await session.call_tool(tool_name, arguments)
        return result.content

    async def list_resources(self, session: ClientSession) -> list[dict]:
        """List available resources from a connected server."""
        result = await session.list_resources()
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mimeType,
            }
            for resource in result.resources
        ]

    async def read_resource(self, session: ClientSession, uri: str) -> Any:
        """Read a resource from a connected server."""
        result = await session.read_resource(uri)
        return result.contents


class MCPToolRunner:
    """Simplified interface for running MCP tools."""

    def __init__(self):
        self.client = MCPClient()

    async def run(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Run a tool on an MCP server."""
        async with self.client.connect(server_name) as session:
            return await self.client.call_tool(session, tool_name, arguments)

    async def get_tools(self, server_name: str) -> list[dict]:
        """Get available tools from an MCP server."""
        async with self.client.connect(server_name) as session:
            return await self.client.list_tools(session)
