"""Search tool call limiter for controlling API usage during testing."""

from collections.abc import Callable
from typing import Any


class SearchLimiter:
    """Limits search tool calls globally across all agents."""

    def __init__(self, max_search_calls: int = 1):
        self.max_search_calls = max_search_calls
        self.current_search_calls = 0

    def can_use_search_tool(self) -> bool:
        """Check if a search tool can be used."""
        return self.current_search_calls < self.max_search_calls

    def increment_search_calls(self) -> None:
        """Increment the search call counter."""
        self.current_search_calls += 1

    def wrap_tool(self, tool) -> Any:
        """Wrap a tool to limit search calls."""
        if not self._is_search_tool(tool):
            return tool

        # Handle BaseTool objects
        if hasattr(tool, "invoke"):
            original_invoke = tool.invoke

            def limited_invoke(*args, **kwargs):
                if not self.can_use_search_tool():
                    return "Error: Search tool limit reached (1 search call per program run). Cannot perform additional searches."
                self.increment_search_calls()
                return original_invoke(*args, **kwargs)

            tool.invoke = limited_invoke
            return tool
        else:
            # Handle raw functions - wrap the function directly
            original_func = tool

            def limited_func(*args, **kwargs):
                if not self.can_use_search_tool():
                    return "Error: Search tool limit reached (1 search call per program run). Cannot perform additional searches."
                self.increment_search_calls()
                return original_func(*args, **kwargs)

            return limited_func

    def _is_search_tool(self, tool) -> bool:
        """Check if a tool is a search tool based on its name."""
        search_tool_names = [
            "brave_search",
            "web_search",
            "google_search",
            "search_web",
            "search_internet",
            "gpt_researcher",
            "research",
            "search_files",  # file system search
        ]

        # Handle BaseTool objects
        if hasattr(tool, "name"):
            tool_name = tool.name.lower()
            return any(search_name in tool_name for search_name in search_tool_names)
        # Handle raw functions - check function name
        elif hasattr(tool, "__name__"):
            tool_name = tool.__name__.lower()
            return any(search_name in tool_name for search_name in search_tool_names)
        else:
            # Unknown tool type - assume not a search tool
            return False


# Global limiter instance
search_limiter = SearchLimiter(max_search_calls=1)
