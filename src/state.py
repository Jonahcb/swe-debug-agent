"""Agent state definitions for SGLang LoRA MoE debugging."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state for the SGLang LoRA MoE debug workflow.

    The debug loop tracks:
    - messages: Conversation history between agents
    - task: The debugging task (hardcoded to LoRA MoE test)
    - context: Debug state (test results, files modified, etc.)
    - current_agent: Which agent is currently active
    - status: Overall workflow status
    - iteration_count: Total agent turns taken
    """

    # Conversation history (auto-accumulates messages)
    messages: Annotated[list, add_messages]

    # The task description
    task: str

    # Context for debug loop - tracks:
    # - test_status: "passed" | "failed" | None
    # - last_test_error: str (error message from last failed test)
    # - debug_cycle: int (number of identify->fix->test cycles)
    # - files_modified: list[str] (files changed during debugging)
    # - bugs_found: list[dict] (bugs identified with details)
    context: dict[str, Any]

    # Current agent running
    current_agent: str

    # Workflow status: "running" | "complete" | "failed"
    status: str

    # Total iterations (agent turns)
    iteration_count: int

    # Global search tool call limit (1 per program run)
    search_calls_used: int

    # Path to the isolated worktree for this agent run
    repo_path: str
