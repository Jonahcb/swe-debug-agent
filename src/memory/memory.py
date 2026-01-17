"""Memory configuration for deep agents using disk-based storage in git worktree.

Deep agents provides built-in long-term memory via:
- StoreBackend: Durable cross-thread storage using LangGraph's BaseStore
- Filesystem middleware: Context management with read/write tools
- Disk persistence: Memory stored in agent's git worktree directory using SQLite

See: https://docs.langchain.com/oss/python/deepagents/backends
"""

import os
from pathlib import Path

from langgraph.store.sqlite import SqliteStore


def create_memory_store():
    """Create a disk-based memory store in the agent's git worktree.

    Memory is persisted to disk in the worktree directory for durability across
    agent restarts and sessions. The worktree path is determined by the
    WORKTREE_REPO_PATH environment variable.

    Uses SQLite for persistent storage.
    """
    # Get the worktree path where memory should be stored
    worktree_path = os.environ.get("WORKTREE_REPO_PATH")
    if not worktree_path:
        print("Warning: WORKTREE_REPO_PATH not set, falling back to InMemoryStore")
        from langgraph.store.memory import InMemoryStore

        return InMemoryStore()

    # Create memory directory in the worktree
    memory_dir = Path(worktree_path) / ".agent_memory"
    memory_dir.mkdir(exist_ok=True)

    # Create SQLite database file in the worktree
    db_path = memory_dir / "agent_memory.db"

    # Return SQLite store for persistent disk memory
    return SqliteStore.from_file(str(db_path))


# Shared store instance for cross-thread memory
store = create_memory_store()
