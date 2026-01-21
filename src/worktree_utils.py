"""Shared worktree utilities for both LATS and other agent systems."""

import os

from src.state import AgentState
from src.utils.git_manager import WorktreeManager


def setup_worktree(state: AgentState) -> dict:
    """Set up a git worktree environment for isolated agent execution."""
    print("🚀 Setting up worktree environment...")

    worktree_manager = WorktreeManager(
        base_repo_path=os.environ.get("SGLANG_DIR", ""),
        worktrees_dir=os.environ.get("WORKTREES_DIR", ""),
    )

    print("📁 WorktreeManager initialized")
    repo_path = worktree_manager.create_environment(base_branch="add-moe-lora-support")

    print(f"🏁 Worktree setup complete: {repo_path}")

    return {
        "repo_path": repo_path,
        "context": {
            **state.get("context", {}),
            # Don't store WorktreeManager object in state - it's not serializable
            # Store only the paths needed for cleanup
            "worktree_base_repo": os.environ.get("BASE_REPO_PATH", "/path/to/sglang/repo"),
            "worktree_worktrees_dir": os.environ.get("WORKTREES_DIR", "/path/to/worktrees"),
        },
    }


def cleanup_worktree(state: AgentState) -> dict:
    """Clean up the git worktree environment."""
    context = state.get("context", {})
    repo_path = state.get("repo_path")

    # Recreate WorktreeManager from stored paths instead of retrieving from state
    base_repo = context.get("worktree_base_repo", os.environ.get("SGLANG_DIR", ""))
    worktrees_dir = context.get("worktree_worktrees_dir", os.environ.get("WORKTREES_DIR", ""))

    if repo_path:
        worktree_manager = WorktreeManager(
            base_repo_path=base_repo,
            worktrees_dir=worktrees_dir,
        )
        worktree_manager.cleanup_environment(repo_path)

    return {"status": "completed"}
