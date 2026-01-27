"""Shared worktree utilities for both LATS and other agent systems."""

import os
import subprocess

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
    repo_path = worktree_manager.create_environment(
        base_branch=os.environ.get("BASE_BRANCH", "main")
    )

    print(f"🏁 Worktree setup complete: {repo_path}")

    # Install sglang in editable mode in the worktree
    print("📦 Installing sglang in editable mode...")
    python_dir = os.path.join(repo_path, "python")

    if os.path.exists(python_dir):
        try:
            # 1. Move to the python package root in the worktree
            print(f"   📂 Changing to python directory: {python_dir}")

            # 2. Uninstall the incorrect version
            print("   🗑️ Uninstalling existing sglang package...")
            subprocess.run(
                ["pip", "uninstall", "sglang", "-y"],
                cwd=python_dir,
                check=False,  # Don't fail if package isn't installed
                capture_output=True,
                text=True
            )

            # 3. Install the current directory in editable mode
            print("   🔧 Installing sglang in editable mode...")
            result = subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=python_dir,
                check=True,
                capture_output=True,
                text=True
            )

            print("   ✅ SGLang installed successfully in editable mode")
            if result.stdout:
                print(f"   📄 Install output: {result.stdout.strip()}")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install sglang: {e}")
            if e.stdout:
                print(f"   📄 stdout: {e.stdout}")
            if e.stderr:
                print(f"   📄 stderr: {e.stderr}")
            raise
    else:
        print(f"   ⚠️ Warning: Python directory not found at {python_dir}")

    return {
        "repo_path": repo_path,
        "context": {
            **state.get("context", {}),
            # Don't store WorktreeManager object in state - it's not serializable
            # Store only the paths needed for cleanup
            "worktree_base_repo": os.environ.get("SGLANG_DIR", ""),
            "worktree_worktrees_dir": os.environ.get("WORKTREES_DIR", ""),
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
