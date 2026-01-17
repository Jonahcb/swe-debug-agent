"""Git worktree management for isolated agent environments."""

import os
import subprocess
import uuid
import shutil


class WorktreeManager:
    """Manages git worktrees for isolated agent execution environments."""

    def __init__(self, base_repo_path, worktrees_dir):
        """Initialize the worktree manager.

        Args:
            base_repo_path: Path to the base git repository (.git folder)
            worktrees_dir: Directory to store worktree folders
        """
        self.base_repo = os.path.abspath(base_repo_path)
        self.worktrees_dir = os.path.abspath(worktrees_dir)
        os.makedirs(self.worktrees_dir, exist_ok=True)

    def create_environment(self, base_branch="main"):
        """Create a new isolated worktree environment.

        Args:
            base_branch: Branch to base the new worktree on

        Returns:
            Absolute path to the new worktree directory
        """
        # Generate unique session ID
        session_id = f"agent-{uuid.uuid4().hex[:8]}"
        new_branch = f"feat/{session_id}"
        path = os.path.join(self.worktrees_dir, session_id)

        print(f"🔧 Creating new worktree environment: {session_id} (branch: {new_branch}, path: {path})")

        # Create worktree with new branch
        # MUST run inside the actual repo path (self.base_repo)
        subprocess.run(
            ["git", "worktree", "add", "-b", new_branch, path, base_branch],
            cwd=self.base_repo,
            check=True
        )

        print(f"✅ Worktree created successfully at: {path}")

        return path

    def cleanup_environment(self, path):
        """Clean up a worktree environment.

        Args:
            path: Path to the worktree directory to clean up
        """
        if not os.path.exists(path):
            return

        # Remove worktree (command runs in base repo)
        subprocess.run(
            ["git", "worktree", "remove", path, "--force"],
            cwd=self.base_repo,
            check=False
        )

        # Delete branch (command runs in base repo)
        branch = f"feat/{os.path.basename(path)}"
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=self.base_repo,
            check=False
        )