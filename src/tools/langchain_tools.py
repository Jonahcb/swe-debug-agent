"""LangChain tool wrappers for SGLang LoRA MoE debugging."""

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from src.tools.github import GitHubClient
from src.tools.ruff import format_file, lint_file

from config.settings import settings

# =============================================================================
# Shell / Command Execution Tools
# =============================================================================


def run_command(command: str, cwd: str | None = None, timeout: int = 300) -> str:
    """Execute a shell command and return output.

    SECURITY: Agents cannot push to git repositories or modify remote operations.

    Args:
        command: The shell command to run
        cwd: Working directory (optional, defaults to worktree path)
        timeout: Timeout in seconds (default: 300 = 5 minutes)

    Returns:
        Combined stdout and stderr output, plus exit code
    """
    print(f"⚡ [TOOL] run_command: {command}")
    # SECURITY: Block dangerous git commands that could affect remote repositories
    blocked_commands = [
        "git push",
        "git fetch",
        "git pull",
        "git clone",
        "git remote",
    ]

    command_lower = command.lower().strip()
    for blocked in blocked_commands:
        if blocked in command_lower:
            return f"SECURITY BLOCKED: Agents are not allowed to execute '{blocked}' commands. This prevents accidental or malicious remote repository modifications. Only local git operations are permitted."

    # Use worktree path for command execution
    cwd = os.getenv("WORKTREE_REPO_PATH")

    print(f"\n{'=' * 60}")
    print(f"EXECUTING COMMAND: {command}")
    if cwd:
        print(f"WORKING DIRECTORY: {cwd}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"\nExit code: {result.returncode}"

        print("COMMAND RESULTS:")
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"{'=' * 60}\n")

        return output
    except subprocess.TimeoutExpired:
        error_msg = f"Error: Command timed out after {timeout} seconds"
        print(f"COMMAND ERROR: {error_msg}")
        print(f"{'=' * 60}\n")
        return error_msg
    except Exception as e:
        error_msg = f"Error executing command: {e}"
        print(f"COMMAND ERROR: {error_msg}")
        print(f"{'=' * 60}\n")
        return error_msg


def run_pytest(test_path: str, verbose: bool = True, timeout: int = 600) -> str:
    """Run pytest on a specific test file or test case.

    Args:
        test_path: Path to test file or specific test (module::class::method)
        verbose: Whether to run with -v flag (default: True)
        timeout: Timeout in seconds (default: 600 = 10 minutes)

    Returns:
        Test output with results
    """
    print(f"Running pytest: {test_path}")

    # Check if we should use mock mode (return fake output instead of running real pytest)
    if settings.pytest_mock_mode:
        print("Using mock pytest mode - returning fake test output")
        # Return fake test results instead of actually running pytest
        fake_output = """----------------------------------------
Prompt 1
----------------------------------------
LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

Prefill logprobs:
  Shape:           [4, 5]
  Max difference:  6.472468e-02
  Mean difference: 1.176369e-02
  Status:          PASS (threshold: 1e-01)

Decode logprobs:
  Shape:           [32, 5]
  Max difference:  7.100700e+00
  Mean difference: 1.039451e+00
  Status:          FAIL (threshold: 1e-01)

Output strings:
  Status:      DIFFER
  SGLang:       leading provider of software solutions for the global energy industry. SGL's software solutions are used by more than 1,000 companies in 10
  HuggingFace:  leading provider of software solutions for the global steel industry. SGL is a global leader in the development and implementation of software solutions for the steel industry. SGL

----------------------------------------
Prompt 2
----------------------------------------
LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

Prefill logprobs:
  Shape:           [9, 5]
  Max difference:  1.736760e-01
  Mean difference: 2.966831e-02
  Status:          FAIL (threshold: 1e-01)

Decode logprobs:
  Shape:           [32, 5]
  Max difference:  2.816558e-01
  Mean difference: 3.807396e-02
  Status:          FAIL (threshold: 1e-01)

Output strings:
  Status:      MATCH
  SGLang:       creating machines that can perform tasks that would typically require human intelligence. AI has the potential to revolutionize the way we live and work, and it is already being
  HuggingFace:  creating machines that can perform tasks that would typically require human intelligence. AI has the potential to revolutionize the way we live and work, and it is already being

----------------------------------------
Prompt 3
----------------------------------------
LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

Prefill logprobs:
  Shape:           [6, 5]
  Max difference:  1.380231e-01
  Mean difference: 2.518422e-02
  Status:          FAIL (threshold: 1e-01)

Decode logprobs:
  Shape:           [32, 5]
  Max difference:  1.874228e-01
  Mean difference: 3.678390e-02
  Status:          FAIL (threshold: 1e-01)

Output strings:
  Status:      MATCH
  SGLang:       computers and computing. It includes the theory of computation, algorithms, programming languages, software, hardware, and the interactions between all of these. Computer science is the
  HuggingFace:  computers and computing. It includes the theory of computation, algorithms, programming languages, software, hardware, and the interactions between all of these. Computer science is the

----------------------------------------
Prompt 4
----------------------------------------
LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

Prefill logprobs:
  Shape:           [5, 5]
  Max difference:  4.049866e-01
  Mean difference: 5.439247e-02
  Status:          FAIL (threshold: 1e-01)

Decode logprobs:
  Shape:           [32, 5]
  Max difference:  4.049864e-01
  Mean difference: 3.622374e-02
  Status:          FAIL (threshold: 1e-01)

Output strings:
  Status:      MATCH
  SGLang:       Once upon a time, in a small village nestled in the heart of a dense forest, there lived a young girl named Lily. She was known throughout the village
  HuggingFace:  Once upon a time, in a small village nestled in the heart of a dense forest, there lived a young girl named Lily. She was known throughout the village

----------------------------------------
Prompt 5
----------------------------------------
LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

Prefill logprobs:
  Shape:           [9, 5]
  Max difference:  3.239720e-01
  Mean difference: 2.978070e-02
  Status:          FAIL (threshold: 1e-01)

Decode logprobs:
  Shape:           [32, 5]
  Max difference:  1.144438e+01
  Mean difference: 2.303469e+00
  Status:          FAIL (threshold: 1e-01)

Output strings:
  Status:      DIFFER
  SGLang:       Please provide a detailed explanation of each component and its function. Additionally, can you explain how these components work together to perform tasks? Please provide a visual representation of
  HuggingFace:  A computer consists of several main components, including:

1. Central Processing Unit (CPU): This is the brain of the computer and performs all the calculations and logical

================================================================================
Overall Statistics
================================================================================

Logprob Differences:
  Prefill:
    Max of max:   4.049866e-01
    Mean of max:  2.210765e-01
    Mean of mean: 3.015788e-02
  Decode:
    Max of max:   1.144438e+01
    Mean of max:  3.883830e+00
    Mean of mean: 6.908004e-01

Logprob Statistics (threshold: 1e-01):
  Overall logprob: 0/5 FAILED
  Prefill logprob: 1/5
  Decode logprob:  0/5

String Statistics:
  Output strings:  3/5"""

        return fake_output
    else:
        # Run actual pytest command
        cmd_parts = ["python", "-m", "pytest"]
        if verbose:
            cmd_parts.append("-v")
        cmd_parts.append(test_path)

        cmd = " ".join(cmd_parts)
        print(f"Executing: {cmd}")

        try:
            result = run_command(cmd, timeout=timeout)
            return result
        except Exception as e:
            error_msg = f"Failed to run pytest: {e}"
            print(f"COMMAND ERROR: {error_msg}")
            return error_msg


# =============================================================================
# File System Tools
# =============================================================================


def file_exists(path: str) -> str:
    """Check if a file or directory exists.

    Args:
        path: Path to check (relative to worktree)
    """
    print(f"❓ [TOOL] file_exists: {path}")
    try:
        worktree_path = os.getenv("WORKTREE_REPO_PATH")
        full_path = os.path.join(worktree_path, path)
        exists = Path(full_path).exists()
        return f"{'Exists' if exists else 'Does not exist'}: {path}"
    except Exception as e:
        return f"Error checking file existence: {e}"


# =============================================================================
# Git Tools - Controlled access to specific git operations
# =============================================================================


def git_status() -> str:
    """Get the current git status of the repository.

    Returns:
        Git status output showing staged, unstaged, and untracked files
    """
    print("📊 [TOOL] git_status")
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            status_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if not status_lines or status_lines == [""]:
                return "Git status: clean (no changes)"

            formatted_output = "Git status changes:\n"
            for line in status_lines:
                if line.strip():
                    status_code = line[:2]
                    filename = line[3:]
                    formatted_output += f"  {status_code} {filename}\n"
            return formatted_output
        else:
            return f"Error getting git status: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git status timed out"
    except Exception as e:
        return f"Error executing git status: {e}"


def git_diff(all_files: bool = False) -> str:
    """Get git diff of unstaged changes.

    Args:
        all_files: If True, show diff for all files including staged changes

    Returns:
        Git diff output showing changes
    """
    print(f"🔄 [TOOL] git_diff: all_files={all_files}")
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        cmd = ["git", "diff"]
        if all_files:
            cmd.append("--cached")  # Show staged changes too

        result = subprocess.run(cmd, cwd=worktree_path, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            if result.stdout.strip():
                return f"Git diff:\n{result.stdout}"
            else:
                return "No changes to show in git diff"
        else:
            return f"Error getting git diff: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git diff timed out"
    except Exception as e:
        return f"Error executing git diff: {e}"


def git_log(max_count: int = 10) -> str:
    """Get recent git commit history.

    Args:
        max_count: Maximum number of commits to show (default: 10, max: 50)

    Returns:
        Formatted git log output
    """
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        # Limit max_count for safety
        max_count = min(max(max_count, 1), 50)

        result = subprocess.run(
            ["git", "log", "--oneline", f"-{max_count}"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            if result.stdout.strip():
                return f"Recent git commits:\n{result.stdout}"
            else:
                return "No commits found in git log"
        else:
            return f"Error getting git log: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git log timed out"
    except Exception as e:
        return f"Error executing git log: {e}"


def git_add(file_path: str) -> str:
    """Stage a file for commit.

    Args:
        file_path: Path to the file to stage (relative to worktree root)

    Returns:
        Success message or error
    """
    print(f"➕ [TOOL] git_add: {file_path}")
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        # Validate file path is within worktree
        full_path = os.path.join(worktree_path, file_path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(worktree_path)):
            return f"Error: File path {file_path} is outside the worktree"

        # Check if file exists
        if not os.path.exists(full_path):
            return f"Error: File {file_path} does not exist"

        result = subprocess.run(
            ["git", "add", file_path], cwd=worktree_path, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            return f"Successfully staged {file_path} for commit"
        else:
            return f"Error staging file: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git add timed out"
    except Exception as e:
        return f"Error executing git add: {e}"


def git_commit(message: str) -> str:
    """Create a commit with staged changes.

    Args:
        message: Commit message (must be non-empty)

    Returns:
        Success message with commit hash or error
    """
    print(f"💾 [TOOL] git_commit: {message[:50]}{'...' if len(message) > 50 else ''}")
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        # Validate commit message
        if not message or not message.strip():
            return "Error: Commit message cannot be empty"

        message = message.strip()
        if len(message) > 200:  # Reasonable limit
            return "Error: Commit message too long (max 200 characters)"

        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            # Extract commit hash from output
            output_lines = result.stdout.strip().split("\n")
            commit_hash = ""
            for line in output_lines:
                if line.startswith("[") and " " in line:
                    commit_hash = line.split()[1].strip("]")
                    break
            return f"Successfully committed changes: {commit_hash}"
        else:
            return f"Error committing changes: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git commit timed out"
    except Exception as e:
        return f"Error executing git commit: {e}"


def git_checkout(branch: str) -> str:
    """Checkout a git branch or file.

    Args:
        branch: Branch name to checkout (cannot contain dangerous flags)

    Returns:
        Success message or error
    """
    print(f"🔀 [TOOL] git_checkout: {branch}")
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        # Security: Block dangerous checkout operations
        dangerous_flags = ["-f", "--force", "--hard", "--merge", "--conflict"]
        for flag in dangerous_flags:
            if flag in branch:
                return f"Error: Dangerous checkout flag '{flag}' not allowed"

        # Basic validation - branch name should not contain spaces or special chars that indicate file paths
        if any(char in branch for char in ["/", "\\", ".."]):
            return f"Error: Invalid branch name '{branch}' - contains path separators"

        result = subprocess.run(
            ["git", "checkout", branch],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return f"Successfully checked out {branch}"
        else:
            return f"Error checking out branch: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git checkout timed out"
    except Exception as e:
        return f"Error executing git checkout: {e}"


def git_rm(file_path: str, cached: bool = False) -> str:
    """Remove a file from git tracking.

    Args:
        file_path: Path to the file to remove (relative to worktree root)
        cached: If True, only remove from index (--cached flag)

    Returns:
        Success message or error
    """
    print(f"🗑️ [TOOL] git_rm: {file_path} (cached={cached})")
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        # Validate file path is within worktree
        full_path = os.path.join(worktree_path, file_path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(worktree_path)):
            return f"Error: File path {file_path} is outside the worktree"

        cmd = ["git", "rm"]
        if cached:
            cmd.append("--cached")
        cmd.append(file_path)

        result = subprocess.run(cmd, cwd=worktree_path, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            action = "removed from index" if cached else "removed"
            return f"Successfully {action}: {file_path}"
        else:
            return f"Error removing file: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git rm timed out"
    except Exception as e:
        return f"Error executing git rm: {e}"


def git_ls_files() -> str:
    """List all files tracked by git.

    Returns:
        List of all files tracked by git
    """
    try:
        worktree_path = os.environ.get("WORKTREE_REPO_PATH")
        if not worktree_path:
            return "Error: No worktree path configured"

        result = subprocess.run(
            ["git", "ls-files"], cwd=worktree_path, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            files = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if files:
                return f"Git tracked files ({len(files)}):\n" + "\n".join(f"  {f}" for f in files)
            else:
                return "No files tracked by git"
        else:
            return f"Error listing git files: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: git ls-files timed out"
    except Exception as e:
        return f"Error executing git ls-files: {e}"


# =============================================================================
# GitHub Tools
# =============================================================================
# =============================================================================
# GitHub Tools
# =============================================================================


@tool
def github_get_pr(owner: str, repo: str, pr_number: int) -> str:
    """Get pull request information from GitHub.

    Args:
        owner: Repository owner (e.g., 'sgl-project')
        repo: Repository name (e.g., 'sglang')
        pr_number: Pull request number
    """
    try:
        client = GitHubClient()
        pr = client.get_pr(owner, repo, pr_number)
        return f"PR #{pr.number}: {pr.title}\nState: {pr.state}\nURL: {pr.url}\n\n{pr.body}"
    except Exception as e:
        return f"Error: {e}"


@tool
def github_get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """Get the diff/changes for a pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
    """
    try:
        client = GitHubClient()
        return client.get_pr_diff(owner, repo, pr_number)
    except Exception as e:
        return f"Error: {e}"


@tool
def github_get_issue(owner: str, repo: str, issue_number: int) -> str:
    """Get issue information from GitHub.

    Args:
        owner: Repository owner
        repo: Repository name
        issue_number: Issue number
    """
    try:
        client = GitHubClient()
        issue = client.get_issue(owner, repo, issue_number)
        labels = ", ".join(issue.labels) if issue.labels else "None"
        return f"Issue #{issue.number}: {issue.title}\nState: {issue.state}\nLabels: {labels}\n\n{issue.body}"
    except Exception as e:
        return f"Error: {e}"


@tool
def github_get_file(owner: str, repo: str, path: str, ref: str = "main") -> str:
    """Get content of a file from a GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        path: Path to file in repo
        ref: Branch or commit ref (default: main)
    """
    try:
        client = GitHubClient()
        return client.get_file_content(owner, repo, path, ref)
    except Exception as e:
        return f"Error: {e}"


@tool
def github_list_repo_files(owner: str, repo: str, path: str = "", ref: str = "main") -> str:
    """List files and directories in a GitHub repository path.

    Args:
        owner: Repository owner
        repo: Repository name
        path: Path to directory in repo (empty string for root)
        ref: Branch or commit ref (default: main)
    """
    try:
        client = GitHubClient()
        files = client.list_repo_files(owner, repo, path, ref)
        return "\n".join(files)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Ruff Linting Tools
# =============================================================================


def ruff_lint(path: str, fix: bool = False) -> str:
    """Run Ruff linter on a Python file or directory.

    Args:
        path: Path to file or directory to lint
        fix: Whether to auto-fix issues (default: False)
    """
    try:
        results = lint_file(path, fix=fix)
        if not results:
            return "No linting issues found."
        issues = [
            f"{r['file']}:{r['line']}:{r['column']} {r['code']} {r['message']}" for r in results
        ]
        return f"Found {len(issues)} issues:\n" + "\n".join(issues)
    except Exception as e:
        return f"Error: {e}"


def ruff_format(path: str) -> str:
    """Format Python code using Ruff formatter.

    Args:
        path: Path to file or directory to format
    """
    try:
        success = format_file(path)
        return "Formatting successful." if success else "Formatting found issues."
    except Exception as e:
        return f"Error: {e}"

    # =============================================================================


# =============================================================================
# Fix Validation Tools
# =============================================================================


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for comparison purposes.

    - Strips leading/trailing whitespace from each line
    - Removes empty lines
    - Normalizes indentation by removing common leading whitespace
    - Joins lines back together

    Args:
        text: The text to normalize

    Returns:
        Normalized text with consistent whitespace handling
    """
    if not text:
        return ""

    # Split into lines and strip leading/trailing whitespace
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines
    lines = [line for line in lines if line]

    if not lines:
        return ""

    # Find minimum indentation (ignoring empty lines)
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())

    # Remove common indentation
    if min_indent > 0:
        lines = [line[min_indent:] if len(line) > min_indent else line.lstrip() for line in lines]

    # Join back together
    return "\n".join(lines)


def _contains_normalized(needle: str, haystack: str) -> bool:
    """Check if needle exists in haystack with whitespace normalization.

    Args:
        needle: The string to search for
        haystack: The string to search in

    Returns:
        True if needle is found (with normalization), False otherwise
    """
    # First try exact match (fast path)
    if needle in haystack:
        return True

    # Normalize both strings and try again
    normalized_needle = _normalize_whitespace(needle)
    normalized_haystack = _normalize_whitespace(haystack)

    return normalized_needle in normalized_haystack


@tool
def simple_check_fixes(fixes_list: list) -> str:
    """Validate that fixes can be applied to files by checking if old_string exists in each file.

    Args:
        fixes_list: List of tuples [(file_path, old_string), ...] for all code blocks being modified

    Returns:
        Validation results indicating which fixes can be applied successfully
    """
    print(f"🔍 [TOOL] simple_check_fixes: validating {len(fixes_list)} fix tuples")

    if not fixes_list:
        return "❌ No fixes found in list"

    results = []
    all_valid = True

    for i, (file_path, old_string) in enumerate(fixes_list):
        if not file_path or not old_string:
            results.append(f"❌ Fix {i + 1}: Missing file_path or old_string")
            all_valid = False
            continue

        # Get worktree path and construct full file path
        worktree_path = os.getenv("WORKTREE_REPO_PATH")
        if not worktree_path:
            results.append(f"❌ Fix {i + 1}: No worktree path configured")
            all_valid = False
            continue

        full_path = os.path.join(worktree_path, file_path)

        # Print base path and suffix path for debugging
        print(f"🔍 [TOOL] simple_check_fixes: Base path: {worktree_path}")
        print(f"🔍 [TOOL] simple_check_fixes: Suffix path: {file_path}")
        print(f"🔍 [TOOL] simple_check_fixes: Full path: {full_path}")

        # Check if file exists
        if not os.path.exists(full_path):
            results.append(f"❌ Fix {i + 1}: File does not exist: {file_path}")
            all_valid = False
            continue

        # Read file content
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            results.append(f"❌ Fix {i + 1}: Cannot read file {file_path}: {e}")
            all_valid = False
            continue

        # Check if old_string exists in file content (with whitespace normalization)
        if _contains_normalized(old_string, file_content):
            results.append(f"✅ Fix {i + 1}: {file_path} - old_string found and can be replaced")
        else:
            results.append(
                f"❌ Fix {i + 1}: {file_path} - old_string NOT found in file (cannot apply fix)"
            )
            all_valid = False

    # Overall summary
    if all_valid:
        results.insert(
            0,
            "🎉 ALL FIXES VALID: All fixes can be successfully applied to files",
        )
    else:
        results.insert(
            0,
            "⚠️ SOME FIXES INVALID: One or more fixes cannot be applied - please revise",
        )

    return "\n".join(results)


# =============================================================================
# Tool Collections for Different Agent Roles
# =============================================================================

# Architect: Planning and coordination (no tools - pure reasoning)
ARCHITECT_TOOLS = []

# Coder: Implementing code changes from architect's plan
# Note: Filesystem tools (read_file, write_file, edit_file, list_files, glob, grep)
# are automatically provided by deepagents FilesystemMiddleware
CODER_TOOLS = [
    git_status,
    git_diff,
    git_add,
    git_commit,
    git_checkout,
    git_rm,
]

# Critic: Code review
# Note: read_file is automatically provided by deepagents FilesystemMiddleware
CRITIC_TOOLS = [
    ruff_lint,
    github_get_pr_diff,
]

# Librarian tools have been split between internal_librarian and external_librarian

# Internal Librarian: Codebase search and analysis
INTERNAL_LIBRARIAN_TOOLS = [
    git_status,
    git_diff,
    git_log,
    git_ls_files,
]

# External Librarian: Internet and external source search
EXTERNAL_LIBRARIAN_TOOLS = [
    github_get_pr,
    github_get_pr_diff,
    github_get_issue,
    github_get_file,
    github_list_repo_files,
]
