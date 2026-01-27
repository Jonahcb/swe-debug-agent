"""LangChain tool wrappers for SGLang LoRA MoE debugging."""

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from src.tools.github import GitHubClient
from src.tools.ruff import lint_file
from src.utils.linting import create_linter, CodeLinter
from src.schemas import (
    BugInfo,
    FinalBugAnalysisInput,
    FixCheckerInput,
    FixValidationResult,
    SubmitFixesInput,
)


# =============================================================================
# File System Tools
# =============================================================================


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


def _find_normalized_position(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find the start and end positions of needle in haystack using normalized matching.

    Args:
        needle: The string to search for
        haystack: The string to search in

    Returns:
        Tuple of (start, end) positions in the original haystack, or None if not found
    """
    # First try exact match (fast path)
    pos = haystack.find(needle)
    if pos != -1:
        return pos, pos + len(needle)

    # Normalize both strings
    normalized_needle = _normalize_whitespace(needle)
    normalized_haystack = _normalize_whitespace(haystack)

    # Find normalized needle in normalized haystack
    norm_pos = normalized_haystack.find(normalized_needle)
    if norm_pos == -1:
        return None

    # Map back to original positions - this is approximate
    # Split both into lines and find corresponding positions
    needle_lines = [line.strip() for line in needle.split("\n") if line.strip()]
    haystack_lines = [line for line in haystack.split("\n")]

    # Find the sequence of non-empty lines in haystack that match our needle lines
    needle_line_count = len(needle_lines)

    for i in range(len(haystack_lines) - needle_line_count + 1):
        # Check if this sequence matches
        matches = True
        for j in range(needle_line_count):
            if haystack_lines[i + j].strip() != needle_lines[j]:
                matches = False
                break

        if matches:
            # Find the actual start and end positions
            start_line_idx = i
            end_line_idx = i + needle_line_count - 1

            # Calculate character positions
            start_pos = sum(len(haystack_lines[k]) + 1 for k in range(start_line_idx))  # +1 for \n
            end_pos = (
                start_pos
                + sum(len(haystack_lines[k]) + 1 for k in range(start_line_idx, end_line_idx + 1))
                - 1
            )

            return start_pos, end_pos

    return None


def _replace_normalized(old_string: str, new_string: str, content: str) -> str:
    """Replace old_string with new_string in content using normalized matching.

    Args:
        old_string: The string to replace
        new_string: The replacement string
        content: The content to modify

    Returns:
        Modified content with replacement, or original content if replacement fails
    """
    # First try exact replacement (fast path)
    if old_string in content:
        return content.replace(old_string, new_string, 1)

    # Try normalized replacement
    positions = _find_normalized_position(old_string, content)
    if positions is None:
        return content  # No replacement possible

    start_pos, end_pos = positions
    return content[:start_pos] + new_string + content[end_pos:]


@tool(args_schema=FixCheckerInput)
def simple_check_fixes_structured(root) -> dict:
    """Validate that fixes can be applied to files using structured input/output.

    This tool uses SGLang constrained decoding for the tool call arguments.
    The args_schema=FixCheckerInput ensures the LLM generates valid FixCheckerInput.

    Args:
        root: List of FixTuple objects containing the fixes to validate.

    Returns:
        Dict with validation results matching FixCheckerOutput schema:
        {
            "all_valid": bool,
            "results": [{"fix_index": int, "file_path": str, "is_valid": bool, "message": str}, ...],
            "summary": str
        }
    """
    if not root:
        return {
            "all_valid": False,
            "results": [],
            "summary": "No fixes provided",
        }

    # Extract (file_path, old_string) tuples directly from the root list
    # The args_schema ensures we receive properly structured input from the LLM
    fixes_list = [(f.file_path, f.old_string) for f in root]

    print(f"🔍 [TOOL] simple_check_fixes_structured: validating {len(fixes_list)} fix tuples")

    if not fixes_list:
        return {
            "all_valid": False,
            "results": [],
            "summary": "No fixes found in input",
        }

    results = []
    all_valid = True

    for i, (file_path, old_string) in enumerate(fixes_list):
        if not file_path or not old_string:
            results.append(
                FixValidationResult(
                    fix_index=i,
                    file_path=file_path or "",
                    is_valid=False,
                    message="Missing file_path or old_string",
                )
            )
            all_valid = False
            continue

        # Get worktree path and construct full file path
        worktree_path = os.getenv("WORKTREE_REPO_PATH")
        if not worktree_path:
            results.append(
                FixValidationResult(
                    fix_index=i,
                    file_path=file_path,
                    is_valid=False,
                    message="No worktree path configured",
                )
            )
            all_valid = False
            continue

        # Normalize file_path to be relative (remove leading / if present)
        normalized_file_path = file_path.lstrip("/") if file_path.startswith("/") else file_path
        full_path = os.path.join(worktree_path, normalized_file_path)

        # Print debug info
        print(f"🔍 [TOOL] simple_check_fixes_structured: Checking {file_path}")

        # Check if file exists
        if not os.path.exists(full_path):
            results.append(
                FixValidationResult(
                    fix_index=i,
                    file_path=file_path,
                    is_valid=False,
                    message=f"File does not exist: {file_path}",
                )
            )
            all_valid = False
            continue

        # Read file content
        try:
            with open(full_path, encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            results.append(
                FixValidationResult(
                    fix_index=i,
                    file_path=file_path,
                    is_valid=False,
                    message=f"Cannot read file: {e}",
                )
            )
            all_valid = False
            continue

        # Check if old_string exists in file content (with whitespace normalization)
        if _contains_normalized(old_string, file_content):
            results.append(
                FixValidationResult(
                    fix_index=i,
                    file_path=file_path,
                    is_valid=True,
                    message="old_string found and can be replaced",
                )
            )
        else:
            results.append(
                FixValidationResult(
                    fix_index=i,
                    file_path=file_path,
                    is_valid=False,
                    message="old_string NOT found in file (cannot apply fix)",
                    file_contents=file_content,  # Include full file contents for debugging
                )
            )
            all_valid = False

    # Build summary
    valid_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)

    if all_valid:
        # Basic validations passed - now run linters on each candidate fix
        print(f"🔍 [TOOL] simple_check_fixes_structured: Running linters on {len(fixes_list)} fixes")

        # Get worktree path for linting
        worktree_path = os.getenv("WORKTREE_REPO_PATH")
        if not worktree_path:
            summary = f"BASIC VALIDATION FAILED: No worktree path configured for linting"
            return {
                "all_valid": False,
                "results": [r.model_dump() for r in results],
                "summary": summary,
            }

        # Create linter instance
        linter = create_linter()

        # Prepare fixes for linting (need new_string, so we need to extract from root)
        fixes_with_new_strings = []
        for i, fix_tuple in enumerate(root):
            fixes_with_new_strings.append((
                fix_tuple.file_path,
                fix_tuple.old_string,
                fix_tuple.new_string
            ))

        # Run linting on each fix independently
        lint_results = linter.validate_multiple_fixes(worktree_path, fixes_with_new_strings)

        # Check if all linting passed
        linting_passed = all(success for success, _, _ in lint_results.values())

        if linting_passed:
            summary = f"ALL FIXES VALID: {valid_count}/{total_count} fixes can be successfully applied and pass linting"
            # Set validation flag when both basic validation and linting pass
            set_simple_check_validation_passed(True)
        else:
            # Linting failed - update results with linting errors
            all_valid = False
            for fix_index, (success, errors, output) in lint_results.items():
                if not success:
                    # Update the corresponding result with linting errors
                    results[fix_index].is_valid = False
                    error_messages = []
                    for error in errors:
                        error_messages.append(f"{error.code}: {error.message} (line {error.line})")
                    linting_message = f"Linting failed: {'; '.join(error_messages)}"
                    if results[fix_index].message:
                        results[fix_index].message += f" | {linting_message}"
                    else:
                        results[fix_index].message = linting_message

            valid_count = sum(1 for r in results if r.is_valid)
            summary = f"SOME FIXES INVALID: {valid_count}/{total_count} fixes valid - linting failed for some fixes"
    else:
        summary = f"SOME FIXES INVALID: {valid_count}/{total_count} fixes valid - please revise invalid fixes"

    return {
        "all_valid": all_valid,
        "results": [r.model_dump() for r in results],
        "summary": summary,
    }


# =============================================================================
# Final Bug Analysis Tool
# =============================================================================


# Global state to trigger expand phase from tool
_expand_trigger_data = None


def trigger_expand_phase(bug_data):
    """Set global trigger for expand phase."""
    global _expand_trigger_data
    _expand_trigger_data = bug_data
    print("🎯 [TOOL] final_bug_analysis: Expand phase triggered with bug data")


def get_expand_trigger():
    """Get and clear the expand trigger data."""
    global _expand_trigger_data
    data = _expand_trigger_data
    _expand_trigger_data = None
    return data


# Global state to trigger execute phase from submit_fixes tool
_execute_trigger_data = None

# Global state to track simple_check_fixes_structured validation status
_simple_check_validation_passed = False


def trigger_execute_phase(fixes_data):
    """Set global trigger for execute phase."""
    global _execute_trigger_data
    _execute_trigger_data = fixes_data
    print("🎯 [TOOL] submit_fixes: Execute phase triggered with fixes data")


def get_execute_trigger():
    """Get and clear the execute trigger data."""
    global _execute_trigger_data
    data = _execute_trigger_data
    _execute_trigger_data = None
    return data


def set_simple_check_validation_passed(passed: bool):
    """Set the simple_check_fixes_structured validation status."""
    global _simple_check_validation_passed
    _simple_check_validation_passed = passed
    print(f"🔍 [TOOL] simple_check_fixes_structured validation status set to: {passed}")


def get_simple_check_validation_passed() -> bool:
    """Get the current simple_check_fixes_structured validation status."""
    global _simple_check_validation_passed
    return _simple_check_validation_passed


def reset_simple_check_validation():
    """Reset the simple_check_fixes_structured validation status for new expand phase."""
    global _simple_check_validation_passed
    _simple_check_validation_passed = False
    print("🔄 [TOOL] simple_check_fixes_structured validation status reset for new expand phase")


@tool(args_schema=FinalBugAnalysisInput)
def final_bug_analysis(bug_analysis: FinalBugAnalysisInput = None, root=None) -> str:
    """Provide the final bug analysis to transition to the expand/coder phase.

    This tool uses SGLang's strict tool call constrained decoding with the
    FinalBugAnalysisInput schema to ensure the architect provides bug analysis
    in the exact structured format expected by the coder agent.

    Args:
        bug_analysis: Dictionary containing bug analysis in the format:
            {
                "root": {
                    "bug_1": {
                        "relevant_files_and_lines": "file.py:123-125, file2.py:456",
                        "description": "Technical low-level one sentence bug description and potential fixes"
                    },
                    "bug_2": {
                        "relevant_files_and_lines": "file.py:789, file3.py:101",
                        "description": "Technical low-level one sentence bug description and potential fixes"
                    },
                    ...
                }
            }

    Returns:
        Confirmation message that the bug analysis has been accepted and will be passed to the coder
    """
    print("🎯 [TOOL] final_bug_analysis: Architect providing final bug analysis")

    # Handle the case where LangChain passes root as a keyword argument
    if root is not None:
        bug_analysis = FinalBugAnalysisInput(root=root)

    if not bug_analysis or not bug_analysis.root:
        return "❌ Error: No bug analysis provided"

    # Get the actual bug dict from the RootModel
    bugs_dict = bug_analysis.root

    # Check that all keys start with "bug_"
    for key in bugs_dict.keys():
        if not key.startswith("bug_"):
            return f"❌ Error: All bug keys must start with 'bug_', found: {key}"

    # All validation is handled by Pydantic schema, so if we get here, the data is valid
    print(f"✅ Bug analysis validation passed: {len(bugs_dict)} bugs identified")

    # Validate and resolve file paths using PathResolver
    worktree_path = os.environ.get("WORKTREE_REPO_PATH")
    if not worktree_path:
        return "❌ Error: WORKTREE_REPO_PATH not set. Cannot validate file paths."

    from src.utils.path_resolver import PathResolver
    resolver = PathResolver(worktree_path)

    unresolved_files = []
    resolved_paths = {}

    for bug_key, bug_info in bugs_dict.items():
        # Extract file paths from relevant_files_and_lines
        # Format: "file.py:123-125, file2.py:456" -> ["file.py", "file2.py"]
        relevant_files_str = bug_info.relevant_files_and_lines
        if relevant_files_str:
            # Split by comma and extract file paths (before colon)
            file_entries = [entry.strip() for entry in relevant_files_str.split(",")]
            for file_entry in file_entries:
                # Extract file path (everything before the first colon, if present)
                if ":" in file_entry:
                    extracted_file_path = file_entry.split(":")[0].strip()
                else:
                    extracted_file_path = file_entry.strip()

                if extracted_file_path:  # Only check non-empty file paths
                    # Try to resolve the path using PathResolver
                    result = resolver.resolve_path(extracted_file_path)

                    if result['resolved_path']:
                        # Path resolved successfully - use the resolved path
                        resolved_paths[extracted_file_path] = result['resolved_path']
                        print(f"✅ Path resolved: '{extracted_file_path}' -> '{result['resolved_path']}'")
                    elif result['candidates']:
                        # Multiple candidates - use the first (best) candidate
                        best_candidate = result['candidates'][0]
                        resolved_paths[extracted_file_path] = best_candidate
                        print(f"⚠️  Path partially resolved: '{extracted_file_path}' -> '{best_candidate}' ({result['strategy']})")
                    else:
                        # No resolution possible
                        unresolved_files.append(extracted_file_path)
                        print(f"❌ Path unresolvable: '{extracted_file_path}' ({result['strategy']})")

    # Check if resolved paths actually exist
    missing_files = []
    for original_path, resolved_path in resolved_paths.items():
        full_path = Path(worktree_path) / resolved_path
        if not full_path.exists():
            missing_files.append(resolved_path)

    if unresolved_files or missing_files:
        error_msg = "❌ FILE PATH VALIDATION FAILED:\n"
        if unresolved_files:
            error_msg += f"Unresolvable file paths: {', '.join(unresolved_files)}\n"
        if missing_files:
            error_msg += f"Resolved paths that don't exist: {', '.join(missing_files)}\n"
        error_msg += "\nPlease revise your bug analysis to reference valid file paths."
        return error_msg

    # Update bugs_dict with resolved file paths before proceeding
    updated_bugs_dict = {}
    for bug_key, bug_info in bugs_dict.items():
        relevant_files_str = bug_info.relevant_files_and_lines
        if relevant_files_str:
            # Replace original paths with resolved paths in the relevant_files_and_lines string
            updated_files_str = relevant_files_str
            file_entries = [entry.strip() for entry in relevant_files_str.split(",")]
            for file_entry in file_entries:
                # Extract file path (everything before the first colon, if present)
                if ":" in file_entry:
                    extracted_file_path = file_entry.split(":")[0].strip()
                    line_info = file_entry.split(":", 1)[1]  # Everything after first colon
                else:
                    extracted_file_path = file_entry.strip()
                    line_info = ""

                if extracted_file_path in resolved_paths:
                    resolved_path = resolved_paths[extracted_file_path]
                    # Replace the original path with resolved path in the string
                    if line_info:
                        updated_files_str = updated_files_str.replace(f"{extracted_file_path}:{line_info}", f"{resolved_path}:{line_info}")
                    else:
                        updated_files_str = updated_files_str.replace(extracted_file_path, resolved_path)

            # Create updated bug info with resolved paths
            updated_bugs_dict[bug_key] = BugInfo(
                relevant_files_and_lines=updated_files_str,
                description=bug_info.description
            )
        else:
            updated_bugs_dict[bug_key] = bug_info

    # Format validation passed
    bug_count = len(updated_bugs_dict)
    print(f"✅ Final bug analysis accepted: {bug_count} bugs identified")

    # TRIGGER EXPAND PHASE DIRECTLY FROM TOOL
    trigger_expand_phase(updated_bugs_dict)

    # Return formatted confirmation - this will be used by the LATS agent
    formatted_analysis = "\n".join(
        [
            f'{{{key}: {{relevant_files_and_lines: "{info.relevant_files_and_lines}", description: "{info.description}"}}}}'
            for key, info in updated_bugs_dict.items()
        ]
    )

    return f"🎯 FINAL BUG ANALYSIS ACCEPTED - Transitioning to expand/coder phase\n\n{formatted_analysis}"


@tool(args_schema=SubmitFixesInput)
def submit_fixes(fixes: SubmitFixesInput = None, root=None) -> str:
    """Submit candidate fixes to transition to execute/critic phase.

    This tool uses SGLang's strict tool call constrained decoding with the
    SubmitFixesInput schema to ensure the coder agent provides candidate fixes
    in the exact structured format expected by the execute/critic phase.

    Args:
        fixes: Dictionary containing candidate fixes in the format:
            {
                "root": [
                    {
                        "description": "Brief description of this candidate fix",
                        "modified_files": [
                            {
                                "file_path": "path/to/file.py",
                                "old_string": "existing code block to replace",
                                "new_string": "new code to replace the old_string with"
                            }
                        ]
                    }
                ]
            }

    Returns:
        Confirmation message that the fixes have been accepted and will be passed to execute/critic
    """
    print("🎯 [TOOL] submit_fixes: Coder providing final candidate fixes")

    # Check if simple_check_fixes_structured validation has passed
    if not get_simple_check_validation_passed():
        return "ERROR: submit_fixes tool unavailable until `simple_check_fixes_structured` tool returns all old_strings are valid and all fixes pass the linting checks."

    try:
        # Handle the case where LangChain passes root as a keyword argument
        if root is not None:
            fixes = SubmitFixesInput(root=root)

        if not fixes or not fixes.root:
            return "❌ Error: No fixes provided"

        # Get the actual fixes list from the RootModel
        fixes_list = fixes.root

        # Validate each candidate fix has valid modifications
        valid_candidates = []
        for i, candidate in enumerate(fixes_list):
            valid_modifications = []
            invalid_count = 0

            for j, mod_file in enumerate(candidate.modified_files):
                try:
                    # The Pydantic validation will happen here
                    # If validation passes, add to valid modifications
                    valid_modifications.append(mod_file)
                    print(f"   ✅ Candidate {i + 1}, File {j + 1}: Valid modification")
                except ValueError as e:
                    print(f"   ❌ Candidate {i + 1}, File {j + 1}: {e}")
                    invalid_count += 1

            # Only include candidates with at least one valid modification
            if valid_modifications:
                # Create a new candidate with only valid modifications
                valid_candidate = candidate.model_copy()
                valid_candidate.modified_files = valid_modifications
                valid_candidates.append(valid_candidate)
                print(f"   ✅ Candidate {i + 1}: {len(valid_modifications)} valid modifications")
            else:
                print(f"   ❌ Candidate {i + 1}: No valid modifications, skipping")

        if not valid_candidates:
            return "❌ Error: No valid candidate fixes provided. All candidates had invalid or missing file modifications."

        print(f"✅ Fixes validation passed: {len(valid_candidates)} valid candidate fixes submitted")

        # TRIGGER EXECUTE PHASE DIRECTLY FROM TOOL
        trigger_execute_phase(valid_candidates)

        # Return formatted confirmation - this will be used by the LATS agent
        formatted_fixes = "\n".join(
            [
                f"Fix {i + 1}: {fix.description} ({len(fix.modified_files)} files)"
                for i, fix in enumerate(valid_candidates)
            ]
        )

        return (
            f"🎯 CANDIDATE FIXES ACCEPTED - Transitioning to execute/critic phase\n\n{formatted_fixes}"
        )

    except Exception as e:
        error_msg = f"❌ Error validating fixes: {str(e)}"
        print(error_msg)
        return error_msg


# =============================================================================
# Tool Collections for Different Agent Roles
# =============================================================================

# Architect: Planning and coordination (final bug analysis tool)
ARCHITECT_TOOLS = [final_bug_analysis]

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
    simple_check_fixes_structured,
    submit_fixes,
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
