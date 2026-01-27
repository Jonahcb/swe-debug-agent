"""Linting utilities for validating code changes using sglang pre-commit."""
import os
import subprocess
from typing import Optional

from dataclasses import dataclass


@dataclass
class LintError:
    """Represents a linting error."""
    file_path: str
    line: int
    column: int
    code: str
    message: str
    severity: str = "error"


@dataclass
class LintResult:
    """Result of running linters on code."""
    success: bool
    errors: list[LintError]
    output: str


class CodeLinter:
    """Utility class for linting Python code using sglang pre-commit hooks."""

    def __init__(self, linter_config: Optional[dict[str, bool]] = None):
        """Initialize the linter.

        Args:
            linter_config: Dictionary mapping linter names to enable/disable flags.
                          Each linter name maps to actual pre-commit hooks:
                          - 'ast' -> python-check-ast hook
                          - 'ruff' -> ruff, ruff-format hooks
                          - 'black' -> black hook
                          - 'mypy' -> mypy hook
                          - 'flake8' -> flake8 hook
                          - 'isort' -> isort hook
                          If None, defaults to only 'ast' and 'ruff' enabled.
        """
        if linter_config is None:
            # Default: only AST and ruff
            self.linter_config = {
                'ast': True,
                'ruff': True,
                'black': False,
                'mypy': False,
                'flake8': False,
                'isort': False
            }
        else:
            self.linter_config = linter_config



    def enable_linter(self, linter_name: str, enabled: bool = True):
        """Enable or disable a specific linter.

        Args:
            linter_name: Name of the linter ('ast', 'ruff', 'black', 'mypy', 'flake8', 'isort')
            enabled: Whether to enable the linter
        """
        if linter_name in self.linter_config:
            self.linter_config[linter_name] = enabled

    def get_enabled_linters(self) -> list[str]:
        """Get list of currently enabled linters."""
        return [name for name, enabled in self.linter_config.items() if enabled]

    def run_sglang_pre_commit(self, repo_path: str, file_path: Optional[str] = None, modified_files: Optional[set[str]] = None) -> LintResult:
        """Run sglang pre-commit checks and pylint on modified files.

        Args:
            repo_path: Path to the repository root
            file_path: Optional specific file to check (relative to repo_path)
            modified_files: Optional set of modified file paths. If None, will detect automatically.

        Returns:
            LintResult with pre-commit and pylint check results
        """
        try:
            # Get the hook IDs based on current configuration
            enabled_linters = [name for name, enabled in self.linter_config.items() if enabled]
            if not enabled_linters:
                # No linters enabled
                return LintResult(success=True, errors=[], output="No linters enabled")

            # Map linter names to pre-commit hook IDs
            hook_mapping = {
                'ast': ['check-ast'],
                'ruff': ['ruff'],
                'black': ['black'],
                'mypy': ['mypy'],
                'flake8': ['flake8'],
                'isort': ['isort']
            }

            # Collect all hook IDs for enabled linters
            hook_ids = []
            for linter in enabled_linters:
                if linter in hook_mapping:
                    hook_ids.extend(hook_mapping[linter])

            if not hook_ids:
                return LintResult(success=True, errors=[], output="No matching hooks found")

            # Log the pre-commit execution details
            print(f"🔍 [LINTING] Running pre-commit on repository path: {repo_path}")
            commands = []
            for hook_id in hook_ids:
                commands.append(f"pre-commit run {hook_id} --all-files")
            print(f"🔍 [LINTING] Commands: {'; '.join(commands)}")

            # Run each command separately and collect results
            all_success = True
            combined_output = ""
            all_errors = []

            for hook_id in hook_ids:
                cmd = ["pre-commit", "run", hook_id, "--all-files"]
                result = subprocess.run(
                    cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120  # Longer timeout for pre-commit as it may run multiple tools
                )

                combined_output += f"\n--- {hook_id} ---\n"
                combined_output += result.stdout
                if result.stderr:
                    combined_output += "\nSTDERR:\n" + result.stderr

                if result.returncode != 0:
                    all_success = False
                    all_errors.append(LintError(
                        file_path=file_path or repo_path,
                        line=1,
                        column=1,
                        code=f"PRECOMMIT_{hook_id.upper()}",
                        message=f"Pre-commit {hook_id} failed:\n{result.stdout + result.stderr}",
                        severity="error"
                    ))

            # Run pylint on modified Python files
            pylint_result = self.run_pylint_on_modified_files(repo_path, modified_files)
            if not pylint_result.success:
                all_success = False
                all_errors.extend(pylint_result.errors)
                combined_output += "\n" + pylint_result.output

            return LintResult(success=all_success, errors=all_errors, output=combined_output.strip())

        except subprocess.TimeoutExpired:
            return LintResult(
                success=False,
                errors=[LintError(
                    file_path=file_path or repo_path,
                    line=1,
                    column=1,
                    code="TIMEOUT",
                    message="Pre-commit checks timed out",
                    severity="error"
                )],
                output="Pre-commit checks timed out after 120 seconds"
            )
        except Exception as e:
            return LintResult(
                success=False,
                errors=[LintError(
                    file_path=file_path or repo_path,
                    line=1,
                    column=1,
                    code="PRECOMMIT004",
                    message=f"Pre-commit check failed: {str(e)}",
                    severity="error"
                )],
                output=str(e)
            )


    def validate_fix_with_linters(
        self,
        repo_path: str,
        file_path: str,
        old_string: str,
        new_string: str,
        modified_files: Optional[set[str]] = None
    ) -> tuple[bool, list[LintError], str]:
        """Validate a single fix by applying it and running sglang pre-commit checks and pylint.

        Args:
            repo_path: Path to the repository root
            file_path: Path to the file to modify (relative to repo_path)
            old_string: The old string to replace
            new_string: The new string to replace it with

        Returns:
            Tuple of (success, errors, combined_output)
        """
        full_file_path = os.path.join(repo_path, file_path)

        # Read the original file
        try:
            with open(full_file_path, encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return False, [LintError(
                file_path=file_path,
                line=1,
                column=1,
                code="FILE001",
                message=f"Cannot read file: {str(e)}",
                severity="error"
            )], f"Cannot read file: {str(e)}"

        # Apply the fix
        if old_string not in original_content:
            return False, [LintError(
                file_path=file_path,
                line=1,
                column=1,
                code="FIX001",
                message="old_string not found in file content",
                severity="error"
            )], "old_string not found in file content"

        modified_content = original_content.replace(old_string, new_string, 1)

        # Write the modified content back to the file temporarily
        try:
            with open(full_file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
        except Exception as e:
            return False, [LintError(
                file_path=file_path,
                line=1,
                column=1,
                code="FILE002",
                message=f"Cannot write modified file: {str(e)}",
                severity="error"
            )], f"Cannot write modified file: {str(e)}"

        try:
            # Run sglang pre-commit checks and pylint on the entire repo
            pre_commit_result = self.run_sglang_pre_commit(repo_path, file_path, modified_files)

            success = pre_commit_result.success
            all_errors = pre_commit_result.errors
            combined_output = f"SGLang Pre-commit and Pylint Check: {pre_commit_result.output}"

            return success, all_errors, combined_output

        finally:
            # Always restore the original file content, even if linting fails
            try:
                with open(full_file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            except Exception as restore_error:
                # If we can't restore, that's a problem but don't mask the original linting result
                print(f"Warning: Could not restore original file content: {restore_error}")
                pass

    def validate_multiple_fixes(
        self,
        repo_path: str,
        fixes: list[tuple[str, str, str]]
    ) -> dict[int, tuple[bool, list[LintError], str]]:
        """Validate multiple fixes by running linters on each.

        Args:
            fixes: List of (file_path, old_string, new_string) tuples

        Returns:
            Dict mapping fix index to (success, errors, output)
        """
        # Extract modified Python files from the fixes
        modified_files = set()
        for file_path, _, _ in fixes:
            if file_path.endswith('.py'):
                modified_files.add(file_path)

        results = {}

        for i, (file_path, old_string, new_string) in enumerate(fixes):
            success, errors, output = self.validate_fix_with_linters(
                repo_path, file_path, old_string, new_string, modified_files
            )
            results[i] = (success, errors, output)

        return results

    def get_modified_files(self, repo_path: str) -> set[str]:
        """Get set of modified Python files in the repository.

        Args:
            repo_path: Path to the repository root

        Returns:
            Set of modified Python file paths (relative to repo_path)
        """
        try:
            # Get git status to find modified files
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"Warning: Could not get git status: {result.stderr}")
                return set()

            modified_files = set()
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Git status --porcelain format: XY filename
                    # X = index status, Y = working tree status
                    # Filename starts after the status codes (usually at position 2 or 3)
                    # Find the filename by skipping leading status characters
                    filename_start = 2
                    while filename_start < len(line) and line[filename_start] == ' ':
                        filename_start += 1
                    filename = line[filename_start:].strip()

                    # Include modified, added, and renamed files (not deleted or untracked)
                    # Check both index and working tree status
                    index_status = line[0] if len(line) > 0 else ' '
                    working_status = line[1] if len(line) > 1 else ' '

                    # Include if either status shows modification/addition/rename (not deleted, unmodified, or untracked)
                    if ((index_status in ['M', 'A', 'R', 'C', 'U']) or
                        (working_status in ['M', 'A', 'R', 'C', 'U'])):
                        # Check if it's a Python file
                        if filename.endswith('.py'):
                            modified_files.add(filename)

            return modified_files

        except subprocess.TimeoutExpired:
            print("Warning: Git status timed out")
            return set()
        except Exception as e:
            print(f"Warning: Could not get modified files: {e}")
            return set()

    def run_pylint_on_modified_files(self, repo_path: str, modified_files: Optional[set[str]] = None) -> LintResult:
        """Run pylint on modified Python files.

        Args:
            repo_path: Path to the repository root
            modified_files: Optional set of modified files. If None, will detect automatically.

        Returns:
            LintResult with pylint check results
        """
        if modified_files is None:
            modified_files = self.get_modified_files(repo_path)

        if not modified_files:
            return LintResult(success=True, errors=[], output="No modified Python files to check")

        try:
            all_errors = []
            combined_output = ""
            all_success = True

            for file_path in sorted(modified_files):
                full_path = os.path.join(repo_path, file_path)

                # Skip if file doesn't exist (might be deleted)
                if not os.path.exists(full_path):
                    continue

                print(f"🔍 [PYLINT] Running pylint on: {file_path}")

                cmd = ["pylint", "--errors-only", "--disable=E0401,E0611", full_path]
                result = subprocess.run(
                    cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                combined_output += f"\n--- pylint {file_path} ---\n"
                combined_output += result.stdout
                if result.stderr:
                    combined_output += "\nSTDERR:\n" + result.stderr

                if result.returncode != 0:
                    all_success = False
                    # Parse pylint output to extract errors
                    for line in result.stdout.split('\n'):
                        if line.strip() and not line.startswith('*') and not line.startswith('-'):
                            # Try to parse pylint error format: file:line:col: error_code: message
                            parts = line.split(':', 4)
                            if len(parts) >= 4:
                                try:
                                    line_num = int(parts[1])
                                    col_num = int(parts[2]) if len(parts) > 2 else 1
                                    error_code = parts[3].strip() if len(parts) > 3 else "PYLINT"
                                    message = parts[4].strip() if len(parts) > 4 else line

                                    all_errors.append(LintError(
                                        file_path=file_path,
                                        line=line_num,
                                        column=col_num,
                                        code=error_code,
                                        message=message,
                                        severity="error"
                                    ))
                                except (ValueError, IndexError):
                                    # If we can't parse, add as generic error
                                    all_errors.append(LintError(
                                        file_path=file_path,
                                        line=1,
                                        column=1,
                                        code="PYLINT",
                                        message=line,
                                        severity="error"
                                    ))
                            else:
                                # Generic error
                                all_errors.append(LintError(
                                    file_path=file_path,
                                    line=1,
                                    column=1,
                                    code="PYLINT",
                                    message=line,
                                    severity="error"
                                ))

            return LintResult(success=all_success, errors=all_errors, output=combined_output.strip())

        except subprocess.TimeoutExpired:
            return LintResult(
                success=False,
                errors=[LintError(
                    file_path="",
                    line=1,
                    column=1,
                    code="TIMEOUT",
                    message="Pylint checks timed out",
                    severity="error"
                )],
                output="Pylint checks timed out after 60 seconds"
            )
        except Exception as e:
            return LintResult(
                success=False,
                errors=[LintError(
                    file_path="",
                    line=1,
                    column=1,
                    code="PYLINT004",
                    message=f"Pylint check failed: {str(e)}",
                    severity="error"
                )],
                output=str(e)
            )


def create_linter(linter_config: Optional[dict[str, bool]] = None) -> CodeLinter:
    """Factory function to create a CodeLinter instance.

    Args:
        linter_config: Optional dictionary mapping linter names to enable/disable flags.
                      Each name corresponds to pre-commit hooks:
                      - 'ast' -> python-check-ast hook
                      - 'ruff' -> ruff, ruff-format hooks
                      - 'black' -> black hook
                      - 'mypy' -> mypy hook
                      - 'flake8' -> flake8 hook
                      - 'isort' -> isort hook
                      Defaults to only 'ast' and 'ruff' enabled.

    Examples:
        # Default (AST + ruff pre-commit hooks)
        linter = create_linter()

        # Custom configuration
        linter = create_linter({'ast': True, 'ruff': True, 'black': True})

        # Only AST checks
        linter = create_linter({'ast': True})
    """
    return CodeLinter(linter_config)
