"""Linting utilities for validating code changes using sglang pre-commit."""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
    errors: List[LintError]
    output: str


class CodeLinter:
    """Utility class for linting Python code using sglang pre-commit hooks."""

    def __init__(self, linter_config: Optional[Dict[str, bool]] = None):
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

        # Build pre-commit command based on enabled linters
        self.pre_commit_cmd = self._build_pre_commit_command()

    def _build_pre_commit_command(self) -> List[str]:
        """Build the pre-commit command based on enabled linters."""
        enabled_linters = [name for name, enabled in self.linter_config.items() if enabled]

        # If no linters are enabled, return empty command
        if not enabled_linters:
            return []

        # Map linter names to pre-commit hook IDs
        hook_mapping = {
            'ast': ['python-check-ast'],
            'ruff': ['ruff', 'ruff-format'],
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

        # If we have specific hooks, run only those
        if hook_ids:
            cmd = ["pre-commit", "run"]
            for hook_id in hook_ids:
                cmd.extend(["--hook-id", hook_id])
            cmd.append("--all-files")
            return cmd
        else:
            # Fallback to all files if no matching hooks
            return ["pre-commit", "run", "--all-files"]

    def enable_linter(self, linter_name: str, enabled: bool = True):
        """Enable or disable a specific linter.

        Args:
            linter_name: Name of the linter ('ast', 'ruff', 'black', 'mypy', 'flake8', 'isort')
            enabled: Whether to enable the linter
        """
        if linter_name in self.linter_config:
            self.linter_config[linter_name] = enabled
            self.pre_commit_cmd = self._build_pre_commit_command()

    def get_enabled_linters(self) -> List[str]:
        """Get list of currently enabled linters."""
        return [name for name, enabled in self.linter_config.items() if enabled]

    def run_sglang_pre_commit(self, repo_path: str, file_path: Optional[str] = None) -> LintResult:
        """Run sglang pre-commit checks on the repository.

        Args:
            repo_path: Path to the repository root
            file_path: Optional specific file to check (relative to repo_path)

        Returns:
            LintResult with pre-commit check results
        """
        try:
            # Build the command based on current configuration
            cmd = self._build_pre_commit_command()
            if not cmd:
                # No linters enabled
                return LintResult(success=True, errors=[], output="No linters enabled")

            # Log the pre-commit execution details
            print(f"🔍 [LINTING] Running pre-commit on repository path: {repo_path}")
            print(f"🔍 [LINTING] Full pre-commit command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for pre-commit as it may run multiple tools
            )

            # Pre-commit returns 0 for success, non-zero for failure
            success = result.returncode == 0

            # For failed pre-commit runs, return the full output as a single error
            errors = []
            combined_output = result.stdout
            if result.stderr:
                combined_output += "\nSTDERR:\n" + result.stderr

            if not success:
                errors.append(LintError(
                    file_path=file_path or repo_path,
                    line=1,
                    column=1,
                    code="PRECOMMIT001",
                    message=f"Pre-commit failed:\n{combined_output}",
                    severity="error"
                ))

            return LintResult(success=success, errors=errors, output=combined_output)

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
        new_string: str
    ) -> Tuple[bool, List[LintError], str]:
        """Validate a single fix by applying it and running sglang pre-commit checks.

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
            with open(full_file_path, 'r', encoding='utf-8') as f:
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
            # Run sglang pre-commit checks on the entire repo
            pre_commit_result = self.run_sglang_pre_commit(repo_path, file_path)

            success = pre_commit_result.success
            all_errors = pre_commit_result.errors
            combined_output = f"SGLang Pre-commit Check: {pre_commit_result.output}"

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
        fixes: List[Tuple[str, str, str]]
    ) -> Dict[int, Tuple[bool, List[LintError], str]]:
        """Validate multiple fixes by running linters on each.

        Args:
            fixes: List of (file_path, old_string, new_string) tuples

        Returns:
            Dict mapping fix index to (success, errors, output)
        """
        results = {}

        for i, (file_path, old_string, new_string) in enumerate(fixes):
            success, errors, output = self.validate_fix_with_linters(
                repo_path, file_path, old_string, new_string
            )
            results[i] = (success, errors, output)

        return results


def create_linter(linter_config: Optional[Dict[str, bool]] = None) -> CodeLinter:
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