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

    def __init__(self, pre_commit_cmd: Optional[List[str]] = None):
        """Initialize the linter.

        Args:
            pre_commit_cmd: Command to run for linting. Defaults to sglang pre-commit command.
        """
        if pre_commit_cmd is None:
            self.pre_commit_cmd = ["pre-commit", "run", "--all-files"]
        else:
            self.pre_commit_cmd = pre_commit_cmd

    def run_sglang_pre_commit(self, repo_path: str, file_path: Optional[str] = None) -> LintResult:
        """Run sglang pre-commit checks on the repository.

        Args:
            repo_path: Path to the repository root
            file_path: Optional specific file to check (relative to repo_path)

        Returns:
            LintResult with pre-commit check results
        """
        try:
            # Run pre-commit command
            cmd = self.pre_commit_cmd.copy()
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for pre-commit as it may run multiple tools
            )

            # Pre-commit returns 0 for success, non-zero for failure
            success = result.returncode == 0

            # Parse output to extract errors
            errors = []
            output_lines = result.stdout.split('\n') + result.stderr.split('\n')

            # Look for failed checks in the output
            current_file = None
            for line in output_lines:
                line = line.strip()
                if not line:
                    continue

                # Try to identify file paths and error messages
                if line.startswith(('Failed', 'Error', 'SyntaxError')) or 'error' in line.lower():
                    if current_file:
                        errors.append(LintError(
                            file_path=current_file,
                            line=1,  # We don't have line info from pre-commit output
                            column=1,
                            code="PRECOMMIT001",
                            message=line,
                            severity="error"
                        ))
                    else:
                        # General error not associated with a specific file
                        errors.append(LintError(
                            file_path=file_path or repo_path,
                            line=1,
                            column=1,
                            code="PRECOMMIT002",
                            message=line,
                            severity="error"
                        ))

            # If no specific errors were parsed but command failed, create a general error
            if not success and not errors:
                errors.append(LintError(
                    file_path=file_path or repo_path,
                    line=1,
                    column=1,
                    code="PRECOMMIT003",
                    message="Pre-commit checks failed",
                    severity="error"
                ))

            combined_output = result.stdout
            if result.stderr:
                combined_output += "\nSTDERR:\n" + result.stderr

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


def create_linter(pre_commit_cmd: Optional[List[str]] = None) -> CodeLinter:
    """Factory function to create a CodeLinter instance.

    Args:
        pre_commit_cmd: Optional custom pre-commit command. Defaults to sglang pre-commit.
    """
    return CodeLinter(pre_commit_cmd)