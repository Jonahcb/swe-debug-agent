"""Linting utilities for validating code changes."""

import ast
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
    """Utility class for linting Python code."""

    def __init__(self):
        pass

    def check_python_ast(self, code: str, file_path: str = "<string>") -> LintResult:
        """Check Python code for AST parsing errors.

        Args:
            code: The Python code to check
            file_path: File path for error reporting

        Returns:
            LintResult with any syntax errors found
        """
        try:
            ast.parse(code, filename=file_path)
            return LintResult(success=True, errors=[], output="")
        except SyntaxError as e:
            error = LintError(
                file_path=file_path,
                line=e.lineno or 1,
                column=e.offset or 1,
                code="E999",
                message=f"SyntaxError: {e.msg}",
                severity="error"
            )
            return LintResult(success=False, errors=[error], output=str(e))
        except Exception as e:
            error = LintError(
                file_path=file_path,
                line=1,
                column=1,
                code="E999",
                message=f"AST Error: {str(e)}",
                severity="error"
            )
            return LintResult(success=False, errors=[error], output=str(e))

    def run_ruff_check(self, repo_path: str, file_path: Optional[str] = None) -> LintResult:
        """Run ruff check on the repository or a specific file.

        Args:
            repo_path: Path to the repository root
            file_path: Optional specific file to check (relative to repo_path)

        Returns:
            LintResult with ruff check results
        """
        target_path = file_path if file_path else repo_path

        try:
            # Run ruff check with JSON output
            cmd = ["ruff", "check", target_path, "--output-format", "json"]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            errors = []
            if result.stdout:
                try:
                    import json
                    issues = json.loads(result.stdout)
                    errors = [
                        LintError(
                            file_path=issue.get("filename", ""),
                            line=issue.get("location", {}).get("row", 0),
                            column=issue.get("location", {}).get("column", 0),
                            code=issue.get("code", ""),
                            message=issue.get("message", ""),
                            severity="error" if issue.get("code", "").startswith("E") else "warning"
                        )
                        for issue in issues
                    ]
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a single error with the raw output
                    errors = [LintError(
                        file_path=target_path,
                        line=1,
                        column=1,
                        code="RUFF001",
                        message="Failed to parse ruff output",
                        severity="error"
                    )]

            success = result.returncode == 0
            output = result.stdout + (result.stderr if result.stderr else "")

            return LintResult(success=success, errors=errors, output=output)

        except subprocess.TimeoutExpired:
            return LintResult(
                success=False,
                errors=[LintError(
                    file_path=target_path,
                    line=1,
                    column=1,
                    code="TIMEOUT",
                    message="Ruff check timed out",
                    severity="error"
                )],
                output="Ruff check timed out after 30 seconds"
            )
        except Exception as e:
            return LintResult(
                success=False,
                errors=[LintError(
                    file_path=target_path,
                    line=1,
                    column=1,
                    code="RUFF002",
                    message=f"Ruff check failed: {str(e)}",
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
        """Validate a single fix by applying it and running linters.

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

        # Run AST check on the modified content
        ast_result = self.check_python_ast(modified_content, file_path)

        # If AST check fails, return immediately
        if not ast_result.success:
            return False, ast_result.errors, ast_result.output

        # Run ruff check on the entire repo (since changes might affect imports/other files)
        ruff_result = self.run_ruff_check(repo_path)

        # Combine errors from both linters
        all_errors = ast_result.errors + ruff_result.errors
        combined_output = f"AST Check: {ast_result.output}\nRuff Check: {ruff_result.output}"

        success = ast_result.success and ruff_result.success

        return success, all_errors, combined_output

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


def create_linter() -> CodeLinter:
    """Factory function to create a CodeLinter instance."""
    return CodeLinter()