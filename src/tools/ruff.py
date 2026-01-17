"""Ruff Python linter tool."""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Union


@dataclass
class LintResult:
    file: str
    line: int
    column: int
    code: str
    message: str
    severity: str = "error"


class RuffLinter:
    """Wrapper for the Ruff Python linter."""

    def __init__(self, config_path: Union[str, None] = None) -> None:
        self.config_path = config_path

    def _run_ruff(self, args: list[str], cwd: Union[str, None] = None) -> subprocess.CompletedProcess:
        cmd = ["ruff"] + args
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    def check(self, path: str, fix: bool = False) -> list[LintResult]:
        """Run ruff check on a file or directory."""
        args = ["check", path, "--output-format", "json"]
        if fix:
            args.append("--fix")
        
        result = self._run_ruff(args)
        if not result.stdout:
            return []
        
        try:
            issues = json.loads(result.stdout)
            return [
                LintResult(
                    file=issue.get("filename", ""),
                    line=issue.get("location", {}).get("row", 0),
                    column=issue.get("location", {}).get("column", 0),
                    code=issue.get("code", ""),
                    message=issue.get("message", ""),
                )
                for issue in issues
            ]
        except json.JSONDecodeError:
            return []

    def format(self, path: str, check_only: bool = False) -> bool:
        """Run ruff format on a file or directory."""
        args = ["format", path]
        if check_only:
            args.append("--check")
        
        result = self._run_ruff(args)
        return result.returncode == 0

    def fix(self, path: str) -> list[LintResult]:
        """Run ruff check with auto-fix enabled."""
        return self.check(path, fix=True)


def lint_file(path: str, fix: bool = False) -> list[dict]:
    """Convenience function to lint a single file."""
    linter = RuffLinter()
    results = linter.check(path, fix=fix)
    return [
        {
            "file": r.file,
            "line": r.line,
            "column": r.column,
            "code": r.code,
            "message": r.message,
        }
        for r in results
    ]


def format_file(path: str) -> bool:
    """Convenience function to format a single file."""
    return RuffLinter().format(path)
