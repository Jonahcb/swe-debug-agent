"""Pydantic schemas for constrained decoding with SGLang.

These schemas define the structured output formats for:
1. Coding agent's final output (candidate fixes)
2. Fix checker subagent's input/output
3. Simple check fixes tool input

SGLang's constrained decoding uses JSON Schema to enforce valid output structure,
improving reliability and reducing parsing errors.
"""

from typing import Union
from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Coding Agent Output Schemas
# =============================================================================


class ModifiedFile(BaseModel):
    """A single file modification in a candidate fix."""

    file_path: str = Field(description="Relative path to the file being modified (relative to worktree root)", min_length=1)
    old_string: str = Field(
        description="Existing code block to replace, with enough context to be unique",
        min_length=1
    )
    new_string: str = Field(description="New code to replace the old_string with")

    @model_validator(mode='after')
    def validate_file_modification(self) -> 'ModifiedFile':
        """Validate that the file modification is complete and valid."""
        reasons = []

        if not self.file_path or self.file_path.strip() == "":
            reasons.append("missing or empty file_path")

        if not self.old_string or self.old_string.strip() == "":
            reasons.append("missing or empty old_string")

        # new_string can be empty (for deletions), so no validation needed

        if reasons:
            raise ValueError(f"Invalid file modification: {', '.join(reasons)}")

        return self


class CandidateFix(BaseModel):
    """A single candidate fix that addresses one or more bugs."""

    description: str = Field(
        description="Brief description of this candidate fix and which bugs it addresses"
    )
    modified_files: list[ModifiedFile] = Field(
        description="List of file modifications for this fix"
    )


class CoderOutput(BaseModel):
    """Structured output from the coding agent containing candidate fixes.

    This schema is used for SGLang constrained decoding to ensure the
    coder agent always produces valid, parseable candidate fixes.
    """

    candidates: list[CandidateFix] = Field(
        description="List of exactly 3 distinct candidate fixes",
        min_length=1,
        max_length=5,
    )
    summary: str = Field(description="Brief summary of the candidate approaches generated")


# =============================================================================
# Fix Checker Schemas
# =============================================================================


class FixTuple(BaseModel):
    """A single fix to validate - file path, old string, and new string."""

    file_path: str = Field(description="Path to the file to check")
    old_string: str = Field(description="The string that should exist in the file")
    new_string: str = Field(description="The new string to replace the old_string with")


class FixCheckerInput(BaseModel):
    """Input format for the fix_checker subagent.

    The coder agent must produce a dictionary with a 'root' key containing
    the list of fix tuples when calling the fix_checker subagent.
    """

    root: list[FixTuple]


class FixValidationResult(BaseModel):
    """Result of validating a single fix."""

    fix_index: int = Field(description="Index of the fix in the input list (0-based)")
    file_path: str = Field(description="Path to the file checked")
    is_valid: bool = Field(description="Whether the old_string was found in the file")
    message: str = Field(description="Detailed validation message")
    file_contents: Union[str, None] = Field(
        default=None, description="Full file contents when validation fails (for debugging)"
    )


class FixCheckerOutput(BaseModel):
    """Output format from the fix_checker subagent.

    This schema is used for SGLang constrained decoding to ensure
    the fix_checker always returns structured validation results.
    """

    all_valid: bool = Field(description="True if all fixes are valid, False otherwise")
    results: list[FixValidationResult] = Field(description="Detailed results for each fix")
    summary: str = Field(description="Overall summary of validation results")


# =============================================================================
# Simple Check Fixes Tool Input Schema
# =============================================================================


class SimpleCheckFixesInput(BaseModel):
    """Structured input for the simple_check_fixes tool.

    This schema enforces that the fix_checker subagent always provides
    properly structured input to the simple_check_fixes tool.
    """

    fixes_list: list[tuple[str, str]] = Field(
        description="List of (file_path, old_string) tuples to validate"
    )


# =============================================================================
# Final Bug Analysis Tool Input Schema
# =============================================================================


class BugInfo(BaseModel):
    """Information about a single identified bug."""

    relevant_files_and_lines: str = Field(
        description="Specific file paths and line numbers where the bug manifests (e.g., 'file.py:123-125, file2.py:456')"
    )
    description: str = Field(
        description="Technical low-level one sentence description of the bug and potential fixes"
    )


class FinalBugAnalysisInput(BaseModel):
    """Input format for the final_bug_analysis tool.

    This schema enforces SGLang constrained decoding to ensure the architect
    provides bug analysis in the exact expected format. Due to SGLang constraints,
    the output must include a 'root' key containing the bug dictionary.
    """

    root: dict[str, BugInfo] = Field(
        description="Dictionary of bugs identified with keys starting with 'bug_'"
    )


class SubmitFixesInput(BaseModel):
    """Input format for the submit_fixes tool.

    This schema enforces SGLang constrained decoding to ensure the coder agent
    provides candidate fixes in the exact expected format. Due to SGLang constraints,
    the output must include a 'root' key containing the list of candidate fixes.
    """

    root: list[CandidateFix] = Field(
        description="List of candidate fixes to submit for execution and evaluation"
    )


# =============================================================================
# Schema Utilities
# =============================================================================


def get_json_schema(model: type[BaseModel]) -> dict:
    """Get JSON schema from a Pydantic model for SGLang constrained decoding.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        JSON schema dict suitable for SGLang's response_format parameter
    """
    return model.model_json_schema()


# Pre-computed schemas for use with SGLang
CODER_OUTPUT_SCHEMA = get_json_schema(CoderOutput)
FIX_CHECKER_INPUT_SCHEMA = get_json_schema(FixCheckerInput)
FIX_CHECKER_OUTPUT_SCHEMA = get_json_schema(FixCheckerOutput)
SIMPLE_CHECK_FIXES_INPUT_SCHEMA = get_json_schema(SimpleCheckFixesInput)
SUBMIT_FIXES_INPUT_SCHEMA = get_json_schema(SubmitFixesInput)
