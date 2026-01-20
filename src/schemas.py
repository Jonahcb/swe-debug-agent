"""Pydantic schemas for constrained decoding with SGLang.

These schemas define the structured output formats for:
1. Coding agent's final output (candidate fixes)
2. Fix checker subagent's input/output
3. Simple check fixes tool input

SGLang's constrained decoding uses JSON Schema to enforce valid output structure,
improving reliability and reducing parsing errors.
"""

from pydantic import BaseModel, Field


# =============================================================================
# Coding Agent Output Schemas
# =============================================================================


class ModifiedFile(BaseModel):
    """A single file modification in a candidate fix."""

    file_path: str = Field(description="Full absolute path to the file being modified")
    old_string: str = Field(description="Existing code block to replace, with enough context to be unique")
    new_string: str = Field(description="New code to replace the old_string with")


class CandidateFix(BaseModel):
    """A single candidate fix that addresses one or more bugs."""

    description: str = Field(description="Brief description of this candidate fix and which bugs it addresses")
    modified_files: list[ModifiedFile] = Field(description="List of file modifications for this fix")


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
    """A single fix to validate - file path and old string pair."""

    file_path: str = Field(description="Path to the file to check")
    old_string: str = Field(description="The string that should exist in the file")


class FixCheckerInput(BaseModel):
    """Input format for the fix_checker subagent.
    
    The coder agent must produce this structured format when calling
    the fix_checker subagent.
    """

    fixes_to_validate: list[FixTuple] = Field(
        description="List of (file_path, old_string) pairs to validate"
    )


class FixValidationResult(BaseModel):
    """Result of validating a single fix."""

    fix_index: int = Field(description="Index of the fix in the input list (0-based)")
    file_path: str = Field(description="Path to the file checked")
    is_valid: bool = Field(description="Whether the old_string was found in the file")
    message: str = Field(description="Detailed validation message")


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
