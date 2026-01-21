#!/usr/bin/env python3
"""
Language Agent Tree Search (LATS) for SGLang LoRA MoE Bug Fixing.

This module implements a tree search algorithm that uses the existing agent
infrastructure (architect, critic, test runner agent, librarians) to
iteratively generate, evaluate, and refine code fixes.

Architecture:
    - TreeState: LangGraph state containing the search tree
    - Node: Individual attempts with code snapshots and scores
    - LATS Nodes: select -> expand (using agents) -> execute -> backprop
    - Integration with existing agents for specialized tasks
    - SQLite persistence for crash recovery
"""

from __future__ import annotations
import sys
from pathlib import Path

# Load environment from .env file FIRST, before any other imports
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        pass

import json
import math
import os
import re
import signal
import subprocess
import sqlite3
import sys
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# LangSmith tracing imports
from langsmith import traceable, trace, Client

# Import existing agent infrastructure
from src.agents import create_agent, get_llm
from src.agents.agents import (
    architect_node,
    coder_node,
    critic_node,
    internal_librarian_node,
    external_librarian_node,
)
from src.memory import store
from src.worktree_utils import setup_worktree, cleanup_worktree
from src.utils.git_manager import WorktreeManager
from src.state import AgentState

# Load LATS prompts from prompts.yaml
PROMPTS_PATH = Path(__file__).parent / "config" / "prompts.yaml"
with open(PROMPTS_PATH, "r") as f:
    PROMPTS = yaml.safe_load(f)

LATS_PROMPTS = PROMPTS.get("lats", {})

# Load settings
from config.settings import settings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Note: No file restrictions - coder agent can modify any file

# Immutable file patterns (kernels, configs, etc.)
IMMUTABLE_PATTERNS = [
    r"sglang/kernels/.*",
    r".*global_config\.py$",
    r".*triton.*kernel.*\.py$",
]

# Test configuration
TEST_TIMEOUT = 600  # 10 minutes

# LATS configuration
NUM_CANDIDATES = 3  # Number of candidates to generate per expansion
MAX_DEPTH = 10  # Maximum tree depth
UCB_EXPLORATION_WEIGHT = 1.41  # UCB exploration constant (sqrt(2))
MAX_ITERATIONS = 50  # Maximum LATS iterations

# SQLite database for persistence
CHECKPOINT_DB = "lats_checkpoint.db"


# =============================================================================
# NODE DATA STRUCTURE
# =============================================================================


@dataclass
class Node:
    """
    Represents a single attempt in the LATS tree.

    Each node stores the complete code state (not diffs) and test results.
    The tree structure enables UCB-based selection and backpropagation.
    """

    # Unique identifier for this node
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Code changes as blocks: {"filename": [{"old_string": "...", "new_string": "..."}]}
    code_changes: dict[str, list[dict]] = field(default_factory=dict)

    # Test execution results
    test_output: str = ""
    return_code: int | None = None

    # Score (0.0 to 1.0): 1.0 = passed, 0.0 = segfault, intermediate for errors
    score: float = 0.0

    # Tree structure
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    # UCB statistics
    visits: int = 0
    value: float = 0.0

    # Metadata
    hypothesis: str = ""  # The reasoning behind this attempt
    agent_feedback: str = ""  # Feedback from critic/architect agents
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    depth: int = 0

    def ucb_score(
        self, total_visits: int, exploration_weight: float = UCB_EXPLORATION_WEIGHT
    ) -> float:
        """Calculate Upper Confidence Bound score for node selection."""
        if self.visits == 0:
            return float("inf")  # Prioritize unvisited nodes

        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(total_visits + 1) / self.visits)
        return exploitation + exploration

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state (test passed or max depth)."""
        return self.score >= 1.0 or self.depth >= MAX_DEPTH

    def is_leaf(self) -> bool:
        """Check if this node has no children."""
        return len(self.children_ids) == 0

    def to_dict(self) -> dict:
        """Serialize node to dictionary for persistence."""
        return {
            "id": self.id,
            "code_changes": self.code_changes,
            "test_output": self.test_output,
            "return_code": self.return_code,
            "score": self.score,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "visits": self.visits,
            "value": self.value,
            "hypothesis": self.hypothesis,
            "agent_feedback": self.agent_feedback,
            "created_at": self.created_at,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        """Deserialize node from dictionary."""
        return cls(**data)


# =============================================================================
# TREE STATE (LATS + Agent Integration)
# =============================================================================


def create_trim_messages_reducer(max_tokens: int = 28000):
    """Factory function to create a message trimming reducer with the correct signature.

    LangGraph requires reducers to have signature (a, b) -> c, so we use a closure
    to capture the max_tokens parameter.
    """

    def trim_messages_reducer(existing_messages: list, new_messages: list) -> list:
        """Custom message reducer that trims messages to stay under token limit.

        Uses trim_messages to keep recent messages and system messages while
        preventing context overflow. Adds context awareness when trimming occurs.
        """
        # First add new messages to existing ones
        all_messages = add_messages(existing_messages, new_messages)

        # Check if we need to trim
        should_trim = False
        if all_messages and len(all_messages) > 10:  # Only trim if we have many messages
            try:
                # Quick check: estimate tokens (rough approximation: ~4 chars per token)
                estimated_chars = sum(
                    len(str(msg.content)) for msg in all_messages if hasattr(msg, "content")
                )
                estimated_tokens = estimated_chars // 4
                should_trim = estimated_tokens > max_tokens * 0.8  # Trim when 80% of limit reached
            except:
                should_trim = len(all_messages) > 20  # Fallback: trim if too many messages

        if should_trim:
            try:
                # Use approximate token counting for speed
                trimmed = trim_messages(
                    all_messages,
                    max_tokens=max_tokens,
                    token_counter="approximate",  # Fast approximate counting
                    strategy="last",  # Keep most recent messages
                    start_on="human",  # Ensure conversation starts with human message
                    include_system=True,  # Keep system messages
                    allow_partial=False,  # Don't split messages
                )

                # Add a context awareness message if we trimmed
                if len(trimmed) < len(all_messages):
                    from langchain_core.messages import SystemMessage

                    context_warning = SystemMessage(
                        content="⚠️ CONTEXT TRIMMED: Previous conversation history has been condensed to stay within token limits. "
                        "Focus on the most recent analysis and maintain continuity with the current debugging session."
                    )
                    trimmed.insert(0, context_warning)  # Add at the beginning after system messages

                return trimmed
            except Exception as e:
                # If trimming fails, return the original messages to avoid losing data
                print(f"Warning: Message trimming failed: {e}")
                return all_messages

        return all_messages

    return trim_messages_reducer


def trim_messages_if_needed(messages: list, max_tokens: int = 28000) -> list:
    """Trim messages if they're approaching the token limit.

    Returns trimmed messages with context awareness warning if trimming occurred.
    """
    if not messages or len(messages) < 10:
        return messages

    try:
        # Estimate tokens (rough approximation: ~4 chars per token)
        estimated_chars = sum(len(str(msg.content)) for msg in messages if hasattr(msg, "content"))
        estimated_tokens = estimated_chars // 4

        if estimated_tokens > max_tokens * 0.8:  # Trim when 80% of limit reached
            trimmed = trim_messages(
                messages,
                max_tokens=max_tokens,
                token_counter="approximate",
                strategy="last",
                start_on="human",
                include_system=True,
                allow_partial=False,
            )

            # Add context awareness warning if we trimmed
            if len(trimmed) < len(messages):
                from langchain_core.messages import SystemMessage

                context_warning = SystemMessage(
                    content="⚠️ CONTEXT TRIMMED: Previous conversation history has been condensed to stay within token limits. "
                    "Focus on the most recent analysis and maintain continuity with the current debugging session."
                )
                trimmed.insert(0, context_warning)

            return trimmed
    except Exception as e:
        print(f"Warning: Message trimming failed: {e}")

    return messages


class TreeState(TypedDict):
    """LangGraph state containing the LATS tree and agent context."""

    # The search tree stored as a dictionary of nodes
    nodes: dict[str, dict]  # node_id -> serialized Node

    # ID of the root node
    root_id: str

    # ID of the currently selected node
    selected_node_id: str | None

    # List of candidate node IDs to evaluate
    candidate_ids: list[str]

    # Current iteration count
    iteration: int

    # Status: "searching" | "found" | "max_iterations" | "max_depth"
    status: str

    # Best solution found so far (node ID with highest score)
    best_node_id: str | None

    # Original input/task description
    input: str

    # Messages for agent communication (accumulated with trimming in nodes)
    messages: Annotated[list, add_messages]

    # Context shared between agents
    context: dict[str, Any]

    # Current agent (for routing)
    current_agent: str

    # Worktree information
    repo_path: str | None  # Path to the isolated worktree


# =============================================================================
# WORKTREE WRAPPER FUNCTIONS
# =============================================================================


def setup_worktree_lats(state: TreeState) -> dict:
    """Set up a git worktree environment for LATS execution."""
    # Convert TreeState to AgentState for setup_worktree function
    agent_state: AgentState = {
        "messages": state.get("messages", []),
        "task": state.get("input", ""),
        "context": state.get("context", {}),
        "current_agent": state.get("current_agent", "lats"),
        "status": state.get("status", "running"),
        "iteration_count": state.get("iteration", 0),
        "search_calls_used": 0,  # Not used in LATS
    }

    result = setup_worktree(agent_state)

    # Update the workspace path in context to use worktree path
    updated_context = result.get("context", {})
    worktree_path = result.get("repo_path")

    # Update workspace in context to point to worktree
    if worktree_path:
        updated_context["workspace"] = worktree_path
        # Set environment variable for tools that need it
        os.environ["WORKTREE_REPO_PATH"] = worktree_path

    return {
        "repo_path": worktree_path,
        "context": updated_context,
    }


def cleanup_worktree_lats(state: TreeState) -> dict:
    """Clean up the git worktree environment for LATS."""
    # Convert TreeState to AgentState for cleanup_worktree function
    agent_state: AgentState = {
        "messages": state.get("messages", []),
        "task": state.get("input", ""),
        "context": state.get("context", {}),
        "current_agent": state.get("current_agent", "lats"),
        "status": state.get("status", "completed"),
        "iteration_count": state.get("iteration", 0),
        "search_calls_used": 0,  # Not used in LATS
        "repo_path": state.get("repo_path"),
    }

    result = cleanup_worktree(agent_state)

    return {
        "status": "completed",
        "context": result.get("context", {}),
    }


# =============================================================================
# TOOL: RUN AND TEST CODE
# =============================================================================


def is_file_mutable(filename: str) -> bool:
    """Check if a file can be modified."""
    # Check against immutable patterns (critical files that should never be modified)
    for pattern in IMMUTABLE_PATTERNS:
        if re.match(pattern, filename):
            return False

    # Allow modification of any other file
    return True


def restore_original_files(workspace_dir: str) -> None:
    """Restore original files from .lats_orig backups."""
    workspace = Path(workspace_dir)

    # Find all .lats_orig backup files and restore them
    for backup_path in workspace.rglob("*.lats_orig"):
        original_path = backup_path.with_suffix(backup_path.suffix[:-10])  # Remove .lats_orig
        if backup_path.exists():
            try:
                original_path.write_text(backup_path.read_text())
                backup_path.unlink()  # Remove backup after restore
                print(f"   Restored {original_path} to original")
            except Exception as e:
                print(f"   Failed to restore {original_path}: {e}")


def restore_selected_node_files(workspace_dir: str) -> None:
    """Restore files to the selected node's state from .lats_session_backup."""
    workspace = Path(workspace_dir)

    # Find all .lats_session_backup files and restore them
    for backup_path in workspace.rglob("*.lats_session_backup"):
        original_path = backup_path.with_suffix(
            backup_path.suffix[:-21]
        )  # Remove .lats_session_backup
        if backup_path.exists():
            try:
                original_path.write_text(backup_path.read_text())
                print(f"   Restored {original_path} to selected node state")
            except Exception as e:
                print(f"   Failed to restore {original_path}: {e}")
    # Note: We keep the session backups for future restorations


def filter_lora_test_output(output: str) -> str:
    """
    Filter LoRA test output to extract only the relevant comparison results.

    This removes the verbose model loading information and keeps only the
    logprob comparison results that are useful for debugging.

    Args:
        output: Raw test output from LoRA comparison test

    Returns:
        Filtered output containing only the comparison results
    """
    # Look for the start marker
    start_marker = "Comparing Log Probabilities"
    start_idx = output.find(start_marker)

    if start_idx == -1:
        # If we can't find the comparison section, the test probably failed
        # before running the actual comparison. Return the full output.
        return output

    # Look for the end marker (String Statistics section)
    end_marker = "String Statistics:"
    end_idx = output.find(end_marker, start_idx)

    if end_idx == -1:
        # If we can't find the end marker, return from start to end of output
        return output[start_idx:]
    else:
        # Find the end of the String Statistics line
        # Look for the next newline after the end marker
        next_newline = output.find("\n", end_idx)
        if next_newline == -1:
            return output[start_idx:]
        else:
            return output[start_idx:next_newline]


def run_and_test_code(node: Node, workspace_dir: str) -> tuple[str, int]:
    """
    Execute test with the code snapshot from the given node.

    This tool enforces basic guardrails:
    1. Prevents modification of critical system files
    2. Detects segmentation faults (return code -11)
    3. Captures all stdout/stderr

    Args:
        node: Node containing the code snapshot to test
        workspace_dir: Path to the workspace directory

    Returns:
        Tuple of (test_output, return_code)
    """
    workspace = Path(workspace_dir)

    # Validate file permissions
    for filename in node.code_changes.keys():
        # Normalize filename for validation (remove leading slash if absolute)
        normalized_filename = filename.lstrip("/") if filename.startswith("/") else filename
        if not is_file_mutable(normalized_filename):
            return (
                f"PERMISSION DENIED: Cannot modify '{filename}'. This file matches an immutable pattern.",
                -1,
            )

    # Apply code changes to files
    modified_files = []
    for filename, changes in node.code_changes.items():
        # Ensure filename is relative to workspace (not absolute)
        if filename.startswith("/"):
            # Convert absolute path to relative by removing leading slash
            filename = filename.lstrip("/")

        filepath = workspace / filename

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Backup original if it exists and we haven't backed it up yet
        backup_path = filepath.with_suffix(filepath.suffix + ".lats_orig")
        if filepath.exists() and not backup_path.exists():
            backup_path.write_text(filepath.read_text())

        # Read current file content
        if filepath.exists():
            current_content = filepath.read_text()
        else:
            current_content = ""

        # Apply each change in sequence
        modified_content = current_content
        for change in changes:
            old_string = change["old_string"]
            new_string = change["new_string"]

            # Find and replace the old_string with new_string
            if old_string in modified_content:
                modified_content = modified_content.replace(
                    old_string, new_string, 1
                )  # Replace only first occurrence
            else:
                # If old_string not found, log warning but continue
                print(
                    f"   ⚠️ Warning: Could not find old_string in {filename}: {old_string[:50]}..."
                )

        # Write modified content
        filepath.write_text(modified_content)
        modified_files.append(filepath)

    # Store modified files for potential restoration
    node._modified_files = modified_files

    # ----------------------------------------------------------
    # Temporary for testing

    #     fake_output = """----------------------------------------
    # Prompt 1
    # ----------------------------------------
    # LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

    # Prefill logprobs:
    #   Shape:           [4, 5]
    #   Max difference:  6.472468e-02
    #   Mean difference: 1.176369e-02
    #   Status:          PASS (threshold: 1e-01)

    # Decode logprobs:
    #   Shape:           [32, 5]
    #   Max difference:  7.100700e+00
    #   Mean difference: 1.039451e+00
    #   Status:          FAIL (threshold: 1e-01)

    # Output strings:
    #   Status:      DIFFER
    #   SGLang:       leading provider of software solutions for the global energy industry. SGL's software solutions are used by more than 1,000 companies in 10
    #   HuggingFace:  leading provider of software solutions for the global steel industry. SGL is a global leader in the development and implementation of software solutions for the steel industry. SGL

    # ----------------------------------------
    # Prompt 2
    # ----------------------------------------
    # LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

    # Prefill logprobs:
    #   Shape:           [9, 5]
    #   Max difference:  1.736760e-01
    #   Mean difference: 2.966831e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Decode logprobs:
    #   Shape:           [32, 5]
    #   Max difference:  2.816558e-01
    #   Mean difference: 3.807396e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Output strings:
    #   Status:      MATCH
    #   SGLang:       creating machines that can perform tasks that would typically require human intelligence. AI has the potential to revolutionize the way we live and work, and it is already being
    #   HuggingFace:  creating machines that can perform tasks that would typically require human intelligence. AI has the potential to revolutionize the way we live and work, and it is already being

    # ----------------------------------------
    # Prompt 3
    # ----------------------------------------
    # LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

    # Prefill logprobs:
    #   Shape:           [6, 5]
    #   Max difference:  1.380231e-01
    #   Mean difference: 2.518422e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Decode logprobs:
    #   Shape:           [32, 5]
    #   Max difference:  1.874228e-01
    #   Mean difference: 3.678390e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Output strings:
    #   Status:      MATCH
    #   SGLang:       computers and computing. It includes the theory of computation, algorithms, programming languages, software, hardware, and the interactions between all of these. Computer science is the
    #   HuggingFace:  computers and computing. It includes the theory of computation, algorithms, programming languages, software, hardware, and the interactions between all of these. Computer science is the

    # ----------------------------------------
    # Prompt 4
    # ----------------------------------------
    # LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

    # Prefill logprobs:
    #   Shape:           [5, 5]
    #   Max difference:  4.049866e-01
    #   Mean difference: 5.439247e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Decode logprobs:
    #   Shape:           [32, 5]
    #   Max difference:  4.049864e-01
    #   Mean difference: 3.622374e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Output strings:
    #   Status:      MATCH
    #   SGLang:       Once upon a time, in a small village nestled in the heart of a dense forest, there lived a young girl named Lily. She was known throughout the village
    #   HuggingFace:  Once upon a time, in a small village nestled in the heart of a dense forest, there lived a young girl named Lily. She was known throughout the village

    # ----------------------------------------
    # Prompt 5
    # ----------------------------------------
    # LoRA adapter: sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest

    # Prefill logprobs:
    #   Shape:           [9, 5]
    #   Max difference:  3.239720e-01
    #   Mean difference: 2.978070e-02
    #   Status:          FAIL (threshold: 1e-01)

    # Decode logprobs:
    #   Shape:           [32, 5]
    #   Max difference:  1.144438e+01
    #   Mean difference: 2.303469e+00
    #   Status:          FAIL (threshold: 1e-01)

    # Output strings:
    #   Status:      DIFFER
    #   SGLang:       Please provide a detailed explanation of each component and its function. Additionally, can you explain how these components work together to perform tasks? Please provide a visual representation of
    #   HuggingFace:  A computer consists of several main components, including:

    # 1. Central Processing Unit (CPU): This is the brain of the computer and performs all the calculations and logical

    # ================================================================================
    # Overall Statistics
    # ================================================================================

    # Logprob Differences:
    #   Prefill:
    #     Max of max:   4.049866e-01
    #     Mean of max:  2.210765e-01
    #     Mean of mean: 3.015788e-02
    #   Decode:
    #     Max of max:   1.144438e+01
    #     Mean of max:  3.883830e+00
    #     Mean of mean: 6.908004e-01

    # Logprob Statistics (threshold: 1e-01):
    #   Overall logprob: 0/5 FAILED
    #   Prefill logprob: 1/5
    #   Decode logprob:  0/5

    # String Statistics:
    #   Output strings:  3/5"""

    #     return fake_output, 1

    # ----------------------------------------------------------

    # Prepare test command
    # Run test with subprocess using the configured test command

    with trace("Test_Execution", "tool", inputs={"node_id": node.id}) as rt:
        try:
            sglang_workspace = Path(workspace_dir)

            cmd = [
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                str(sglang_workspace / "test" / "registered" / "lora"),
                "-p",
                "test_lora_hf_sgl_logprob_diff.py",
            ]

            test_env = os.environ.copy()
            test_env["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                env=test_env,
                capture_output=True,
                text=True,
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            return_code = result.returncode

            # Detect segmentation fault
            if return_code == -signal.SIGSEGV:  # -11 on most systems
                output += "\n\n⚠️ CRITICAL: SEGMENTATION FAULT DETECTED!\n"
                output += "The code caused a memory access violation.\n"

            # Filter test output to extract only relevant LoRA comparison results
            filtered_output = filter_lora_test_output(output)

            # Print test output to terminal
            print(f"\n📄 [TEST OUTPUT] Return code: {return_code}")
            print("=" * 50)
            print(filtered_output)
            print("=" * 50)

            rt.end(outputs={"output": filtered_output[:500], "return_code": return_code})

            return filtered_output, return_code

        except subprocess.TimeoutExpired:
            timeout_message = f"ERROR: Test timed out after {TEST_TIMEOUT} seconds"
            print(f"\n⏰ [TEST TIMEOUT] {timeout_message}")

            rt.end(outputs={"error": "timeout", "timeout_seconds": TEST_TIMEOUT})

            return timeout_message, -1
        except Exception as e:
            error_message = f"ERROR: Failed to run test: {e}"
            print(f"\n❌ [TEST ERROR] {error_message}")

            rt.end(outputs={"error": str(e)})

            return error_message, -1


def evaluate_candidate_with_critic(
    candidate: Node, workspace_dir: str, original_code: dict = None
) -> tuple[float, str]:
    """
    Use the critic agent to evaluate a single candidate and provide both score and feedback.

    Args:
        candidate: The candidate node to evaluate
        workspace_dir: Path to the workspace directory
        original_code: Dict of filename -> original code content (optional)

    Returns:
        Tuple of (score, feedback) where score is 0.0-1.0 and feedback is detailed analysis
    """
    print(f"   🤔 Evaluating candidate {candidate.id} with critic...")

    # Get the modified files (showing changes)
    modified_files_info = ""
    for filename, changes in candidate.code_changes.items():
        modified_files_info += f"\n### {filename}:\n"
        for i, change in enumerate(changes):
            modified_files_info += f"**Change {i + 1}:**\n"
            modified_files_info += f"**Old code:**\n```\n{change['old_string']}\n```\n"
            modified_files_info += f"**New code:**\n```\n{change['new_string']}\n```\n"

    # Extract the relevant file content for display
    original_file = (
        list(candidate.code_changes.keys())[0]
        if candidate.code_changes
        else "python/sglang/srt/lora/layers.py"
    )

    # Read original code from filesystem if not provided in original_code dict
    if original_code and original_file in original_code:
        original_code_display = original_code[original_file]
    else:
        # Read original code from filesystem
        try:
            original_file_path = Path(workspace_dir) / original_file
            if original_file_path.exists():
                original_code_display = original_file_path.read_text()
            else:
                original_code_display = f"# Original file not found at: {original_file_path}"
        except Exception as e:
            original_code_display = f"# Error reading original file: {e}"

    # Format as full code block
    original_code_display = f"### {original_file}:\n```\n{original_code_display}\n```"

    # Use the reflection_prompt from the critic config
    critic_prompt = PROMPTS["critic"]["reflection_prompt"].format(
        original_code=original_code_display,
        proposed_code=modified_files_info,
        test_output=candidate.test_output[:2000] if candidate.test_output else "No test output",
        score=0.0,  # First evaluation, no previous score
    )

    # Create mini-state for critic
    critic_state = {
        "messages": [HumanMessage(content=critic_prompt)],
        "context": {},
        "current_agent": "critic",
    }

    # Run critic agent
    result = critic_node(critic_state)

    # Extract response
    critic_response = ""
    if result.get("messages"):
        last_msg = result["messages"][-1]
        critic_response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Parse JSON response
    score = 0.5  # default
    feedback = critic_response

    try:
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", critic_response, re.DOTALL)
        if json_match:
            evaluation_data = json.loads(json_match.group(1))
            score = float(evaluation_data.get("score", 0.5))
            feedback = evaluation_data.get("reasoning", critic_response)
            # Include improvements if present (from reflection_prompt format)
            if "improvements" in evaluation_data and evaluation_data["improvements"]:
                feedback += f"\n\nSuggested improvements:\n" + "\n".join(
                    f"- {imp}" for imp in evaluation_data["improvements"]
                )
        else:
            # Try direct JSON parse
            evaluation_data = json.loads(critic_response)
            score = float(evaluation_data.get("score", 0.5))
            feedback = evaluation_data.get("reasoning", critic_response)
            # Include improvements if present (from reflection_prompt format)
            if "improvements" in evaluation_data and evaluation_data["improvements"]:
                feedback += f"\n\nSuggested improvements:\n" + "\n".join(
                    f"- {imp}" for imp in evaluation_data["improvements"]
                )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"   ⚠️ Could not parse critic evaluation JSON: {e}")
        # Fallback: extract score from text if possible
        score_match = re.search(r'"score":\s*([0-9.]+)', critic_response)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                pass

    # Ensure score is in valid range
    score = max(0.0, min(1.0, score))

    return score, feedback


# =============================================================================
# AGENT-INTEGRATED LATS NODES
# =============================================================================


def get_node(state: TreeState, node_id: str) -> Node:
    """Retrieve a Node from state by ID."""
    return Node.from_dict(state["nodes"][node_id])


def set_node(state: TreeState, node: Node) -> TreeState:
    """Update a Node in state."""
    state["nodes"][node.id] = node.to_dict()
    return state


@traceable(run_type="tool", name="LATS_Select")
def select_node(state: TreeState) -> dict:
    """
    Select the most promising node using UCB (Upper Confidence Bound).

    Traverses from root to a leaf, always choosing the child with highest UCB score.
    """
    print(f"\n🔍 [SELECT] Iteration {state['iteration']}")

    # Start from root
    current_id = state["root_id"]
    current = get_node(state, current_id)

    # Calculate total visits for UCB
    total_visits = sum(Node.from_dict(n).visits for n in state["nodes"].values())

    # Traverse to leaf using UCB
    path = [current.id]
    while not current.is_leaf():
        # Get children
        children = [get_node(state, cid) for cid in current.children_ids]

        # Select child with highest UCB score
        best_child = max(children, key=lambda c: c.ucb_score(total_visits))
        current = best_child
        path.append(current.id)

    print(f"   Selected path: {' -> '.join(path)}")
    print(
        f"   Selected node: {current.id} (depth={current.depth}, visits={current.visits}, score={current.score:.3f})"
    )

    # Check for terminal conditions
    if current.score >= 1.0:
        return {
            "selected_node_id": current.id,
            "status": "found",
            "best_node_id": current.id,
        }

    if current.depth >= MAX_DEPTH:
        return {
            "selected_node_id": current.id,
            "status": "max_depth",
        }

    # Apply the selected node's code changes to the worktree
    # This enables building upon partial solutions
    workspace_dir = state.get("repo_path")

    if current.code_changes:
        print(f"   Applying {len(current.code_changes)} file(s) from selected node to worktree...")
        for filename, changes in current.code_changes.items():
            # Ensure filename is relative to workspace (not absolute)
            if filename.startswith("/"):
                # Convert absolute path to relative by removing leading slash
                filename = filename.lstrip("/")

            filepath = Path(workspace_dir) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if this is the first time modifying this file in this session
            backup_path = filepath.with_suffix(filepath.suffix + ".lats_session_backup")
            if filepath.exists() and not backup_path.exists():
                backup_path.write_text(filepath.read_text())
                print(f"     Backed up {filename}")

            # Read current content and apply changes
            if filepath.exists():
                current_content = filepath.read_text()
            else:
                current_content = ""

            modified_content = current_content
            for change in changes:
                old_string = change["old_string"]
                new_string = change["new_string"]
                if old_string in modified_content:
                    modified_content = modified_content.replace(old_string, new_string, 1)
                else:
                    print(f"     ⚠️ Warning: Could not find old_string in {filename}")

            # Apply the node's changes
            filepath.write_text(modified_content)
            print(f"     Applied changes to {filename}")

    # Prepare context for the next agents
    context = state.get("context", {}).copy()
    context["selected_node_id"] = current.id
    # Set current_code to show the first modified file's changes, or empty if no files
    if current.code_changes:
        first_file = list(current.code_changes.keys())[0]
        first_changes = current.code_changes[first_file]
        context["current_code"] = f"Changes to {first_file}: {len(first_changes)} modification(s)"
    else:
        context["current_code"] = ""
    context["test_output"] = current.test_output
    context["current_score"] = current.score
    context["hypothesis"] = current.hypothesis

    return {
        "selected_node_id": current.id,
        "context": context,
    }


def run_initial_test(state: TreeState) -> dict:
    """
    Run the test on the root node to get initial test output for the architect.

    This provides the baseline test failure information that the architect will analyze.
    """
    print(f"\n🧪 [INITIAL TEST] Running test on root node to get baseline failure")

    workspace_dir = os.environ.get("SGLANG_DIR", "")
    root_id = state["root_id"]
    root = get_node(state, root_id)

    # Run test on root node
    print(f"   Running initial test on root node {root_id}...")
    test_output, return_code = run_and_test_code(root, workspace_dir)

    # Update root node with test results
    root.test_output = test_output
    root.return_code = return_code

    # Score the root node (will be low since it's failing)
    if return_code == 0:
        root.score = 1.0  # Test passed
    else:
        # Use basic scoring for initial failure
        if return_code == -signal.SIGSEGV:
            root.score = 0.0  # Segfault
        elif "import" in test_output.lower() or "module" in test_output.lower():
            root.score = 0.2  # Import error
        elif "syntax" in test_output.lower():
            root.score = 0.2  # Syntax error
        elif "assertion" in test_output.lower() or "failed" in test_output.lower():
            root.score = 0.5  # Test failure (expected for this task)
        else:
            root.score = 0.3  # Other runtime error

    nodes = dict(state["nodes"])
    nodes[root_id] = root.to_dict()

    print(f"   Initial test completed - return_code: {return_code}, score: {root.score:.3f}")

    # Store initial test results in context for architect
    context = state.get("context", {}).copy()
    context["initial_test_output"] = test_output
    context["initial_return_code"] = return_code
    context["initial_score"] = root.score

    return {
        "nodes": nodes,
        "context": context,
    }


def analyze_with_architect(state: TreeState) -> dict:
    """
    Use the architect agent to analyze the current failure and plan fixes.

    The architect receives:
    - Test output and current code
    - Research from internal librarian (SGLang codebase)
    - Research from external librarian (vLLM, HuggingFace, etc.)

    Then produces an analysis with recommended fix strategies.
    """
    print(f"\n🏗️ [ARCHITECT] Analyzing failure and planning fixes (with research)")

    selected_id = state["selected_node_id"]
    selected = get_node(state, selected_id)
    context = state.get("context", {})

    # Get research from librarians (may be empty on first call)
    internal_research = context.get(
        "internal_research", "No internal research yet - architect called first"
    )
    external_research = context.get(
        "external_research", "No external research yet - architect called first"
    )

    # Check if this is the initial analysis (iteration 0) - use initial test results
    is_initial_analysis = state["iteration"] == 0 and "initial_test_output" in context
    repo_path = state.get("repo_path", "")
    if is_initial_analysis:
        # Use initial test results for the first architect analysis
        test_output = context.get("initial_test_output", "No initial test output")
        current_score = context.get("initial_score", 0.0)
        print(f"   Using initial test results (iteration 0)")
    else:
        # Use selected node's results for subsequent analyses
        test_output = selected.test_output if selected.test_output else "No test output yet"
        current_score = selected.score

    # Trim messages if needed to prevent context overflow
    messages = trim_messages_if_needed(state.get("messages", []))

    # Check for context awareness
    context_warning = ""
    if len(messages) > 15:  # If we have many messages, warn about context limits
        context_warning = (
            "\n\n⚠️ CONTEXT LIMIT AWARENESS: This conversation has been ongoing for multiple iterations. "
            "Focus on the most recent findings and avoid repeating previous analyses unless specifically relevant."
        )

    # Prepare prompt for architect with research context
    analysis_prompt = (
        """Research, think about, and compose a plan to fix the SGLang LoRA MoE code bug."""
    )

    # Create a mini-state for the architect
    architect_state = {
        "messages": [HumanMessage(content=analysis_prompt)],
        "context": state.get("context", {}),
        "current_agent": "architect",
    }

    # Make current node's code snapshot available to filesystem tools
    import json

    os.environ["CURRENT_NODE_CODE_SNAPSHOT"] = json.dumps(selected.code_changes)

    # Run architect agent
    result = architect_node(architect_state)

    # Extract analysis from architect's response
    architect_response = ""
    final_bug_analysis_called = False
    final_bug_data = None

    if result.get("messages"):
        last_msg = result["messages"][-1]
        architect_response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # Check if architect called final_bug_analysis tool
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                if tool_call.get("name") == "final_bug_analysis":
                    final_bug_analysis_called = True
                    # Extract the bug analysis data directly from the tool call args (RootModel provides dict)
                    try:
                        final_bug_data = tool_call.get("args", {})
                        print(
                            f"🎯 Architect called final_bug_analysis tool with {len(final_bug_data)} bugs"
                        )
                    except Exception as e:
                        print(f"⚠️ Error parsing final_bug_analysis tool call: {e}")
                    break

    # If final_bug_analysis was called, use that data directly
    if final_bug_analysis_called and final_bug_data:
        parsed_bugs = final_bug_data
        bug_analysis_content = json.dumps(parsed_bugs, indent=2)
    else:
        # Extract structured bug analysis from architect's response (legacy parsing)
        # The architect returns bug analysis in format: {bug_1: {...}, bug_2: {...}, ...}
        # We need to extract only the structured response, filtering out any thinking or fluff

        # Look for the structured bug analysis format (supports multiple formats)
        bug_analysis_content = ""
        parsed_bugs = None

        # First, try to parse markdown header + JSON code block format (new architect format)
        try:
            # Pattern to match "## **Bug #X: Title**" followed by "```json" block
            bug_header_pattern = r"## \*\*Bug #(\d+):.*?\*\*\s*```json\s*(\{.*?\})\s*```"
            matches = re.findall(bug_header_pattern, architect_response, re.DOTALL)

            if matches:
                parsed_bugs = {}
                for bug_number, json_content in matches:
                    try:
                        bug_data = json.loads(json_content.strip())
                        parsed_bugs[f"bug_{bug_number}"] = bug_data
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON

                if parsed_bugs:
                    bug_analysis_content = json.dumps(parsed_bugs, indent=2)
        except Exception:
            pass  # Markdown + JSON parsing failed, try next method

        # If markdown parsing failed, try to parse as JSON format
        try:
            # Look for JSON object containing bug entries
            json_match = re.search(
                r'\{[^{}]*"bug_\d+"[^{}]*\{.*?\}[^{}]*\}', architect_response, re.DOTALL
            )
            if json_match:
                potential_json = json_match.group(0)
                # Try to balance braces to get complete JSON object
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(potential_json):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    json_str = potential_json[:end_pos]
                    parsed_bugs = json.loads(json_str)
                    bug_analysis_content = json.dumps(parsed_bugs, indent=2)
        except (json.JSONDecodeError, ValueError):
            pass  # JSON parsing failed, try next method

        # If JSON parsing failed, try Python dictionary format
        if not parsed_bugs:
            try:
                # Find Python dict structure with bug entries
                dict_match = re.search(
                    r"\{bug_\d+:\s*\{.*?\}\s*\}(?:\s*\{bug_\d+:\s*\{.*?\}\s*\})*",
                    architect_response,
                    re.DOTALL,
                )
                if dict_match:
                    dict_str = dict_match.group(0)
                    # Convert Python dict syntax to JSON-compatible format for parsing
                    json_compatible = dict_str

                    # Replace specific known keys that should be quoted
                    json_compatible = json_compatible.replace("bug_", '"bug_')
                    json_compatible = json_compatible.replace(": {", '": {')
                    json_compatible = json_compatible.replace(
                        "relevant_files_and_lines:", '"relevant_files_and_lines":'
                    )
                    json_compatible = json_compatible.replace("description:", '"description":')

                    # Replace single quotes with double quotes for string values
                    json_compatible = json_compatible.replace("'", '"')

                    parsed_bugs = json.loads(json_compatible)
                    bug_analysis_content = json.dumps(parsed_bugs, indent=2)
            except (json.JSONDecodeError, ValueError):
                pass  # Dict parsing failed, try fallback

        # If both structured parsing failed, use line-based fallback
        if not parsed_bugs:
            # Fallback: look for any content that looks like structured bug analysis
            # Look for lines containing "bug_" and "relevant_files_and_lines"
            lines = architect_response.split("\n")
            bug_lines = []
            in_bug_section = False
            for line in lines:
                if "bug_" in line and "relevant_files_and_lines" in line:
                    in_bug_section = True
                if in_bug_section:
                    bug_lines.append(line)
                    # Stop when we hit a non-bug line or section break
                    if line.strip() and not (
                        "bug_" in line
                        or "relevant_files_and_lines" in line
                        or "description" in line
                    ):
                        break
            if bug_lines:
                bug_analysis_content = "\n".join(bug_lines).strip()

        # If no structured bugs found, use the full response (fallback)
        if not bug_analysis_content:
            bug_analysis_content = "No bug analysis found"

    # Update context with architect's analysis (both structured and string formats)
    context = state.get("context", {}).copy()
    context["architect_analysis"] = bug_analysis_content
    context["final_bug_analysis_called"] = final_bug_analysis_called
    if parsed_bugs:
        context["parsed_bug_analysis"] = parsed_bugs

    # Add to messages for continuity
    messages = [
        HumanMessage(content=f"[LATS Iteration {state['iteration']}] Analyzing node {selected_id}"),
        AIMessage(content=architect_response),
    ]

    return {
        "messages": messages,
        "context": context,
    }


@traceable(run_type="llm", name="LATS_Expand")
def expand_node(state: TreeState) -> dict:
    """
    Generate concrete code fixes using the coder agent's expertise and prompt.

    Based on the architect's bug analysis and librarian research, generates
    actual code modifications as candidate nodes using the coder's specialized knowledge.
    """
    # Get num candidates from context
    num_candidates = state.get("context", {}).get("num_candidates", NUM_CANDIDATES)

    print(f"\n🔧 [EXPAND] Using coder expertise to generate {num_candidates} candidate fixes")

    selected_id = state["selected_node_id"]
    selected = get_node(state, selected_id)
    context = state.get("context", {})

    # The coder agent can modify any file it deems necessary

    # Get research context and architect's bug analysis
    # Prefer parsed structured format if available, fallback to string format
    parsed_bug_analysis = context.get("parsed_bug_analysis")
    if parsed_bug_analysis:
        # Format the parsed bugs into a readable string for the prompt
        architect_analysis = "\n".join(
            [
                f'{{bug_{bug_key}: {{relevant_files_and_lines: "{bug_info["relevant_files_and_lines"]}", description: "{bug_info["description"]}"}}}}'
                for bug_key, bug_info in parsed_bug_analysis.items()
            ]
        )
    else:
        architect_analysis = context.get("architect_analysis", "No architect analysis available")

    # Prepare prompt for the coder agent to generate fixes based on the architect's bug analysis
    coder_prompt = f"""

Generate {num_candidates} DISTINCT code fixes based on the architect's bug analysis provided below.

## Architect's Bug Analysis:
{architect_analysis}


## Instructions:
Based on the architect's bug analysis above, generate exactly {num_candidates} different fixes.

### 🔴 CRITICAL: LIVE TOOL ENVIRONMENT 🔴
**READ THIS CAREFULLY**: Although you are generating candidates for a Tree Search (LATS), **YOUR TOOLS ARE LIVE AND CONNECTED TO THE REAL REPO**.
* **YOU ARE NOT** in a hallucination-only simulation.
* **YOU MUST** use the `task` tool to call the `internal_librarian` to read the actual file contents.
* **DO NOT** guess the code based on line numbers. The line numbers are hints, not the source of truth.
* **FAILURE MODE**: If you generate an `old_string` that does not match the file exactly (character-for-character), the patch will fail. You cannot know the exact indentation or context without reading the file first using the `task` tool.

### Execution Steps:
1.  **CALL TOOLS FIRST**: Immediately use the `task` tool (with `internal_librarian`) to read the relevant lines in relevant files.
2.  **GENERATE CANDIDATES**: Once you have the *real* source code in your context window, generate the 3 distinct fixes.
3.  **FORMAT**: For each file modification, you must provide:
    * **old_string**: The EXACT unique block of existing code you found using the librarian (copy-paste it).
    * **new_string**: The new code to replace the old_string with.

If your initial old_string is not unique within the file, you MUST add more lines of context above or below until it becomes unique.

Format your response as JSON:
```json
{{
        "candidates": [
        {{
            "description": "Brief description of this candidate fix and which bugs it addresses",
            "modified_files": [
                {{
                "file_path": "path/to/file.py",
                    "old_string": "existing code block to replace\nwith enough context to be unique",
                    "new_string": "new code to replace the old_string with"
                }}
            ]
        }}
    ]
}}

IMPORTANT: For each modified file, provide the EXACT code block to replace (old_string) and what to replace it with (new_string). The old_string must be unique within the file - if it's not unique, add more surrounding lines until it becomes unique. Do not provide full file contents. Do not make any code up or make any assumptions about the code. Do not use any placeholder text. Use the exact real code block from the codebase for old_string. Use the task tool to call the internal_librarian subagent (via the task tool) to get the full content of the file if you need to.

### 🔄 IMMEDIATE FEEDBACK LOOP 🔄
**VALIDATION REQUIREMENT**: After generating your candidate solutions, you MUST validate them using the `fix_checker` subagent. This subagent call should be your FINAL ANSWER - do not return the JSON directly.

**Process**:
1. Generate all your candidate fixes as JSON (in the full format shown above)
2. **Extract all modified file blocks** from your candidates into a simplified format: a list of tuples [(file_path, old_string), ...] for ALL code blocks across all candidates
3. **Call the `fix_checker` subagent ONCE with this simplified list** as your final answer/tool call
4. The fix_checker subagent will validate all fixes and provide feedback
5. If validation fails, the system will ask you to revise - then repeat the process
6. Only return validated solutions that pass the fix_checker validation

**IMPORTANT**: Your final output should be a call to the `fix_checker` subagent with the simplified tuple list format, not the full JSON itself.
"""

    # Create a mini-state for the coder agent with plan.md context
    coder_state = {
        "messages": [HumanMessage(content=coder_prompt)],
        "context": context,
        "current_agent": "coder",
        "repo_path": state.get("repo_path"),  # Pass worktree path
    }

    try:
        # Use the full coder agent which has access to read_file and other tools
        result = coder_node(coder_state)
        response = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        print(f"   ✅ Coder agent used to generate fixes with plan.md context")
    except Exception as e:
        print(f"   ⚠️ Error generating fixes with coder agent: {e}")
        response = ""

    # Extract candidates from response
    candidates = []
    if response:
        # Parse JSON response
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                candidates_data = json.loads(json_match.group(1))
                candidates = candidates_data.get("candidates", [])
            else:
                # Try direct JSON parse - strip whitespace and extract JSON object
                stripped_response = response.strip()
                # Look for JSON object start/end
                if stripped_response.startswith("{") and stripped_response.endswith("}"):
                    candidates_data = json.loads(stripped_response)
                    candidates = candidates_data.get("candidates", [])
                else:
                    # Try to extract JSON from mixed content
                    json_start = stripped_response.find("{")
                    json_end = stripped_response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_content = stripped_response[json_start:json_end]
                        candidates_data = json.loads(json_content)
                        candidates = candidates_data.get("candidates", [])
        except json.JSONDecodeError as e:
            print(f"   ⚠️ Could not parse JSON response ({e}), extracting code blocks...")
            # Fallback: extract any code blocks and assign to a reasonable default file
            # For fallback, we assume the code block is new_string and create a minimal old_string
            code_blocks = re.findall(r"```python\s*(.*?)\s*```", response, re.DOTALL)
            for i, code in enumerate(code_blocks[:num_candidates]):
                candidates.append(
                    {
                        "description": f"Fix attempt {i + 1}",
                        "modified_files": [
                            {
                                "file_path": "python/sglang/srt/lora/layers.py",  # Default fallback file
                                "old_string": "# TODO: Replace this placeholder with the actual code to modify",  # Placeholder old_string
                                "new_string": code,
                            }
                        ],
                    }
                )

    # Create candidate nodes
    candidate_ids = []
    nodes = dict(state["nodes"])

    for i, candidate_data in enumerate(candidates[:num_candidates]):
        # Extract all modified files
        description = candidate_data.get("description", f"Candidate fix {i + 1}")
        modified_files = candidate_data.get("modified_files", [])

        # Build code changes from all modified files
        code_changes = {}
        for modified_file in modified_files:
            file_path = modified_file.get("file_path", "")
            old_string = modified_file.get("old_string", "")
            new_string = modified_file.get("new_string", "")

            # Only include files with valid changes
            if file_path and old_string and new_string:
                if file_path not in code_changes:
                    code_changes[file_path] = []
                code_changes[file_path].append({"old_string": old_string, "new_string": new_string})

        # Skip candidates with no valid modified files
        if not code_changes:
            continue

        candidate = Node(
            code_changes=code_changes,
            parent_id=selected_id,
            hypothesis=description,
            depth=selected.depth + 1,
        )

        # Add to parent's children
        selected.children_ids.append(candidate.id)

        nodes[candidate.id] = candidate.to_dict()
        candidate_ids.append(candidate.id)

        print(f"   Created candidate {candidate.id}: {candidate.hypothesis[:60]}...")

    # Update parent node
    nodes[selected_id] = selected.to_dict()

    return {
        "candidate_ids": candidate_ids,
        "nodes": nodes,
    }


@traceable(run_type="tool", name="LATS_Execute")
def execute_candidates(state: TreeState) -> dict:
    """
    Execute tests for all candidate nodes using the test runner agent pattern.
    """
    workspace_dir = os.environ.get("SGLANG_DIR", "")

    print(f"\n⚡ [EXECUTE] Testing {len(state['candidate_ids'])} candidates")

    nodes = dict(state["nodes"])
    best_score = 0.0
    best_id = state.get("best_node_id")

    for candidate_id in state["candidate_ids"]:
        candidate = Node.from_dict(nodes[candidate_id])

        # Restore to selected node's state before each test run
        print(f"   Restoring to selected node state before testing {candidate_id}...")
        restore_selected_node_files(workspace_dir)

        # Run test using our guardrailed tool
        print(f"   Testing candidate {candidate_id}...")
        test_output, return_code = run_and_test_code(candidate, workspace_dir)

        # Update candidate with test results
        candidate.test_output = test_output
        candidate.return_code = return_code

        # Get original code from root node for comparison
        root_node = Node.from_dict(nodes[state["root_id"]])
        original_code = root_node.code_changes

        # Use critic to evaluate and score the candidate
        score, critic_feedback = evaluate_candidate_with_critic(
            candidate, workspace_dir, original_code
        )

        # Store critic evaluation
        candidate.score = score
        candidate.agent_feedback = critic_feedback[:1000]  # Store detailed feedback

        nodes[candidate_id] = candidate.to_dict()

        print(f"   {candidate_id}: score={score:.3f}, return_code={return_code}")

        # Track best
        if score > best_score:
            best_score = score
            best_id = candidate_id

        # Check for success
        if score >= 1.0:
            print(f"   ✅ TEST PASSED! Node {candidate_id}")
            print(f"   📁 Keeping successful changes in worktree")
            return {
                "nodes": nodes,
                "status": "found",
                "best_node_id": candidate_id,
            }

    # If no perfect solution found, restore to selected node state for next iteration
    print("   Restoring to selected node state after testing all candidates...")
    restore_selected_node_files(workspace_dir)

    return {"nodes": nodes, "best_node_id": best_id}


@traceable(run_type="tool", name="LATS_Backprop")
def backprop_node(state: TreeState) -> dict:
    """
    Backpropagate scores from candidate nodes up to root.

    Updates visit counts and values for UCB calculation.
    """
    print(f"\n📊 [BACKPROP] Updating tree statistics")

    nodes = dict(state["nodes"])

    for candidate_id in state["candidate_ids"]:
        if candidate_id not in nodes:
            continue

        candidate = Node.from_dict(nodes[candidate_id])
        score = candidate.score

        # Walk up to root
        current_id = candidate_id
        while current_id is not None:
            if current_id not in nodes:
                break
            current = Node.from_dict(nodes[current_id])
            current.visits += 1
            current.value += score
            nodes[current_id] = current.to_dict()
            current_id = current.parent_id

        print(f"   Backpropagated score {score:.3f} from {candidate_id}")

    return {
        "nodes": nodes,
        "iteration": state["iteration"] + 1,
        "candidate_ids": [],  # Clear candidates for next iteration
    }


def should_continue(state: TreeState) -> Literal["continue", "end"]:
    """Determine if the search should continue."""
    status = state.get("status", "searching")

    if status == "found":
        print("\n✅ Solution found!")
        return "end"

    if status == "max_depth":
        print("\n⚠️ Maximum depth reached")
        return "end"

    # Get max iterations from context or use default
    max_iter = state.get("context", {}).get("max_iterations", MAX_ITERATIONS)

    if state["iteration"] >= max_iter:
        print(f"\n⚠️ Maximum iterations ({max_iter}) reached")
        return "end"

    return "continue"


# =============================================================================
# LIBRARIAN RESEARCH NODES
# =============================================================================


def research_internal(state: TreeState) -> dict:
    """
    Use the internal librarian to research the SGLang codebase.

    Searches for relevant patterns in:
    - LoRA implementations
    - MoE layer code
    - Similar logprob calculations
    - Test patterns and fixtures
    """
    print(f"\n📚 [INTERNAL LIBRARIAN] Researching SGLang codebase")

    selected_id = state.get("selected_node_id")
    selected = get_node(state, selected_id) if selected_id else None
    context = state.get("context", {})

    # Build research prompt based on current state
    with trace("Internal_Research_Prompt_Building", "tool") as rt:
        test_output = (
            selected.test_output[:1500]
            if selected and selected.test_output
            else "No test output yet"
        )
        current_score = selected.score if selected else 0.0

        research_prompt = f"""
[LATS RESEARCH - INTERNAL CODEBASE]

We're fixing LoRA MoE to match HuggingFace logprobs. Current score: {current_score:.3f}

## Recent Test Output:
```
{test_output}
```

## Previous Architect Analysis:
{context.get("architect_analysis", "No analysis yet")[:1000]}

## Research Tasks:
Please search the SGLang codebase for:

1. **LoRA Weight Application**: 
   - Find how LoRA weights are applied in other layers (not MoE)
   - Look for patterns in `python/sglang/srt/lora/`
   - Check weight loading and scaling logic

2. **MoE Expert Routing**:
   - Find how MoE experts are selected and routed
   - Look in `python/sglang/srt/layers/moe/`
   - Check FusedMoE kernel calls

3. **Logprob Calculations**:
   - Find similar logprob comparison code
   - Look at how HuggingFace comparisons are done
   - Check test fixtures and expected values

4. **Tensor Shape Patterns**:
   - Find common reshape/transpose patterns
   - Look for broadcasting conventions
   - Check dtype handling

Provide specific file paths, line numbers, and code snippets that could help fix the issue.
"""
        rt.end(outputs={"research_prompt": research_prompt[:500]})

    librarian_state = {
        "messages": [HumanMessage(content=research_prompt)],
        "context": context,
        "current_agent": "internal_librarian",
    }

    result = internal_librarian_node(librarian_state)

    # Extract research findings
    internal_research = ""
    if result.get("messages"):
        last_msg = result["messages"][-1]
        internal_research = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    print(f"   Found internal research insights")

    context = state.get("context", {}).copy()
    context["internal_research"] = internal_research

    return {
        "context": context,
        "messages": [AIMessage(content=f"[Internal Research]\n{internal_research[:1500]}...")],
    }


def research_external(state: TreeState) -> dict:
    """
    Use the external librarian to research online repositories.

    Searches for relevant patterns in:
    - vLLM repository (similar inference engine)
    - HuggingFace transformers (reference implementation)
    - Related GitHub issues and PRs
    - Documentation and papers
    """
    print(f"\n🌐 [EXTERNAL LIBRARIAN] Researching online repositories")

    selected_id = state.get("selected_node_id")
    selected = get_node(state, selected_id) if selected_id else None
    context = state.get("context", {})

    # Build research prompt based on current state
    test_output = (
        selected.test_output[:1000] if selected and selected.test_output else "No test output yet"
    )
    current_score = selected.score if selected else 0.0

    research_prompt = f"""
[LATS RESEARCH - EXTERNAL REPOSITORIES]

We're fixing LoRA MoE to match HuggingFace logprobs in SGLang. Current score: {current_score:.3f}

## Issue Summary:
{context.get("architect_analysis", "Fixing LoRA application to MoE expert layers")[:800]}

## Research Tasks:
Please search external repositories for:

1. **vLLM LoRA MoE Implementation** (sgl-project/sglang competitor):
   - Repository: vllm-project/vllm
   - Look for how vLLM handles LoRA with MoE models
   - Find their expert layer LoRA integration
   - Check for any similar issues/PRs

2. **HuggingFace Transformers Reference**:
   - Repository: huggingface/transformers
   - Look at MoE model implementations (Mixtral, etc.)
   - Find LoRA integration patterns
   - Check logprob calculation code

3. **PEFT Library** (HuggingFace LoRA):
   - Repository: huggingface/peft
   - Look for MoE-specific LoRA handling
   - Find weight application patterns
   - Check scaling factor implementations

4. **Related Issues/PRs**:
   - Search for "LoRA MoE" issues in SGLang, vLLM
   - Look for logprob mismatch discussions
   - Find shape mismatch fixes

5. **Documentation/Papers**:
   - LoRA paper implementation details
   - MoE architecture specifics
   - Numerical precision considerations

Provide GitHub links, code snippets, and specific insights that could help fix the logprob mismatch.
"""

    librarian_state = {
        "messages": [HumanMessage(content=research_prompt)],
        "context": context,
        "current_agent": "external_librarian",
    }

    result = external_librarian_node(librarian_state)

    # Extract research findings
    external_research = ""
    if result.get("messages"):
        last_msg = result["messages"][-1]
        external_research = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    print(f"   Found external research insights")

    context = state.get("context", {}).copy()
    context["external_research"] = external_research

    # Combine with internal research if available
    combined_research = f"""
## Internal Codebase Research:
{context.get("internal_research", "No internal research yet")[:2000]}

## External Repository Research:
{external_research[:2000]}
"""
    context["combined_research"] = combined_research

    return {
        "context": context,
        "messages": [AIMessage(content=f"[External Research]\n{external_research[:1500]}...")],
    }


# =============================================================================
# BUILD LANGGRAPH WITH AGENT INTEGRATION
# =============================================================================


def build_lats_graph(checkpointer=None):
    """Build the LATS workflow graph with agent integration and worktree support.

    Workflow:
    setup -> initial_test -> select -> analyze (architect first) -> [librarian research loop] -> expand -> execute -> backprop -> cleanup

    The architect is the first agent called and controls research handoffs to librarians.
    Librarians MUST hand back to architect, who can iterate research as needed before moving to expand phase.
    """

    # Create graph
    graph = StateGraph(TreeState)

    # Add worktree setup and cleanup nodes
    graph.add_node("setup", setup_worktree_lats)
    graph.add_node("cleanup", cleanup_worktree_lats)

    # Add LATS nodes (with agent integration)
    graph.add_node("initial_test", run_initial_test)
    graph.add_node("select", select_node)
    graph.add_node("analyze", analyze_with_architect)  # Architect first
    graph.add_node("research_internal", research_internal)
    graph.add_node("research_external", research_external)
    graph.add_node("expand", expand_node)
    graph.add_node("execute", execute_candidates)
    graph.add_node("backprop", backprop_node)

    # Set entry point to setup
    graph.set_entry_point("setup")

    # Connect setup to initial test, then to start of LATS workflow
    graph.add_edge("setup", "initial_test")
    graph.add_edge("initial_test", "select")

    # Add edges - Architect-controlled flow
    # select -> analyze (architect first) -> expand -> execute -> backprop
    graph.add_edge("select", "analyze")

    # Conditional edges from architect to allow research loop
    graph.add_conditional_edges(
        "analyze",
        should_research_more,  # Architect decides if more research needed
        {
            "research_internal": "research_internal",
            "research_external": "research_external",
            "expand": "expand",  # Ready to expand with current knowledge
        },
    )

    # Research nodes always return to architect for analysis
    def research_next_step(state: TreeState) -> str:
        """Research nodes always return to architect after completing their work."""
        return "analyze"

    graph.add_conditional_edges(
        "research_internal",
        research_next_step,
        {
            "research_internal": "research_internal",
            "research_external": "research_external",
            "analyze": "analyze",
        },
    )
    graph.add_conditional_edges(
        "research_external",
        research_next_step,
        {
            "research_internal": "research_internal",
            "research_external": "research_external",
            "analyze": "analyze",
        },
    )

    graph.add_edge("expand", "execute")
    graph.add_edge("execute", "backprop")

    # Conditional edge from backprop - go to cleanup instead of END
    graph.add_conditional_edges(
        "backprop",
        should_continue,
        {
            "continue": "select",
            "end": "cleanup",
        },
    )

    # Connect cleanup to final END
    graph.add_edge("cleanup", END)

    # Compile with checkpointer and store
    return graph.compile(checkpointer=checkpointer, store=store)


def should_research_more(state: TreeState) -> str:
    """
    Determine if architect wants more research or is ready to expand.

    Priority order:
    1. If final_bug_analysis tool was called -> expand (ready to implement)
    2. Look for handoff markers in architect's response:
       - [handoff:internal_librarian] -> research_internal
       - [handoff:external_librarian] -> research_external
    3. No handoff markers -> expand (ready to implement)

    Only the first handoff marker is processed. The architect can only hand off to one librarian per response.
    """
    context = state.get("context", {})

    # Check if final_bug_analysis tool was called (highest priority)
    if context.get("final_bug_analysis_called", False):
        print("🎯 Architect called final_bug_analysis tool - proceeding to expand phase")
        return "expand"

    architect_analysis = context.get("architect_analysis", "").lower()

    # Parse handoff markers from the architect's response
    import re

    handoff_pattern = r"\[handoff:(internal_librarian|external_librarian)\]"
    matches = re.findall(handoff_pattern, architect_analysis)

    if matches:
        # Only process the first handoff marker
        first_handoff = matches[0]
        print(f"🏗️ Architect requesting {first_handoff} research")
        return f"research_{first_handoff}"
    else:
        # No research handoffs or ready to implement
        print("🏗️ Architect ready to proceed to expand phase")
        return "expand"


def create_checkpointer():
    """Create checkpointer for persistence.

    Tries to use SQLite for persistence, falls back to MemorySaver.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        print("📁 Using SQLite persistence")
        return SqliteSaver(conn)
    except ImportError:
        print("⚠️ SQLite checkpointer not available, using in-memory persistence")
        print("   Install langgraph-checkpoint-sqlite for crash recovery")
        return MemorySaver()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


@traceable(run_type="chain", name="LATS_Run")
def run_lats(
    initial_code_path: str | None = None,
    thread_id: str | None = None,
    max_iterations: int | None = None,
    num_candidates: int | None = None,
) -> dict:
    """
    Run the LATS workflow with integrated agents.

    The workflow uses:
    - Internal Librarian for SGLang codebase research
    - External Librarian for vLLM/HuggingFace/online research
    - Debugger agent for instrumenting code and capturing debugging information
    - Architect agent for failure analysis and fix planning
    - Direct LLM for generating code fixes
    - Critic agent for evaluating results

    Args:
        initial_code_path: Path to the initial code file to fix
        thread_id: Thread ID for persistence (allows resumption)
        max_iterations: Maximum number of LATS iterations (default: MAX_ITERATIONS)
        num_candidates: Number of candidates per expansion (default: NUM_CANDIDATES)

    Returns:
        Final state with the best solution found
    """
    # Use provided values or defaults
    iterations_limit = max_iterations if max_iterations is not None else MAX_ITERATIONS
    candidates_count = num_candidates if num_candidates is not None else NUM_CANDIDATES

    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())

        # Create root node
    root = Node(
        code_changes={},  # Start with empty changes - files only added when modified
        hypothesis="Initial code state",
    )

    # Create initial state
    initial_state: TreeState = {
        "nodes": {root.id: root.to_dict()},
        "root_id": root.id,
        "selected_node_id": None,
        "candidate_ids": [],
        "iteration": 0,
        "status": "searching",
        "best_node_id": root.id,
        "input": "Fix LoRA MoE bugs",
        "messages": [],
        "context": {
            "target_files": "Any files in the codebase",
            "max_iterations": iterations_limit,
            "num_candidates": candidates_count,
        },
        "current_agent": "lats",
        "repo_path": None,  # Will be set by setup_worktree
    }

    # Create checkpointer for persistence
    checkpointer = create_checkpointer()

    # Build and run graph
    graph = build_lats_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}

    print(f"""
{"=" * 70}
  LATS: Language Agent Tree Search
  SGLang LoRA MoE Bug Fix (with Agent Integration)
{"=" * 70}

Scope:           Any files in codebase
Max iterations:  {iterations_limit}
Candidates/step: {candidates_count}
Thread ID:       {thread_id}

Agents Used:
  📚 Internal Librarian - SGLang codebase research
  🌐 External Librarian - vLLM/HuggingFace research
  🔧 Debugger           - Code instrumentation and debug data capture
  🏗️  Architect         - Failure analysis and fix planning
  🔧 LLM Direct         - Code fix generation
  🔍 Critic             - Result evaluation

Starting search...
""")

    # Run the graph
    final_state = graph.invoke(initial_state, config=config)

    # Print results
    print_results(final_state)

    return final_state


@traceable(run_type="chain", name="LATS_Resume")
def resume_lats(thread_id: str) -> dict:
    """Resume a previously interrupted LATS run."""
    checkpointer = create_checkpointer()
    graph = build_lats_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}

    print(f"Resuming LATS run with thread ID: {thread_id}")

    # Resume from last checkpoint
    final_state = graph.invoke(None, config=config)

    print_results(final_state)

    return final_state


def print_results(state: TreeState):
    """Print the results of a LATS run."""
    print(f"""
{"=" * 70}
  LATS Search Complete
{"=" * 70}

Status:      {state["status"]}
Iterations:  {state["iteration"]}
Nodes:       {len(state["nodes"])}
""")

    # Print best solution
    best_id = state.get("best_node_id")
    if best_id and best_id in state["nodes"]:
        best = Node.from_dict(state["nodes"][best_id])
        print(f"Best solution:")
        print(f"  Node ID:    {best.id}")
        print(f"  Score:      {best.score:.3f}")
        print(f"  Depth:      {best.depth}")
        print(f"  Hypothesis: {best.hypothesis}")

        if best.agent_feedback:
            print(f"  Feedback:   {best.agent_feedback[:200]}...")

        if best.score >= 1.0:
            print(f"\n✅ TEST PASSED!")
            print(f"\nFixed code saved. The PR is ready!")
        else:
            print(f"\n⚠️ Best score achieved: {best.score:.3f}")
            print("   Consider running with more iterations or reviewing the hypotheses.")

    # Print tree statistics
    nodes = [Node.from_dict(n) for n in state["nodes"].values()]
    depths = [n.depth for n in nodes]
    scores = [n.score for n in nodes]

    print(f"""
Tree Statistics:
  Max depth:     {max(depths)}
  Avg depth:     {sum(depths) / len(depths):.2f}
  Max score:     {max(scores):.3f}
  Avg score:     {sum(scores) / len(scores):.3f}
  Total visits:  {sum(n.visits for n in nodes)}
""")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LATS: Language Agent Tree Search for SGLang LoRA MoE Bug Fix"
    )
    parser.add_argument(
        "--workspace",
        "-w",
        help="Workspace directory containing the code and tests",
        default=os.environ.get("SGLANG_DIR", os.getcwd()),
    )
    parser.add_argument(
        "--code",
        "-c",
        help="Path to the initial code file to fix",
    )
    parser.add_argument(
        "--resume",
        "-r",
        help="Resume a previous run with the given thread ID",
    )
    parser.add_argument(
        "--thread-id",
        "-t",
        help="Thread ID for this run (for persistence)",
        default=None,
    )
    # Store defaults before parsing
    default_iterations = MAX_ITERATIONS
    default_candidates = NUM_CANDIDATES

    parser.add_argument(
        "--iterations",
        "-i",
        help="Maximum number of iterations",
        type=int,
        default=default_iterations,
    )
    parser.add_argument(
        "--candidates",
        "-n",
        help="Number of candidates per expansion",
        type=int,
        default=default_candidates,
    )

    args = parser.parse_args()

    # Initialize LangSmith client for trace flushing
    langsmith_client = Client()

    try:
        if args.resume:
            result = resume_lats(args.resume)
        else:
            result = run_lats(
                initial_code_path=args.code,
                workspace_dir=args.workspace,
                thread_id=args.thread_id,
                max_iterations=args.iterations,
                num_candidates=args.candidates,
            )

        # Exit with appropriate code
        best_id = result.get("best_node_id")
        if best_id and best_id in result["nodes"]:
            best = Node.from_dict(result["nodes"][best_id])
            if best.score >= 1.0:
                sys.exit(0)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Use --resume with your thread ID to continue.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure all traces are submitted before exiting
        try:
            import asyncio

            asyncio.run(langsmith_client.flush())
        except Exception as e:
            print(f"Warning: Failed to flush LangSmith traces: {e}")


if __name__ == "__main__":
    main()
