"""
Helper utilities for the LATS agent.

This module provides:
- Code diff utilities
- Tree visualization
- Checkpoint management
- Advanced scoring heuristics
- Agent integration utilities
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Import Node from main module
from lats_agent import CHECKPOINT_DB, Node, TreeState


# =============================================================================
# CODE DIFF UTILITIES
# =============================================================================


def generate_diff(old_code: str, new_code: str, filename: str = "file.py") -> str:
    """Generate a unified diff between two code strings."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )

    return "".join(diff)


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a unified diff to original code.

    Note: This is a simplified implementation. For production use,
    consider using the `patch` command or a proper diff library.
    """
    # This is a placeholder - in practice, you'd use subprocess to call `patch`
    # or use a library like `unidiff`
    raise NotImplementedError("Use full code snapshots instead of diffs")


def highlight_changes(old_code: str, new_code: str) -> str:
    """Create a side-by-side comparison highlighting changes."""
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    output = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in old_lines[i1:i2]:
                output.append(f"  {line}")
        elif tag == "replace":
            for line in old_lines[i1:i2]:
                output.append(f"- {line}")
            for line in new_lines[j1:j2]:
                output.append(f"+ {line}")
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                output.append(f"- {line}")
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                output.append(f"+ {line}")

    return "\n".join(output)


# =============================================================================
# TREE VISUALIZATION
# =============================================================================


def visualize_tree(state: TreeState, max_depth: int | None = None) -> str:
    """Generate an ASCII visualization of the search tree."""
    nodes = {nid: Node.from_dict(n) for nid, n in state["nodes"].items()}
    root_id = state["root_id"]

    def _visualize(
        node_id: str, prefix: str = "", is_last: bool = True, depth: int = 0
    ) -> list[str]:
        if max_depth is not None and depth > max_depth:
            return []

        node = nodes[node_id]

        # Node representation
        connector = "└── " if is_last else "├── "
        status = "✅" if node.score >= 1.0 else "🔴" if node.score == 0 else "🟡"
        node_repr = f"{prefix}{connector}{status} {node.id} (s={node.score:.2f}, v={node.visits}, d={node.depth})"

        lines = [node_repr]

        # Process children
        children_ids = node.children_ids
        for i, child_id in enumerate(children_ids):
            is_last_child = i == len(children_ids) - 1
            extension = "    " if is_last else "│   "
            lines.extend(_visualize(child_id, prefix + extension, is_last_child, depth + 1))

        return lines

    lines = _visualize(root_id, "", True, 0)
    return "\n".join(lines)


def get_tree_stats(state: TreeState) -> dict:
    """Calculate statistics about the search tree."""
    nodes = [Node.from_dict(n) for n in state["nodes"].values()]

    if not nodes:
        return {}

    depths = [n.depth for n in nodes]
    scores = [n.score for n in nodes]
    visits = [n.visits for n in nodes]

    return {
        "total_nodes": len(nodes),
        "max_depth": max(depths),
        "avg_depth": sum(depths) / len(depths),
        "max_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "total_visits": sum(visits),
        "leaf_nodes": sum(1 for n in nodes if n.is_leaf()),
        "terminal_nodes": sum(1 for n in nodes if n.is_terminal()),
        "passed_nodes": sum(1 for n in nodes if n.score >= 1.0),
    }


def get_best_path(state: TreeState) -> list[Node]:
    """Get the path from root to the best scoring node."""
    nodes = {nid: Node.from_dict(n) for nid, n in state["nodes"].items()}

    # Find best node
    best_node = max(nodes.values(), key=lambda n: n.score)

    # Trace path back to root
    path = []
    current = best_node
    while current is not None:
        path.append(current)
        if current.parent_id:
            current = nodes.get(current.parent_id)
        else:
            current = None

    return list(reversed(path))


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================


def list_checkpoints() -> list[dict]:
    """List all saved checkpoints."""
    if not os.path.exists(CHECKPOINT_DB):
        return []

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT thread_id, thread_ts, parent_ts
            FROM checkpoints
            ORDER BY thread_ts DESC
        """)

        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append(
                {
                    "thread_id": row[0],
                    "timestamp": row[1],
                    "parent_ts": row[2],
                }
            )

        return checkpoints
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def delete_checkpoint(thread_id: str) -> bool:
    """Delete a checkpoint by thread ID."""
    if not os.path.exists(CHECKPOINT_DB):
        return False

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def inspect_checkpoint(thread_id: str) -> dict | None:
    """Inspect a specific checkpoint by thread ID."""
    if not os.path.exists(CHECKPOINT_DB):
        return None

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT thread_id, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY rowid DESC
            LIMIT 1
        """,
            (thread_id,),
        )

        row = cursor.fetchone()
        if row:
            return {
                "thread_id": row[0],
                "checkpoint_id": row[1],
                "parent_checkpoint_id": row[2],
                "checkpoint_data": row[3],
                "metadata": row[4],
            }
        return None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def export_state(state: TreeState, filepath: str):
    """Export the current state to a JSON file."""
    # Convert state to JSON-serializable format
    export_data = {
        "nodes": state["nodes"],
        "root_id": state["root_id"],
        "selected_node_id": state.get("selected_node_id"),
        "candidate_ids": state.get("candidate_ids", []),
        "iteration": state["iteration"],
        "status": state["status"],
        "best_node_id": state.get("best_node_id"),
        "input": state.get("input", ""),
        "exported_at": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)


def import_state(filepath: str) -> TreeState:
    """Import state from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    return TreeState(
        nodes=data["nodes"],
        root_id=data["root_id"],
        selected_node_id=data.get("selected_node_id"),
        candidate_ids=data.get("candidate_ids", []),
        iteration=data["iteration"],
        status=data["status"],
        best_node_id=data.get("best_node_id"),
        input=data.get("input", ""),
        messages=[],
    )


# =============================================================================
# ADVANCED SCORING HEURISTICS
# =============================================================================


@dataclass
class TestMetrics:
    """Extracted metrics from test output."""

    prefill_max_diff: float | None = None
    prefill_mean_diff: float | None = None
    decode_max_diff: float | None = None
    decode_mean_diff: float | None = None
    prefill_pass_ratio: float | None = None
    decode_pass_ratio: float | None = None
    string_match_ratio: float | None = None

    def overall_score(self, threshold: float = 0.1) -> float:
        """Calculate an overall score based on metrics."""
        scores = []

        # Prefill scores
        if self.prefill_max_diff is not None:
            # Score inversely proportional to difference
            prefill_score = max(0, 1 - (self.prefill_max_diff / threshold))
            scores.append(prefill_score * 0.3)

        if self.prefill_pass_ratio is not None:
            scores.append(self.prefill_pass_ratio * 0.2)

        # Decode scores
        if self.decode_max_diff is not None:
            decode_score = max(0, 1 - (self.decode_max_diff / threshold))
            scores.append(decode_score * 0.3)

        if self.decode_pass_ratio is not None:
            scores.append(self.decode_pass_ratio * 0.1)

        # String match
        if self.string_match_ratio is not None:
            scores.append(self.string_match_ratio * 0.1)

        return sum(scores) if scores else 0.0


def extract_test_metrics(test_output: str) -> TestMetrics:
    """Extract numerical metrics from test output."""
    metrics = TestMetrics()

    # Extract prefill metrics
    prefill_match = re.search(
        r"Prefill.*?Max difference:\s*([\d.e+-]+).*?Mean difference:\s*([\d.e+-]+)",
        test_output,
        re.DOTALL | re.IGNORECASE,
    )
    if prefill_match:
        try:
            metrics.prefill_max_diff = float(prefill_match.group(1))
            metrics.prefill_mean_diff = float(prefill_match.group(2))
        except ValueError:
            pass

    # Extract decode metrics
    decode_match = re.search(
        r"Decode.*?Max difference:\s*([\d.e+-]+).*?Mean difference:\s*([\d.e+-]+)",
        test_output,
        re.DOTALL | re.IGNORECASE,
    )
    if decode_match:
        try:
            metrics.decode_max_diff = float(decode_match.group(1))
            metrics.decode_mean_diff = float(decode_match.group(2))
        except ValueError:
            pass

    # Extract pass/fail ratios
    prefill_ratio = re.search(r"Prefill logprob:\s*(\d+)/(\d+)", test_output)
    if prefill_ratio:
        try:
            passed = int(prefill_ratio.group(1))
            total = int(prefill_ratio.group(2))
            metrics.prefill_pass_ratio = passed / total if total > 0 else 0
        except ValueError:
            pass

    decode_ratio = re.search(r"Decode logprob:\s*(\d+)/(\d+)", test_output)
    if decode_ratio:
        try:
            passed = int(decode_ratio.group(1))
            total = int(decode_ratio.group(2))
            metrics.decode_pass_ratio = passed / total if total > 0 else 0
        except ValueError:
            pass

    # Extract string match ratio
    string_ratio = re.search(r"Output strings:\s*(\d+)/(\d+)", test_output)
    if string_ratio:
        try:
            matched = int(string_ratio.group(1))
            total = int(string_ratio.group(2))
            metrics.string_match_ratio = matched / total if total > 0 else 0
        except ValueError:
            pass

    return metrics


def calculate_advanced_score(test_output: str, return_code: int) -> float:
    """
    Calculate an advanced score using extracted metrics.

    This provides more granular scoring than simple heuristics.
    """
    # First check for critical failures
    output_lower = test_output.lower()

    if "test passed" in output_lower or return_code == 0:
        return 1.0

    if return_code == -11 or "segmentation fault" in output_lower:
        return 0.0

    if "permission denied" in output_lower:
        return 0.1

    if "syntaxerror" in output_lower or "importerror" in output_lower:
        return 0.15

    # Extract metrics for more nuanced scoring
    metrics = extract_test_metrics(test_output)
    base_score = metrics.overall_score()

    # Apply penalties/bonuses
    if "typeerror" in output_lower:
        base_score *= 0.7

    if "shape" in output_lower and "mismatch" in output_lower:
        base_score *= 0.8

    # Ensure score is in valid range
    return max(0.1, min(0.95, base_score))


# =============================================================================
# CODE ANALYSIS UTILITIES
# =============================================================================


def extract_function_signatures(code: str) -> list[str]:
    """Extract function signatures from Python code."""
    pattern = r"^(\s*)(async\s+)?def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?:"
    matches = re.findall(pattern, code, re.MULTILINE)
    return [m[2] for m in matches]


def extract_class_names(code: str) -> list[str]:
    """Extract class names from Python code."""
    pattern = r"^class\s+(\w+)(?:\([^)]*\))?:"
    return re.findall(pattern, code, re.MULTILINE)


def find_tensor_operations(code: str) -> list[str]:
    """Find tensor-related operations in code."""
    patterns = [
        r"\.reshape\([^)]+\)",
        r"\.view\([^)]+\)",
        r"\.permute\([^)]+\)",
        r"\.transpose\([^)]+\)",
        r"\.unsqueeze\([^)]+\)",
        r"\.squeeze\([^)]+\)",
        r"torch\.matmul\([^)]+\)",
        r"torch\.einsum\([^)]+\)",
        r"@\s*\w+",  # Matrix multiplication operator
        r"\.contiguous\(\)",
        r"\.to\([^)]+\)",
    ]

    operations = []
    for pattern in patterns:
        operations.extend(re.findall(pattern, code))

    return operations


def analyze_code_complexity(code: str) -> dict:
    """Analyze code complexity metrics."""
    lines = code.splitlines()

    # Count different line types
    code_lines = 0
    comment_lines = 0
    blank_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith("#"):
            comment_lines += 1
        else:
            code_lines += 1

    return {
        "total_lines": len(lines),
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "functions": len(extract_function_signatures(code)),
        "classes": len(extract_class_names(code)),
        "tensor_operations": len(find_tensor_operations(code)),
    }


# =============================================================================
# CLI UTILITIES
# =============================================================================


def print_tree(state: TreeState):
    """Print the search tree to console."""
    print("\n" + "=" * 60)
    print("  Search Tree Visualization")
    print("=" * 60)
    print()
    print(visualize_tree(state))
    print()


def print_stats(state: TreeState):
    """Print tree statistics to console."""
    stats = get_tree_stats(state)

    print("\n" + "=" * 60)
    print("  Tree Statistics")
    print("=" * 60)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()


def print_best_path(state: TreeState):
    """Print the best path through the tree."""
    path = get_best_path(state)

    print("\n" + "=" * 60)
    print("  Best Path to Solution")
    print("=" * 60)

    for i, node in enumerate(path):
        print(f"\n  Step {i}: {node.id}")
        print(f"    Score: {node.score:.3f}")
        print(f"    Hypothesis: {node.hypothesis[:80]}...")
    print()


# =============================================================================
# AGENT INTEGRATION UTILITIES
# =============================================================================


def create_agent_context(state: TreeState) -> dict:
    """
    Create a context dictionary for agent interactions.

    This provides relevant information from the LATS tree to agents.
    """
    context = state.get("context", {}).copy()

    # Add current iteration info
    context["lats_iteration"] = state["iteration"]
    context["lats_status"] = state["status"]

    # Add best node info if available
    best_id = state.get("best_node_id")
    if best_id and best_id in state["nodes"]:
        best = Node.from_dict(state["nodes"][best_id])
        context["best_score"] = best.score
        context["best_hypothesis"] = best.hypothesis
        context["best_test_output"] = best.test_output[:1000] if best.test_output else ""

    # Add tree statistics
    stats = get_tree_stats(state)
    context["tree_stats"] = stats

    return context


def summarize_agent_interactions(state: TreeState) -> str:
    """
    Summarize all agent interactions from the LATS run.

    This is useful for debugging and understanding the search process.
    """
    summary = []
    summary.append("=" * 60)
    summary.append("  LATS Agent Interaction Summary")
    summary.append("=" * 60)

    # Summarize each node's agent feedback
    nodes = [Node.from_dict(n) for n in state["nodes"].values()]
    nodes.sort(key=lambda n: n.depth)

    for node in nodes:
        if node.agent_feedback:
            summary.append(f"\nNode {node.id} (depth={node.depth}, score={node.score:.3f}):")
            summary.append(f"  Hypothesis: {node.hypothesis[:100]}...")
            summary.append(f"  Agent Feedback: {node.agent_feedback[:200]}...")

    # Add context information
    context = state.get("context", {})

    if context.get("architect_analysis"):
        summary.append("\n--- Last Architect Analysis ---")
        summary.append(context["architect_analysis"][:500])

    if context.get("critic_feedback"):
        summary.append("\n--- Last Critic Feedback ---")
        summary.append(context["critic_feedback"][:500])

    if context.get("internal_research"):
        summary.append("\n--- Internal Librarian Research (SGLang) ---")
        summary.append(context["internal_research"][:500])

    if context.get("external_research"):
        summary.append("\n--- External Librarian Research (vLLM/HuggingFace) ---")
        summary.append(context["external_research"][:500])

    return "\n".join(summary)


def create_hypothesis_prompt(
    node: Node,
    hypothesis_type: str,
    context: dict | None = None,
) -> str:
    """
    Create a specialized prompt for a specific hypothesis type.

    This helps guide the LLM to generate targeted fixes.
    """
    prompts = {
        "tensor_broadcasting": """
Focus on tensor broadcasting issues:
- Check torch.unsqueeze() and torch.expand() calls
- Verify dimension ordering in matrix multiplications
- Ensure batch dimensions are handled correctly
""",
        "dtype_conversion": """
Focus on dtype conversion issues:
- Check .to(dtype) calls and their placement
- Verify intermediate computation dtypes
- Look for autocast region problems
""",
        "lora_scaling": """
Focus on LoRA scaling factor:
- Verify alpha / rank calculation
- Check where scaling is applied
- Consider numerical precision issues
""",
        "expert_routing": """
Focus on MoE expert routing:
- Check expert index selection logic
- Verify token-to-expert assignment
- Look for sparse vs dense routing issues
""",
        "contiguity": """
Focus on tensor contiguity:
- Add .contiguous() calls where needed
- Check view/reshape operations
- Verify memory layout requirements
""",
        "accumulation_order": """
Focus on operation order:
- Check lora_B @ lora_A multiplication order
- Verify addition to base output timing
- Look for reduction operation issues
""",
    }

    base_prompt = prompts.get(hypothesis_type, "")

    if not base_prompt:
        return ""

    prompt = f"""
## Specific Focus: {hypothesis_type.replace("_", " ").title()}

{base_prompt}

## Current Code Context:
- Modified Files: {", ".join(node.code_snapshot.keys()) if node.code_snapshot else "None"}
- Current Score: {node.score:.3f}

## Previous Error:
{node.test_output[:500] if node.test_output else "No previous error"}

Generate a fix that specifically addresses {hypothesis_type.replace("_", " ")} issues.
"""

    return prompt


def extract_code_from_agent_response(response: str) -> str | None:
    """
    Extract code from an agent's response.

    Handles various response formats (JSON, markdown code blocks, etc.)
    """
    # Try JSON format first
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            if "code" in data:
                return data["code"]
            if "fixes" in data and data["fixes"]:
                return data["fixes"][0].get("code")
    except json.JSONDecodeError:
        pass

    # Try Python code block
    python_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if python_match:
        return python_match.group(1)

    # Try generic code block
    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1)

    return None


def validate_code_fix(code: str) -> tuple[bool, str]:
    """
    Validate that a proposed code fix is syntactically valid.

    Returns (is_valid, error_message).
    """
    if not code or len(code) < 50:
        return False, "Code is too short or empty"

    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Compilation error: {e}"


def compare_code_changes(original: str, modified: str) -> dict:
    """
    Compare original and modified code to understand the changes.

    Returns a dictionary with change statistics.
    """
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()

    # Use SequenceMatcher to find changes
    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)

    added = 0
    removed = 0
    changed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)

    # Find changed function signatures
    original_funcs = set(extract_function_signatures(original))
    modified_funcs = set(extract_function_signatures(modified))

    new_funcs = modified_funcs - original_funcs
    removed_funcs = original_funcs - modified_funcs

    return {
        "lines_added": added,
        "lines_removed": removed,
        "lines_changed": changed,
        "total_changes": added + removed + changed,
        "new_functions": list(new_funcs),
        "removed_functions": list(removed_funcs),
        "original_lines": len(original_lines),
        "modified_lines": len(modified_lines),
    }


def get_improvement_suggestions(state: TreeState) -> list[str]:
    """
    Analyze the search progress and suggest improvements.

    This is useful when the search is stuck.
    """
    suggestions = []

    stats = get_tree_stats(state)
    nodes = [Node.from_dict(n) for n in state["nodes"].values()]

    # Check if scores are improving
    root = get_node(state, state["root_id"])
    if nodes:
        max_score = max(n.score for n in nodes)
        if max_score <= root.score:
            suggestions.append(
                "❌ No improvement over initial state - try different hypothesis types"
            )

    # Check for common failure patterns
    failure_patterns = {}
    for node in nodes:
        if node.test_output:
            output_lower = node.test_output.lower()
            if "shape" in output_lower:
                failure_patterns["shape_mismatch"] = failure_patterns.get("shape_mismatch", 0) + 1
            if "dtype" in output_lower or "type" in output_lower:
                failure_patterns["dtype_issues"] = failure_patterns.get("dtype_issues", 0) + 1
            if "broadcast" in output_lower:
                failure_patterns["broadcasting"] = failure_patterns.get("broadcasting", 0) + 1

    if failure_patterns:
        most_common = max(failure_patterns.items(), key=lambda x: x[1])
        suggestions.append(f"🔍 Most common issue: {most_common[0]} ({most_common[1]} occurrences)")

    # Check tree balance
    depths = [n.depth for n in nodes]
    if depths and max(depths) > 3 * len(nodes) ** 0.5:
        suggestions.append("⚠️ Tree is too deep - consider more exploration (lower UCB weight)")

    # Check for untried areas
    leaf_nodes = [n for n in nodes if n.is_leaf() and n.visits == 0]
    if len(leaf_nodes) > 5:
        suggestions.append(f"💡 {len(leaf_nodes)} unexplored candidates - consider evaluating them")

    return suggestions


def get_node(state: TreeState, node_id: str) -> Node:
    """Helper to get a node from state."""
    return Node.from_dict(state["nodes"][node_id])


# =============================================================================
# DEBUGGING UTILITIES
# =============================================================================


def extract_debug_values(test_output: str) -> dict:
    """
    Extract debugging values from test output that contains debug instrumentation.

    Looks for DEBUG[...] patterns and extracts key information.
    """
    debug_values = {
        "tensor_shapes": [],
        "tensor_values": [],
        "memory_usage": [],
        "expert_routing": [],
        "lora_weights": [],
        "warnings": [],
    }

    import re

    # Extract tensor shapes
    shape_pattern = r"DEBUG\[([^\]]+)\]:\s*(\w+)\s*shape=\[([^\]]+)\]"
    for match in re.finditer(shape_pattern, test_output):
        step, name, shape = match.groups()
        debug_values["tensor_shapes"].append({"step": step, "name": name, "shape": shape})

    # Extract tensor values
    value_pattern = r"DEBUG\[([^\]]+)\]:\s*(\w+)\s*->\s*(.+)"
    for match in re.finditer(value_pattern, test_output):
        step, name, values = match.groups()
        debug_values["tensor_values"].append({"step": step, "name": name, "values": values.strip()})

    # Extract memory usage
    memory_pattern = r"DEBUG\[([^\]]+)\]:\s*(GPU|CPU)\s*memory[^:]*:\s*(.+)"
    for match in re.finditer(memory_pattern, test_output, re.IGNORECASE):
        step, mem_type, info = match.groups()
        debug_values["memory_usage"].append({"step": step, "type": mem_type, "info": info.strip()})

    # Extract expert routing
    expert_pattern = r"DEBUG\[([^\]]+)\]:\s*Expert\s*(weights|indices)[^:]*:\s*(.+)"
    for match in re.finditer(expert_pattern, test_output):
        step, comp_type, info = match.groups()
        debug_values["expert_routing"].append(
            {"step": step, "component": comp_type, "info": info.strip()}
        )

    # Extract LoRA weights
    lora_pattern = r"DEBUG\[([^\]]+)\]:\s*LoRA\s*(.+)"
    for match in re.finditer(lora_pattern, test_output):
        step, info = match.groups()
        debug_values["lora_weights"].append({"step": step, "info": info.strip()})

    # Extract warnings
    warning_pattern = r"DEBUG\[([^\]]+)\]:\s*WARNING[^:]*:\s*(.+)"
    for match in re.finditer(warning_pattern, test_output):
        step, warning = match.groups()
        debug_values["warnings"].append({"step": step, "warning": warning.strip()})

    return debug_values


def analyze_debug_session(debug_values: dict) -> dict:
    """
    Analyze a debugging session and identify potential issues.

    Returns a dictionary with analysis results and recommendations.
    """
    analysis = {
        "issues_found": [],
        "recommendations": [],
        "severity_score": 0.0,  # 0.0 = no issues, 1.0 = critical issues
    }

    # Check for shape mismatches
    shapes_by_step = {}
    for shape_info in debug_values["tensor_shapes"]:
        step = shape_info["step"]
        if step not in shapes_by_step:
            shapes_by_step[step] = []
        shapes_by_step[step].append(shape_info)

    # Look for inconsistent shapes in the same step
    for step, shapes in shapes_by_step.items():
        if len(shapes) > 1:
            # Check for potential broadcasting issues
            shape_dims = [len(shape["shape"].split(",")) for shape in shapes]
            if len(set(shape_dims)) > 1:
                analysis["issues_found"].append(
                    {
                        "type": "shape_dimension_mismatch",
                        "step": step,
                        "description": f"Different tensor dimensions in step {step}",
                        "severity": "medium",
                    }
                )

    # Check for warnings
    if debug_values["warnings"]:
        analysis["issues_found"].extend(
            [
                {
                    "type": "warning",
                    "step": w["step"],
                    "description": w["warning"],
                    "severity": "medium",
                }
                for w in debug_values["warnings"]
            ]
        )

        # Increase severity if critical warnings found
        critical_warnings = ["NaN", "Inf", "segmentation", "memory"]
        for warning in debug_values["warnings"]:
            if any(cw.lower() in warning["warning"].lower() for cw in critical_warnings):
                analysis["severity_score"] = max(analysis["severity_score"], 0.8)

    # Check memory usage patterns
    memory_readings = debug_values["memory_usage"]
    if len(memory_readings) > 1:
        # Look for memory leaks (increasing usage)
        memory_values = []
        for mem in memory_readings:
            # Extract numeric value from memory info
            import re

            num_match = re.search(r"(\d+(?:\.\d+)?)", mem["info"])
            if num_match:
                try:
                    memory_values.append(float(num_match.group(1)))
                except ValueError:
                    pass

        if len(memory_values) > 1:
            growth = memory_values[-1] - memory_values[0]
            if growth > 100:  # 100MB growth
                analysis["issues_found"].append(
                    {
                        "type": "memory_leak",
                        "description": f"Memory growth of {growth:.1f} MB detected",
                        "severity": "high",
                    }
                )
                analysis["severity_score"] = max(analysis["severity_score"], 0.7)

    # Check expert routing consistency
    expert_info = debug_values["expert_routing"]
    if expert_info:
        # Look for zero weights or uniform selection
        for info in expert_info:
            if "min=0.0" in info["info"] and "max=0.0" in info["info"]:
                analysis["issues_found"].append(
                    {
                        "type": "zero_expert_weights",
                        "step": info["step"],
                        "description": "Zero expert weights detected - routing ineffective",
                        "severity": "high",
                    }
                )
                analysis["severity_score"] = max(analysis["severity_score"], 0.9)

    # Generate recommendations based on issues
    if analysis["issues_found"]:
        analysis["recommendations"].append("Review tensor shapes for broadcasting compatibility")
        analysis["recommendations"].append("Check for NaN/Inf values in tensor operations")
        analysis["recommendations"].append("Monitor memory usage for potential leaks")
        analysis["recommendations"].append("Verify expert routing weights are non-zero")

        if analysis["severity_score"] > 0.7:
            analysis["recommendations"].append("Consider reverting to a simpler implementation")
    else:
        analysis["recommendations"].append("No critical issues detected in debug output")

    return analysis


def format_debug_summary(debug_values: dict, analysis: dict) -> str:
    """
    Format debugging information into a readable summary report.
    """
    summary = []
    summary.append("=" * 60)
    summary.append("  DEBUG SESSION SUMMARY")
    summary.append("=" * 60)

    # Summary statistics
    summary.append(f"\nCaptured Data Points:")
    summary.append(f"  Tensor shapes: {len(debug_values['tensor_shapes'])}")
    summary.append(f"  Tensor values: {len(debug_values['tensor_values'])}")
    summary.append(f"  Memory readings: {len(debug_values['memory_usage'])}")
    summary.append(f"  Expert routing: {len(debug_values['expert_routing'])}")
    summary.append(f"  LoRA weights: {len(debug_values['lora_weights'])}")
    summary.append(f"  Warnings: {len(debug_values['warnings'])}")

    # Issues found
    if analysis["issues_found"]:
        summary.append(f"\nIssues Detected ({len(analysis['issues_found'])}):")
        for issue in analysis["issues_found"]:
            severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}
            emoji = severity_emoji.get(issue["severity"], "❓")
            summary.append(f"  {emoji} {issue['type']}: {issue['description']}")
            if "step" in issue:
                summary.append(f"      Step: {issue['step']}")
    else:
        summary.append("\n✅ No issues detected in debug output")

    # Recommendations
    summary.append(f"\nRecommendations:")
    for rec in analysis["recommendations"]:
        summary.append(f"  • {rec}")

    summary.append(
        f"\nSeverity Score: {analysis['severity_score']:.2f} (0.0 = no issues, 1.0 = critical)"
    )

    return "\n".join(summary)
