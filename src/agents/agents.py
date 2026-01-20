"""Consolidated agents module - all agent node definitions in one file."""

import os
from pathlib import Path

import yaml
from langsmith import traceable

from src.agents.base import create_agent
from src.state import AgentState
from src.tools.langchain_tools import (
    ARCHITECT_TOOLS,
    CODER_TOOLS,
    CRITIC_TOOLS,
    EXTERNAL_LIBRARIAN_TOOLS,
    INTERNAL_LIBRARIAN_TOOLS,
    check_fixes,
)

# Load prompts from YAML file
PROMPTS_PATH = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
with open(PROMPTS_PATH) as f:
    PROMPTS = yaml.safe_load(f)


@traceable(run_type="llm", name="Architect_Agent")
def architect_node(state: AgentState) -> dict:
    """Run architect agent to plan and coordinate debugging using extracted test information."""
    print("\n🏗️ @Architect\n")  # Display handoff message
    system_prompt = PROMPTS["architect"]["system"]

    # Set worktree path for file tools
    repo_path = state.get("repo_path")
    if repo_path:
        os.environ["WORKTREE_REPO_PATH"] = repo_path

    # Create subagents that mirror architect capabilities (pure reasoning)
    # Create subagents that mirror architect capabilities (pure reasoning)
    architect_subagents = [
        {
            "name": "reasoning-assistant",
            "description": "Pure reasoning agent for complex analysis and planning tasks that require isolated thinking",
            "system_prompt": f"""You are a subagent of the Architect agent, specialized in pure reasoning for complex analysis and planning tasks.

{PROMPTS["architect"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on the specific task delegated to you
- Provide concise, actionable results
- Report back to your parent Architect agent with your findings
- Do not attempt to coordinate other agents or make handoff decisions""",
            "tools": ARCHITECT_TOOLS,  # Same tools (none)
        },
        {
            "name": "internal_librarian",
            "description": "Internal librarian subagent for researching the SGLang codebase patterns and similar implementations",
            "system_prompt": f"""You are a subagent of the Architect agent, specialized in internal codebase research for the SGLang project.

{PROMPTS["internal_librarian"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on researching the SGLang codebase for relevant patterns and implementations
- Search for LoRA, MoE, and similar code patterns in the codebase
- Provide specific file paths, line numbers, and code snippets that could help with the debugging
- Report back to your parent Architect agent with your research findings
- Do not attempt to coordinate other agents or make implementation decisions""",
            "tools": INTERNAL_LIBRARIAN_TOOLS,  # Same tools as internal librarian
        },
        {
            "name": "external_librarian",
            "description": "External librarian subagent for researching vLLM, HuggingFace, and other external implementations",
            "system_prompt": f"""You are a subagent of the Architect agent, specialized in external research for similar inference engines and libraries.

{PROMPTS["external_librarian"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on researching external repositories like vLLM, HuggingFace transformers, and PEFT
- Search for LoRA MoE implementations and logprob calculation patterns
- Provide GitHub links, code snippets, and specific insights from external sources
- Report back to your parent Architect agent with your research findings
- Do not attempt to coordinate other agents or make implementation decisions""",
            "tools": EXTERNAL_LIBRARIAN_TOOLS,  # Same tools as external librarian
        },
    ]

    agent = create_agent("architect", system_prompt, subagents=architect_subagents)

    # Add context about current debug state
    messages = state.get("messages", [])
    context = state.get("context", {})

    # If we have extracted test information from the test runner agent, add it to the messages
    extracted_info = context.get("extracted_test_info")
    if extracted_info:
        # Create a structured message with the extracted information
        test_info_message = f"""
[EXTRACTED TEST INFORMATION]

TEST STATUS: {extracted_info.get("test_status", "Unknown")}
CORE ISSUE: {extracted_info.get("core_issue", "Not specified")}
KEY ERROR DETAILS: {extracted_info.get("key_error_details", "Not specified")}
IMPACT ASSESSMENT: {extracted_info.get("impact_assessment", "Not specified")}
DEBUG RECOMMENDATIONS: {extracted_info.get("debug_recommendations", "Not specified")}
"""

        # Add this as a system message or prepend to the last message
        from langchain_core.messages import SystemMessage

        messages = messages + [SystemMessage(content=test_info_message)]
        print("Added extracted test information to architect context")

    result = agent.invoke({"messages": messages})
    return {"messages": result["messages"], "current_agent": "architect"}


@traceable(run_type="llm", name="Critic_Agent")
def critic_node(state: AgentState) -> dict:
    """Run critic agent for code review."""
    print("\n🔍 @Critic\n")  # Display handoff message
    system_prompt = PROMPTS["critic"]["system"]

    # Set worktree path for file tools
    repo_path = state.get("repo_path")
    if repo_path:
        os.environ["WORKTREE_REPO_PATH"] = repo_path

    # Create subagents that mirror critic capabilities (code review)
    critic_subagents = [
        {
            "name": "code-review-assistant",
            "description": "Specialized code reviewer for analyzing code changes, identifying bugs, and ensuring code quality",
            "system_prompt": f"""You are a subagent of the Critic agent, specialized in code review and quality analysis.

{PROMPTS["critic"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on the specific code review task delegated to you
- Provide detailed but concise review findings
- Report back to your parent Critic agent with your analysis
- Do not attempt to coordinate other agents or make implementation decisions""",
            "tools": CRITIC_TOOLS,  # Same tools: read_file, ruff_lint, github_get_pr_diff
        }
    ]

    agent = create_agent("critic", system_prompt, subagents=critic_subagents)
    result = agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"],
        "current_agent": "critic",
    }


@traceable(run_type="tool", name="Internal_Librarian_Agent")
def internal_librarian_node(state: AgentState) -> dict:
    """Run internal librarian agent for codebase search and analysis."""
    print("\n📚 @InternalLibrarian\n")  # Display handoff message
    system_prompt = PROMPTS["internal_librarian"]["system"]

    # Set worktree path for file tools
    repo_path = state.get("repo_path")
    if repo_path:
        os.environ["WORKTREE_REPO_PATH"] = repo_path
        # Also update the system prompt to use worktree path
        system_prompt = system_prompt.replace("{SGLANG_DIR}", repo_path)
    else:
        # Fallback to original behavior
        sglang_dir = os.environ.get("SGLANG_DIR")
        system_prompt = system_prompt.replace("{SGLANG_DIR}", sglang_dir)

    # Create subagents that mirror internal librarian capabilities (codebase analysis)
    internal_librarian_subagents = [
        {
            "name": "codebase-analysis-assistant",
            "description": "Specialized codebase analyst for searching, understanding, and tracing code execution flows",
            "system_prompt": f"""You are a subagent of the Internal Librarian agent, specialized in codebase search and analysis.

{system_prompt}

IMPORTANT: As a subagent, you must:
- Focus on the specific codebase analysis task delegated to you
- Search and analyze code thoroughly but concisely
- Report back to your parent Internal Librarian agent with your findings
- Do not attempt to coordinate other agents or make implementation decisions""",
            "tools": INTERNAL_LIBRARIAN_TOOLS,  # Same tools: run_command, git tools
        }
    ]

    agent = create_agent(
        "internal_librarian",
        system_prompt,
        subagents=internal_librarian_subagents,
    )
    result = agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"],
        "current_agent": "internal_librarian",
    }


@traceable(run_type="tool", name="External_Librarian_Agent")
def external_librarian_node(state: AgentState) -> dict:
    """Run external librarian agent for internet and external source search."""
    print("\n🌐 @ExternalLibrarian\n")  # Display handoff message
    system_prompt = PROMPTS["external_librarian"]["system"]

    # Set worktree path for file tools
    repo_path = state.get("repo_path")
    if repo_path:
        os.environ["WORKTREE_REPO_PATH"] = repo_path

    # Create subagents that mirror external librarian capabilities (external research)
    external_librarian_subagents = [
        {
            "name": "external-research-assistant",
            "description": "Specialized external researcher for searching GitHub, documentation, and research papers",
            "system_prompt": f"""You are a subagent of the External Librarian agent, specialized in external research and documentation.

{PROMPTS["external_librarian"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on the specific external research task delegated to you
- Search external sources and provide relevant findings
- Report back to your parent External Librarian agent with your research results
- Do not attempt to coordinate other agents or make implementation decisions""",
            "tools": EXTERNAL_LIBRARIAN_TOOLS,  # Same tools: github tools
        }
    ]

    agent = create_agent(
        "external_librarian",
        system_prompt,
        subagents=external_librarian_subagents,
        tools=EXTERNAL_LIBRARIAN_TOOLS,
    )
    result = agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"],
        "current_agent": "external_librarian",
    }


@traceable(run_type="llm", name="Coder_Agent")
def coder_node(state: AgentState) -> dict:
    """Run coder agent to implement code changes based on architect's plan."""
    print("\n💻 @Coder\n")  # Display handoff message
    system_prompt = PROMPTS["coder"]["system"]

    # Set worktree path for file tools
    repo_path = state.get("repo_path")
    if repo_path:
        os.environ["WORKTREE_REPO_PATH"] = repo_path

    # Create subagents that mirror coder capabilities (code implementation)
    coder_subagents = [
        {
            "name": "code-implementation-assistant",
            "description": "Specialized coder for implementing planned code changes and fixes",
            "system_prompt": f"""You are a subagent of the Coder agent, specialized in implementing code changes according to specifications.

{PROMPTS["coder"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on the specific implementation task delegated to you
- Implement code changes exactly as specified in the plan
- Report back to your parent Coder agent with your implementation
- Do not attempt to coordinate other agents or make design decisions""",
        },
        {
            "name": "fix_checker",
            "description": "Specialized subagent for validating that candidate fixes can be properly applied to files",
            "system_prompt": """You are a subagent of the Coder agent, specialized in validating fix candidates before they are submitted.

Your sole responsibility is to validate that the candidate fixes provided by the coder can be successfully applied to the codebase. You do this by checking if the old_string exists in each target file.

CRITICAL: You must validate ALL candidate fixes before the coder finishes their work. Do not allow invalid fixes to be submitted.

When called with candidate fixes:
1. Parse the JSON structure containing candidate solutions
2. For each candidate, check that the old_string exists in the specified file
3. Return detailed validation results showing which fixes are valid and which are not
4. If any fixes are invalid, clearly indicate what needs to be corrected

You have access to the check_fixes tool to perform this validation. Use it as your primary (and usually only) action.

**EXACT FORMAT FOR check_fixes TOOL CALL:**

The check_fixes tool expects a JSON string with exactly this structure:

```json
{
    "candidates": [
        {
            "description": "Brief description of this candidate fix and which bugs it addresses",
            "modified_files": [
                {
                    "file_path": "full/absolute/path/to/file.py",
                    "old_string": "existing code block to replace\\nwith enough context to be unique",
                    "new_string": "new code to replace the old_string with"
                }
            ]
        }
    ],
    "summary": "Brief summary of the 3 candidate approaches generated"
}
```

IMPORTANT: As a subagent, you must:
- Focus exclusively on fix validation - do not implement fixes or make code changes
- Report validation results back to your parent Coder agent clearly and comprehensively
- Do not attempt to coordinate other agents or make design decisions""",
            "tools": [check_fixes],  # Only has access to check_fixes tool
        },
        {
            "name": "internal_librarian",
            "description": "Internal librarian subagent for researching the SGLang codebase during implementation",
            "system_prompt": f"""You are a subagent of the Coder agent, specialized in researching the SGLang codebase to support implementation decisions.

{PROMPTS["internal_librarian"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on researching the SGLang codebase for relevant patterns and implementations that support the coder's implementation work
- Search for LoRA, MoE, and similar code patterns that help with understanding the current implementation
- Provide specific file paths, line numbers, and code snippets that help validate implementation approaches
- Report back to your parent Coder agent with your research findings
- Do not attempt to coordinate other agents or make implementation decisions yourself""",
            "tools": INTERNAL_LIBRARIAN_TOOLS,  # Same tools as internal librarian
        },
        {
            "name": "external_librarian",
            "description": "External librarian subagent for researching external implementations during coding",
            "system_prompt": f"""You are a subagent of the Coder agent, specialized in researching external repositories to support implementation decisions.

{PROMPTS["external_librarian"]["system"]}

IMPORTANT: As a subagent, you must:
- Focus on researching external repositories like vLLM, HuggingFace transformers, and PEFT to find similar implementation patterns
- Search for LoRA MoE implementations and patterns that can validate the coder's approach
- Provide GitHub links, code snippets, and specific insights from external sources that help with implementation decisions
- Report back to your parent Coder agent with your research findings
- Do not attempt to coordinate other agents or make implementation decisions yourself""",
            "tools": EXTERNAL_LIBRARIAN_TOOLS,  # Same tools as external librarian
        },
    ]

    agent = create_agent("coder", system_prompt, subagents=coder_subagents, tools=CODER_TOOLS)

    result = agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"],
        "current_agent": "coder",
    }
