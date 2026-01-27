"""Configuration settings for SGLang LoRA MoE Debug Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

AGENT_NAMES = [
    "architect",
    "critic",
    "test runner agent",
    "internal_librarian",
    "external_librarian",
]

# =============================================================================
# HARDCODED TASK: SGLang LoRA MoE Expert Layer Support
# =============================================================================
TASK_NAME = "SGLang LoRA MoE Debug"
TASK_DESCRIPTION = """
Fix the failing test for LoRA support on MoE (Mixture of Experts) expert layers in SGLang.
This PR adds Python paths and a Triton kernel for LoRA MoE support.
The goal is to make the test pass so the PR can be merged.
"""

# The specific test that must pass
TARGET_TEST_MODULE = "test_lora_hf_sgl_logprob_diff"
TARGET_TEST_CLASS = "TestLoRAHFSGLLogprobDifference"
TARGET_TEST_METHOD = "test_moe_lora_logprob_comparison_full"
TARGET_TEST_FULL = f"{TARGET_TEST_MODULE}.{TARGET_TEST_CLASS}.{TARGET_TEST_METHOD}"


# Key directories to search in SGLang codebase
SGLANG_KEY_PATHS = [
    "test/srt/test_lora_hf_sgl_logprob_diff.py",
    "python/sglang/srt/lora/",
    "python/sglang/srt/layers/moe/",
    "python/sglang/srt/layers/",
]

# Maximum debug iterations before giving up
MAX_DEBUG_ITERATIONS = 30

# =============================================================================
# LLM Configuration
# =============================================================================


@dataclass
class LLMConfig:
    """LLM connection settings for a single agent."""

    api_key: str = ""
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    provider: str = "openai"

    @classmethod
    def from_env(cls, agent_name: str) -> "LLMConfig":
        """Load LLM config for a specific agent from environment variables.

        Looks for AGENT_NAME_LLM_* vars first, falls back to LLM_* defaults.
        Example: ARCHITECT_LLM_API_KEY, ARCHITECT_LLM_BASE_URL
        """
        prefix = agent_name.upper()
        return cls(
            api_key=os.environ.get(f"{prefix}_LLM_API_KEY", os.environ.get("LLM_API_KEY", "")),
            base_url=os.environ.get(f"{prefix}_LLM_BASE_URL", os.environ.get("LLM_BASE_URL")),
            model=os.environ.get(f"{prefix}_LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-4o")),
            provider=os.environ.get(
                f"{prefix}_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "openai")
            ),
        )


@dataclass
class MCPSettings:
    """MCP server settings."""

    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    brave_api_key: str = field(default_factory=lambda: os.environ.get("BRAVE_API_KEY", ""))
    filesystem_root: str = field(default_factory=lambda: os.environ.get("FILESYSTEM_ROOT", "/"))


@dataclass
class Settings:
    """Application settings."""

    workspace_dir: Path = field(default_factory=Path.cwd)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    agents: dict[str, LLMConfig] = field(default_factory=dict)

    # Task-specific settings
    task_name: str = TASK_NAME
    target_test: str = TARGET_TEST_FULL
    test_base_path: str = field(
        default_factory=lambda: os.environ.get("TEST_BASE_PATH", TARGET_TEST_FULL)
    )
    max_iterations: int = MAX_DEBUG_ITERATIONS

    # Test execution settings
    pytest_mock_mode: bool = (
        True  # When True, return fake test output instead of running real pytest
    )

    # GPU device settings for test execution
    cuda_visible_devices: str = "4"  # CUDA_VISIBLE_DEVICES environment variable for tests

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            workspace_dir=Path(os.environ.get("WORKSPACE_DIR", Path.cwd())),
            mcp=MCPSettings(),
            agents={name: LLMConfig.from_env(name) for name in AGENT_NAMES},
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
        )

    def get_agent_llm(self, agent_name: str) -> LLMConfig:
        """Get LLM config for a specific agent."""
        return self.agents.get(agent_name, LLMConfig.from_env(agent_name))


settings = Settings.from_env()
