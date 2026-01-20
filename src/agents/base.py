"""Base agent with LLM connectivity and deep agent configuration."""

import json
import os
import re
import uuid
from collections.abc import Iterator
from typing import Any, Callable

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel

from config.settings import settings
from src.memory import store
from src.tools.search_limiter import search_limiter
from src.tools.tool_logger import tool_logger


class RemoveFilesystemPromptMiddleware(AgentMiddleware):
    """Middleware that removes filesystem tools and prompts from architect agent."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Remove filesystem tools and their prompts from the architect agent."""

        # Remove filesystem tools from the tools list
        filesystem_tool_names = {
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
        }
        filtered_tools = [
            tool
            for tool in request.tools
            if (tool.name if hasattr(tool, "name") else tool.get("name"))
            not in filesystem_tool_names
        ]
        request = request.override(tools=filtered_tools)

        # Remove filesystem tool prompts from the system message
        if request.system_message and hasattr(request.system_message, "content"):
            content = request.system_message.content
            # Remove the "## Filesystem Tools..." section and everything after it
            filesystem_section_pattern = r"\n\n## Filesystem Tools.*$"
            cleaned_content = re.sub(filesystem_section_pattern, "", content, flags=re.DOTALL)
            # Also remove any "## Execute Tool..." section that might follow
            execute_section_pattern = r"\n\n## Execute Tool.*$"
            cleaned_content = re.sub(execute_section_pattern, "", cleaned_content, flags=re.DOTALL)

            # Create a new system message with cleaned content
            from langchain_core.messages import SystemMessage

            cleaned_system_message = SystemMessage(content=cleaned_content.strip())
            request = request.override(system_message=cleaned_system_message)

        return handler(request)


def _parse_tool_calls_from_content(content: str) -> tuple[str, list[dict]]:
    """Parse <tool_call> tags from content and return cleaned content + tool calls.

    Args:
        content: The raw content string that may contain <tool_call> tags

    Returns:
        Tuple of (cleaned_content, tool_calls_list)
    """
    tool_calls = []

    # Find all <tool_call>...</tool_call> blocks
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        try:
            # Parse the JSON inside the tool_call tag
            tool_data = json.loads(match.strip())
            tool_name = tool_data.get("name", "")
            tool_args = tool_data.get("arguments", {})

            # Generate a unique ID for the tool call
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

            tool_calls.append(
                {
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": tool_args,
                }
            )
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse tool call JSON: {e}")
            continue

    # Remove the tool_call tags from content to get cleaned content
    # Also remove </think> and <think> tags which some models use for reasoning
    cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL)
    # Clean up thinking tags
    cleaned_content = re.sub(r"</think>\s*", "", cleaned_content)
    cleaned_content = cleaned_content.strip()

    return cleaned_content, tool_calls


def _process_ai_message(message: AIMessage) -> AIMessage:
    """Process an AIMessage to extract tool calls from content if needed."""
    # Only process AIMessage without tool_calls but with content containing <tool_call>
    if not message.tool_calls and message.content and "<tool_call>" in str(message.content):
        cleaned_content, tool_calls = _parse_tool_calls_from_content(str(message.content))

        if tool_calls:
            print(f"✨ Parsed {len(tool_calls)} tool call(s) from content")
            return AIMessage(
                content=cleaned_content,
                tool_calls=tool_calls,
                id=message.id,
                response_metadata=message.response_metadata,
            )

    return message


class ToolCallParsingRunnable(Runnable):
    """A Runnable wrapper that parses tool calls from <tool_call> tags.

    This wraps a RunnableBinding (result of bind_tools) and post-processes
    the output to extract tool calls from content.
    """

    def __init__(self, bound: RunnableBinding | Runnable):
        self.bound = bound

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Invoke the bound runnable and process the response."""
        result = self.bound.invoke(input, config=config, **kwargs)

        if isinstance(result, AIMessage):
            return _process_ai_message(result)
        return result

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Async invoke the bound runnable and process the response."""
        result = await self.bound.ainvoke(input, config=config, **kwargs)

        if isinstance(result, AIMessage):
            return _process_ai_message(result)
        return result

    def bind_tools(self, tools: list, **kwargs: Any) -> "ToolCallParsingRunnable":
        """Bind additional tools."""
        bound = self.bound.bind_tools(tools, **kwargs)
        return ToolCallParsingRunnable(bound=bound)

    @property
    def InputType(self) -> Any:
        """Return the input type of the bound runnable."""
        return self.bound.InputType

    @property
    def OutputType(self) -> Any:
        """Return the output type of the bound runnable."""
        return self.bound.OutputType

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the bound runnable."""
        return getattr(self.bound, name)


class ToolCallParsingWrapper(BaseChatModel):
    """Wrapper that parses tool calls from <tool_call> tags in content.

    Some models (e.g., Qwen on sglang) return tool calls embedded in the content
    field using <tool_call> tags instead of the standard tool_calls array.
    This wrapper intercepts responses and parses those tags into proper tool calls.
    """

    llm: BaseChatModel
    """The underlying LLM to wrap."""

    @property
    def _llm_type(self) -> str:
        return f"tool_call_parsing_wrapper({self.llm._llm_type})"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"wrapped_llm": self.llm._identifying_params}

    def _process_response(self, response: ChatResult) -> ChatResult:
        """Process a response to extract tool calls from content if needed."""
        processed_generations = []

        for generation in response.generations:
            message = generation.message

            # Only process AIMessage without tool_calls but with content
            if isinstance(message, AIMessage):
                new_message = _process_ai_message(message)
                if new_message is not message:
                    processed_generations.append(
                        ChatGeneration(
                            message=new_message, generation_info=generation.generation_info
                        )
                    )
                else:
                    processed_generations.append(generation)
            else:
                processed_generations.append(generation)

        return ChatResult(generations=processed_generations, llm_output=response.llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response and parse tool calls from content if needed."""
        response = self.llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._process_response(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response and parse tool calls from content if needed."""
        response = await self.llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._process_response(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Stream responses - note: tool call parsing happens after full response."""
        # For streaming, we pass through directly since we can't parse partial tool calls
        # The tool calls will be parsed from the accumulated content in the final message
        yield from self.llm._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: list, **kwargs: Any) -> ToolCallParsingRunnable:
        """Bind tools to the underlying LLM and return a parsing-enabled runnable.

        Returns a ToolCallParsingRunnable which wraps the RunnableBinding from bind_tools
        and ensures tool calls are parsed from content.
        """
        bound = self.llm.bind_tools(tools, **kwargs)
        return ToolCallParsingRunnable(bound=bound)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Pass through to underlying LLM."""
        return self.llm.with_structured_output(schema, **kwargs)


def get_llm(agent_name: str):
    """Create LLM client for a specific agent."""
    cfg = settings.get_agent_llm(agent_name)
    kwargs = {"model": cfg.model}

    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
        print(f"Using base URL: {cfg.base_url}")

    if cfg.provider == "anthropic":
        return ChatAnthropic(api_key=cfg.api_key, **kwargs)

    # Wrap OpenAI-compatible clients with tool call parsing
    # This handles sglang servers that return tool calls in <tool_call> tags
    base_llm = ChatOpenAI(api_key=cfg.api_key or "dummy", **kwargs)
    return ToolCallParsingWrapper(llm=base_llm)


def create_agent(
    agent_name: str,
    system_prompt: str,
    subagents: list | None = None,
    tools: list | None = None,
    middleware: list | None = None,
    output_schema: type[BaseModel] | None = None,
    subagent_output_schemas: dict[str, type[BaseModel]] | None = None,
):
    """Create a DeepAgent with built-in standard tools and memory capabilities.

    Deep agents include:
    - Long-term memory via LangGraph Store (StoreBackend)
    - Planning via write_todos tool
    - Context management via filesystem tools
    - Subagent spawning via task tool
    - Search tool limiting (1 search call per program run)
    - Standard tools provided via middleware
    - SGLang constrained decoding via output schemas (LangChain structured output)

    Args:
        agent_name: Name of the agent (used for LLM configuration)
        system_prompt: System prompt for the agent
        subagents: Optional list of subagents
        tools: Optional list of additional tools
        middleware: Optional list of additional middleware to apply after defaults
        output_schema: Optional Pydantic model for constrained decoding of final output
        subagent_output_schemas: Optional dict mapping subagent names to their output schemas
    """
    # Wrap main agent tools if provided
    wrapped_tools = []
    if tools:
        for tool in tools:
            tool = search_limiter.wrap_tool(tool)
            tool = tool_logger.wrap_tool(tool)
            wrapped_tools.append(tool)

    # Wrap subagent tools if any subagents are provided
    wrapped_subagents = []
    if subagents:
        for subagent in subagents:
            if "tools" in subagent and subagent["tools"]:
                wrapped_subagent_tools = []
                for tool in subagent["tools"]:
                    tool = search_limiter.wrap_tool(tool)
                    tool = tool_logger.wrap_tool(tool)
                    wrapped_subagent_tools.append(tool)
                subagent = dict(subagent)  # Create a copy
                subagent["tools"] = wrapped_subagent_tools
            wrapped_subagents.append(subagent)

    # Configure FilesystemBackend to work with real files in the worktree
    worktree_path = os.getenv("WORKTREE_REPO_PATH", ".")
    backend = FilesystemBackend(
        root_dir=worktree_path,
        virtual_mode=True,  # Enable virtual path mapping for security
    )

    # Prepare middleware stack
    final_middleware = middleware or []

    # For architect agent, add middleware to remove filesystem prompts
    if agent_name == "architect" or agent_name == "coder":
        final_middleware.append(RemoveFilesystemPromptMiddleware())

    # Get the base LLM
    llm = get_llm(agent_name)

    # Prepare response_format for structured output (following LangChain guidelines)
    response_format = None
    if output_schema is not None:
        print(
            f"🔒 Enabling SGLang ProviderStrategy constrained decoding for {agent_name} with schema: {output_schema.__name__}"
        )
        # SGLang supports native JSON schema constraints, so explicitly use ProviderStrategy
        # This ensures high reliability and strict validation via provider-native structured output
        response_format = ProviderStrategy(schema=output_schema, strict=True)

    # Apply output schemas to subagents if specified
    if subagent_output_schemas and wrapped_subagents:
        for subagent in wrapped_subagents:
            subagent_name = subagent.get("name")
            if subagent_name and subagent_name in subagent_output_schemas:
                schema = subagent_output_schemas[subagent_name]
                print(
                    f"🔒 Enabling SGLang ProviderStrategy constrained decoding for subagent {subagent_name} with schema: {schema.__name__}"
                )
                # SGLang supports native JSON schema constraints, so explicitly use ProviderStrategy
                subagent["response_format"] = ProviderStrategy(schema=schema, strict=True)
                # Keep output_schema for backward compatibility
                subagent["output_schema"] = schema

    # Prepare kwargs for create_deep_agent
    agent_kwargs = {
        "model": llm,
        "system_prompt": system_prompt,
        "tools": wrapped_tools,
        "subagents": wrapped_subagents,
        "middleware": final_middleware,
        "store": store,  # Enable long-term memory across threads
        "backend": backend,  # Use real filesystem instead of virtual state backend
    }

    # Add response_format if structured output is requested (following LangChain guidelines)
    if response_format is not None:
        agent_kwargs["response_format"] = response_format

    return create_deep_agent(**agent_kwargs)


# Task delegation in LangGraph works via HumanMessage - no need for dynamic system prompts
# Task instructions are passed through the description parameter when calling the task tool
