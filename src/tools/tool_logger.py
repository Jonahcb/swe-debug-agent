"""Tool logging wrapper for debugging agent framework."""

import sys
import time
from typing import Any
from langchain_core.tools import BaseTool


class ToolLogger:
    """Logs all tool calls to terminal for debugging."""

    def wrap_tool(self, tool) -> Any:
        """Wrap a tool to log all invocations by monkey-patching its methods."""
        # Handle BaseTool objects
        if hasattr(tool, '_run'):
            original_run = tool._run
            original_arun = getattr(tool, '_arun', None)

            def logged_run(*args, config=None, **kwargs):
                """Log the tool call and then execute."""
                # Log the tool call start
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                print(f"\n[{timestamp}] 🔧 Tool Call: {tool.name}", file=sys.stderr)

                # Log arguments if any (skip config parameter which is internal)
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
                if args or filtered_kwargs:
                    if args:
                        print(f"  Args: {args}", file=sys.stderr)
                    if filtered_kwargs:
                        print(f"  Kwargs: {filtered_kwargs}", file=sys.stderr)

                # Record start time
                start_time = time.time()

                try:
                    # Execute the tool with config parameter
                    result = original_run(*args, config=config, **kwargs)

                    # Calculate duration
                    duration = time.time() - start_time

                    # Log the result (truncate if too long, skip content for github tools)
                    result_str = str(result)
                    if "github" in tool.name.lower():
                        print(f"  ✅ Result ({duration:.2f}s): [GitHub content - not displayed]", file=sys.stderr)
                    else:
                        if len(result_str) > 500:
                            result_str = result_str[:500] + "... [truncated]"
                        print(f"  ✅ Result ({duration:.2f}s): {result_str}", file=sys.stderr)

                    return result

                except Exception as e:
                    # Log errors
                    duration = time.time() - start_time
                    print(f"  ❌ Error ({duration:.2f}s): {e}", file=sys.stderr)
                    raise

            async def logged_arun(*args, config=None, **kwargs):
                """Log the async tool call and then execute."""
                if original_arun is None:
                    # Fallback to sync version if no async version exists
                    return logged_run(*args, config=config, **kwargs)

                # Log the tool call start
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                print(f"\n[{timestamp}] 🔧 Tool Call: {tool.name}", file=sys.stderr)

                # Log arguments if any (skip config parameter which is internal)
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
                if args or filtered_kwargs:
                    if args:
                        print(f"  Args: {args}", file=sys.stderr)
                    if filtered_kwargs:
                        print(f"  Kwargs: {filtered_kwargs}", file=sys.stderr)

                # Record start time
                start_time = time.time()

                try:
                    # Execute the tool
                    result = await original_arun(*args, config=config, **kwargs)

                    # Calculate duration
                    duration = time.time() - start_time

                    # Log the result (truncate if too long, skip content for github tools)
                    result_str = str(result)
                    if "github" in tool.name.lower():
                        print(f"  ✅ Result ({duration:.2f}s): [GitHub content - not displayed]", file=sys.stderr)
                    else:
                        if len(result_str) > 500:
                            result_str = result_str[:500] + "... [truncated]"
                        print(f"  ✅ Result ({duration:.2f}s): {result_str}", file=sys.stderr)

                    return result

                except Exception as e:
                    # Log errors
                    duration = time.time() - start_time
                    print(f"  ❌ Error ({duration:.2f}s): {e}", file=sys.stderr)
                    raise

            # Monkey patch the methods
            tool._run = logged_run
            if original_arun is not None:
                tool._arun = logged_arun

            return tool
        else:
            # Handle raw functions - wrap the function directly
            original_func = tool

            def logged_func(*args, **kwargs):
                """Log the function call and then execute."""
                # Log the function call start
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                func_name = getattr(original_func, '__name__', str(original_func))
                print(f"\n[{timestamp}] 🔧 Function Call: {func_name}", file=sys.stderr)

                # Log arguments
                if args or kwargs:
                    if args:
                        print(f"  Args: {args}", file=sys.stderr)
                    if kwargs:
                        print(f"  Kwargs: {kwargs}", file=sys.stderr)

                # Record start time
                start_time = time.time()

                try:
                    # Execute the function
                    result = original_func(*args, **kwargs)

                    # Calculate duration
                    duration = time.time() - start_time

                    # Log the result (truncate if too long, skip content for github tools)
                    result_str = str(result)
                    func_name = getattr(original_func, '__name__', str(original_func))
                    if "github" in func_name.lower():
                        print(f"  ✅ Result ({duration:.2f}s): [GitHub content - not displayed]", file=sys.stderr)
                    else:
                        if len(result_str) > 500:
                            result_str = result_str[:500] + "... [truncated]"
                        print(f"  ✅ Result ({duration:.2f}s): {result_str}", file=sys.stderr)

                    return result

                except Exception as e:
                    # Log errors
                    duration = time.time() - start_time
                    print(f"  ❌ Error ({duration:.2f}s): {e}", file=sys.stderr)
                    raise

            return logged_func


# Global logger instance
tool_logger = ToolLogger()