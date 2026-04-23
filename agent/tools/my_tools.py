import io
import contextlib
from datetime import datetime
from langchain_core.tools import tool

# Import tools from other modules
from .internet_tools import get_internet_tools
from .context_tools import get_context_tools
from .librarian_tools import get_librarian_tools


@tool
def get_current_time(timezone_name: str = "") -> str:
    """Returns the current date and time. Accepts a standard timezone string (e.g., 'America/New_York', 'US/Pacific', 'UTC'). Defaults to system local time if none provided."""
    try:
        if timezone_name:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(timezone_name)
            return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Invalid timezone '{timezone_name}'. Use formats like 'America/New_York' or 'US/Pacific'. Error: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """Executes python code dynamically in the console and returns the output string. Perfect for doing complex math or evaluating algorithms. Use standard python syntax."""
    # Strip markdown code fences if the LLM wrapped the code
    if "```" in code:
        parts = code.split("```")
        # Extract the middle part if it's a full block, else just take the last split
        code = parts[1] if len(parts) >= 3 else parts[-1]
        if code.startswith("python"):
            code = code[6:]
        code = code.strip()

    # Remove comment lines before running the code
    code = "\n".join([line for line in code.splitlines() if not line.lstrip().startswith("#")])
    
    output = io.StringIO()
    # Isolated sandbox — variables are discarded after each execution
    sandbox = {"__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(output):
            exec(code, sandbox)
        return output.getvalue() or "Code executed successfully with no output returned. Did you forget to print()?"
    except Exception as e:
        return f"Python Execution Error: {str(e)}"

def get_all_tools():
    """Combines tools from all separate tool files."""
    internet_tools = get_internet_tools()
    context_tools = get_context_tools()
    librarian_tools = get_librarian_tools()
    
    # Other tools defined in this file
    other_tools = [get_current_time, execute_python_code]
    
    return internet_tools + context_tools + librarian_tools + other_tools

