import subprocess
import requests
from langchain_core.tools import tool

@tool
def get_pypi_details(package_name: str) -> str:
    """
    Fetches official metadata for a PyPI package including its latest version, 
    summary, and project URLs (documentation, homepage).
    Use this to find where the official manual for a library is located.
    """
    try:
        # Run pypi information <package_name>
        result = subprocess.run(
            ["pypi", "information", package_name],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            return result.stdout
        return f"Error fetching PyPI info for {package_name}: {result.stderr}"
    except Exception as e:
        return f"Exception while fetching PyPI info: {str(e)}"

@tool
def get_local_pydoc(module_path: str) -> str:
    """
    Fetches the local API documentation and function signatures for an INSTALLED 
    module or class. Very high fidelity.
    Example: 'pandas.DataFrame.head' or 'requests.get'.
    Only works if the library is already available in the environment.
    """
    try:
        # Run python -m pydoc <module_path>
        result = subprocess.run(
            ["python", "-m", "pydoc", module_path],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            # Pydoc can be verbose, return first 5000 chars
            return result.stdout[:5000]
        return f"Error fetching pydoc for {module_path}: {result.stderr}"
    except Exception as e:
        return f"Exception while fetching pydoc: {str(e)}"

@tool
def search_cheat_sheet(library: str, topic: str = "") -> str:
    """
    Fetches high-quality code snippets and common usage patterns for a library 
    from cheat.sh.
    'topic' is optional (e.g., 'post' for requests or 'read_csv' for pandas).
    """
    try:
        query = f"{library}/{topic}" if topic else library
        url = f"https://cheat.sh/python/{query}"
        # We use a user-agent to get plain text without ANSI escape codes if possible, 
        # though cheat.sh is usually smart.
        headers = {"User-Agent": "curl/7.81.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        return f"Error fetching cheat sheet for {query}: Status {response.status_code}"
    except Exception as e:
        return f"Exception while fetching cheat sheet: {str(e)}"

def get_librarian_tools():
    """Returns the set of specialized tools for the Librarian Agent."""
    return [get_pypi_details, get_local_pydoc, search_cheat_sheet]
