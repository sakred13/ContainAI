import bs4
import requests
import io
import contextlib
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

@tool
def fetch_website_content(url: str) -> str:
    """Reads and extracts the main text content from a URL. ONLY call this tool when the user has explicitly provided a URL as a reference in their message. Do NOT use this to look up URLs you found yourself."""
    try:
        response = requests.get(url, timeout=10)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=' ', strip=True)[:2500]
    except Exception as e:
        return f"Error reading website: {str(e)}"

search_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search_tool.name = "wikipedia_search"
search_tool.description = "Search Wikipedia for well-known, widely recognized subjects such as historical events, famous public figures, scientific concepts, countries, organizations, or established topics. Do NOT use this for private individuals, regular people, or niche topics unlikely to have a Wikipedia article."

@tool
def robust_internet_search(query: str) -> str:
    """Searches the internet for live data, news, people, and current events. Returns article titles, snippets, and URLs."""
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"}
    try:
        response = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.find_all('div', class_='result'):
            title = result.find('a', class_='result__a')
            snippet = result.find('a', class_='result__snippet')
            if title and snippet:
                href = title.get('href', '')
                results.append(f"Title: {title.text}\nURL/Source: {href}\nSummary: {snippet.text}")
        return "\n\n".join(results[:5]) if results else "No internet results found for this query."
    except Exception as e:
        return f"Internet Search Error: {str(e)}"


@tool
def deep_internet_search(query: str) -> str:
    """Searches the internet AND reads the full content of the top results. Use this for detailed research, fact-checking, or when snippets are not enough."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"}

    # --- Step 1: DuckDuckGo search ---
    search_url = "https://html.duckduckgo.com/html/"
    try:
        resp = requests.post(search_url, data={"q": query}, headers=headers, timeout=10)
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return f"Search Error: {str(e)}"

    snippets, urls = [], []
    for result in soup.find_all('div', class_='result'):
        title_tag = result.find('a', class_='result__a')
        snippet_tag = result.find('a', class_='result__snippet')
        if title_tag and snippet_tag:
            href = title_tag.get('href', '')
            # DDG HTML sometimes wraps URLs in its redirect — decode if needed
            if 'uddg=' in href:
                from urllib.parse import unquote, urlparse, parse_qs
                parsed = parse_qs(urlparse(href).query)
                href = unquote(parsed.get('uddg', [href])[0])
            snippets.append(f"Title: {title_tag.text.strip()}\nURL: {href}\nSnippet: {snippet_tag.text.strip()}")
            if href.startswith('http'):
                urls.append(href)

    if not snippets:
        return "No search results found."

    output = "=== Search Results ===\n" + "\n\n".join(snippets[:3])

    # --- Step 2: Fetch full content from top 2 URLs ---
    for i, page_url in enumerate(urls[:2]):
        try:
            page_resp = requests.get(page_url, headers=headers, timeout=10)
            page_soup = bs4.BeautifulSoup(page_resp.text, "html.parser")
            # Remove script/style noise
            for tag in page_soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            page_text = page_soup.get_text(separator=' ', strip=True)[:1500]
            output += f"\n\n=== Full Content (Result {i + 1}): {page_url} ===\n{page_text}"
        except Exception as fetch_err:
            output += f"\n\n=== Result {i + 1}: Could not fetch full content ({fetch_err}) ==="

    return output

@tool
def get_current_time(query: str = "") -> str:
    """Returns the current date and time. ONLY use this when the user explicitly asks for the time or date. NEVER use this for greetings or casual conversation."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    return [fetch_website_content, search_tool, robust_internet_search, deep_internet_search, get_current_time, execute_python_code]
