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
def google_search(query: str) -> str:
    """Performs a Google search and returns the top titles, snippets, and URLs."""
    # 1. Prepare the URL and Headers
    encoded_query = quote_plus(query)
    url = f"https://www.google.com/search?q={encoded_query}"
    
    # Google will block you or send a broken page without a real User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # 2. Target Google's result containers
        # Organic results are usually in divs with class 'g'
        for g in soup.select(".g"):
            title_tag = g.select_one("h3")
            link_tag = g.select_one("a")
            # Snippets are usually in a div with a specific class like 'VwiC3b' 
            # or simply the first div after the title.
            snippet_tag = g.select_one(".VwiC3b, .target-class-for-snippet") 

            if title_tag and link_tag:
                title = title_tag.get_text()
                link = link_tag.get("href")
                # Fallback if specific snippet class isn't found
                snippet = snippet_tag.get_text() if snippet_tag else "No snippet available."
                
                # Filter out internal google links
                if link.startswith("http"):
                    results.append(f"Title: {title}\nURL: {link}\nSummary: {snippet}")

        if not results:
            return "No results found. Google might be blocking the request or the structure changed."

        return "\n\n".join(results[:5])

    except Exception as e:
        return f"Search Error: {str(e)}"
        
@tool
def fetch_website_content(url: str) -> str:
    """Reads and extracts the main text content from any website URL provided."""
    try:
        response = requests.get(url, timeout=10)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=' ', strip=True)[:5000]
    except Exception as e:
        return f"Error reading website: {str(e)}"

search_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search_tool.name = "wikipedia_search"
search_tool.description = "Search Wikipedia for general knowledge, facts, and encyclopedia articles."

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
def get_current_time(query: str) -> str:
    """Useful to get the current date and time. Input can be anything (ignored)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def execute_python_code(code: str) -> str:
    """Executes python code dynamically in the console and returns the output string. Perfect for doing complex math or evaluating algorithms. Use standard python syntax."""
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
    return [fetch_website_content, search_tool, robust_internet_search, get_current_time, execute_python_code]
