import requests
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from urllib.parse import unquote, urlparse, parse_qs
import os
from .context_tools import save_page_markdown, compact_page_markdown, summarize_search_results
from common.scraper_utils import fetch_html, get_soup, extract_technical_contracts, clean_search_snippets

@tool
def fetch_website_content(url: str) -> str:
    """Reads and extracts the main text content from a URL. ONLY call this tool when the user has explicitly provided a URL as a reference in their message."""
    html = fetch_html(url)
    soup = get_soup(html)
    if not soup:
        return "Failed to fetch content."
    return soup.get_text(separator=' ', strip=True)[:2500]

@tool
def scrape_technical_docs(url: str) -> str:
    """
    SPECIALIZED TOOL FOR RESEARCH: Scrapes a URL specifically for Technical Contracts, Function Signatures, 
    and Syntax Examples. Use this tool when you need to understand the exact API of a library (methods, parameters, return types).
    It strips away all conversational and advertising content.
    """
    print(f"[TOOL:SCRAPE] Targeting documentation: {url}", flush=True)
    html = fetch_html(url)
    soup = get_soup(html)
    if not soup:
        print(f"[TOOL:SCRAPE] FAILED to fetch HTML for {url}", flush=True)
        return f"Failed to reach {url}"
    
    result = extract_technical_contracts(soup)
    print(f"[TOOL:SCRAPE] Extracted {len(result)} chars of technical content.", flush=True)
    return result

search_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search_tool.name = "wikipedia_search"
search_tool.description = "Search Wikipedia for well-known, widely recognized subjects such as historical events, famous public figures, scientific concepts, countries, organizations, or established topics. Do NOT use this for private individuals, regular people, or niche topics unlikely to have a Wikipedia article."

def deep_internet_search(query: str, conversation_id: str = "default") -> str:
    """Internal logic for deep internet search. Not a direct tool."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"}
    search_url = "https://html.duckduckgo.com/html/"
    
    # --- Step 1: DuckDuckGo search ---
    try:
        resp = requests.post(search_url, data={"q": query}, headers=headers, timeout=10)
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        
        # Save the search results index page (pre-cleaned markdown)
        index_file = save_page_markdown(search_url, conversation_id, f"Search_Index_{query[:20]}", content=resp.text)
        saved_files = [index_file] if index_file else []

        snippets, urls_with_titles = [], []
        for result in soup.find_all('div', class_='result'):
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            if title_tag and snippet_tag:
                href = title_tag.get('href', '')
                title_text = title_tag.text.strip()
                # DDG HTML sometimes wraps URLs in its redirect — decode if needed
                if 'uddg=' in href:
                    parsed = parse_qs(urlparse(href).query)
                    href = unquote(parsed.get('uddg', [href])[0])
                
                snippets.append(f"Title: {title_text}\nURL: {href}\nSnippet: {snippet_tag.text.strip()}")
                if href.startswith('http') and "google.com" not in href:
                    urls_with_titles.append((href, title_text))
                    
        if not snippets or resp.status_code == 202:
            raise Exception(f"DuckDuckGo blocked (status {resp.status_code}) or zero results returned.")
            
    except Exception as scrape_err:
        print(f"[RESEARCH][{conversation_id}] HTML scrape failed ({scrape_err}). Falling back to ddgs library...", flush=True)
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                snippets, urls_with_titles = [], []
                saved_files = [] # Start fresh
                for r in results:
                    title_text = r.get("title", "")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    if href.startswith('http') and "google.com" not in href:
                        urls_with_titles.append((href, title_text))
                        snippets.append(f"Title: {title_text}\nURL: {href}\nSnippet: {body}")
        except Exception as lib_err:
            return f"No search results found. Library fallback also failed: {str(lib_err)}"

    if not snippets:
        return "No search results found."

    # Save initial snippets as a high-accuracy baseline context file
    snippets_md = "# Web Search Snippets\n\n" + "\n\n---\n\n".join(snippets)
    snippets_file = save_page_markdown(search_url, conversation_id, f"Search_Snippets_{query[:20]}", content=snippets_md)
    if snippets_file:
        saved_files.insert(0, snippets_file)

    # --- Step 2: Fetch and save each result URL ---
    for page_url, page_title in urls_with_titles[:3]:
        file_path = save_page_markdown(page_url, conversation_id, page_title)
        if file_path:
            saved_files.append(file_path)

    # --- Step 3: Summarize and return results ---
    if saved_files:
        return summarize_search_results(saved_files, conversation_id)
    
    return "Failed to save or summarize any search results."

@tool
def internet_search(query: str, deep_search: bool = False, conversation_id: str = "default") -> str:
    """
    Searches the internet for live data, news, people, and current events. 
    Returns article titles, snippets, and URLs. 
    
    Set 'deep_search' to True if the user asks for detailed research, full article content, 
    or information about private individuals likely to require more than just top-level snippets.
    'conversation_id' should be the current conversation ID if available.
    """
    if deep_search:
        print(f"[RESEARCH][{conversation_id}] Forced Deep Search requested for: {query}", flush=True)
        return deep_internet_search(query, conversation_id)

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
                if "google.com" in href:
                    continue
                results.append(f"Title: {title.text}\nURL/Source: {href}\nSummary: {snippet.text}")
        
        if not results:
            print(f"[RESEARCH][{conversation_id}] Robust search empty. Falling back to Deep Search...", flush=True)
            return deep_internet_search(query, conversation_id)
            
        return "\n\n".join(results[:5])
    except Exception as e:
        return f"Internet Search Error: {str(e)}"

def get_internet_tools():
    return [fetch_website_content, scrape_technical_docs, search_tool, internet_search]
