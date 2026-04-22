import requests
import bs4
import re

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def fetch_html(url, timeout=15):
    """Utility to fetch HTML with consistent headers and error handling."""
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[SCRAPER_UTILS] Error fetching {url}: {e}")
        return None

def get_soup(html):
    """Utility to parse HTML into BeautifulSoup."""
    if not html:
        return None
    return bs4.BeautifulSoup(html, "html.parser")

def extract_technical_contracts(soup):
    """
    Highly specialized scraper that filters documentation to find ONLY
    technical contracts, function signatures, and code examples.
    """
    if not soup:
        return "No content found."

    extracted_blocks = []
    
    # 1. Target Sphinx/ReadTheDocs 'definition lists' (dl, dt, dd)
    for dl in soup.find_all('dl'):
        dt = dl.find('dt')
        if dt:
            # Clean up signatures (remove the 'link' symbol usually found in Sphinx)
            signature = dt.get_text(strip=True).replace('¶', '')
            if len(signature) > 5:
                extracted_blocks.append(f"### [SIGNATURE]\n`{signature}`")
            
            dd = dl.find('dd')
            if dd:
                # Capture the description but keep it relatively concise
                desc = dd.get_text(separator=' ', strip=True)
                if len(desc) > 300: desc = desc[:300] + "..."
                extracted_blocks.append(f"**Description:** {desc}\n")

    # 2. Target ALL Code blocks (pre, code, div with highlight classes)
    for code_block in soup.find_all(['pre', 'code', 'div'], class_=re.compile(r'highlight|code|example|python|snippet', re.I)):
        code_text = code_block.get_text().strip()
        # Filter for 'meaty' code blocks (contain logic or signatures)
        if len(code_text) > 15 and any(char in code_text for char in '()[]=:.'):
            # Clean up extra line breaks and redundant spaces
            clean_code = "\n".join([line for line in code_text.splitlines() if line.strip()])
            extracted_blocks.append(f"```python\n{clean_code}\n```")

    # 3. Look for plain-text 'signatures' (e.g., lines starting with 'def ' or 'class ')
    text_content = soup.get_text(separator='\n', strip=True)
    potential_sigs = re.findall(r'^(?:def|class)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\)\s*:?.*$', text_content, re.MULTILINE)
    for sig in potential_sigs[:10]: # Limit to avoid context bloat
        if sig not in str(extracted_blocks):
            extracted_blocks.append(f"### [SPEC]\n`{sig.strip()}`")

    if not extracted_blocks:
        # If still nothing structured, return a managed chunk of the raw text
        return text_content[:6000]

    final_text = "\n".join(extracted_blocks)
    
    # SAFETY CAP: 12,000 chars is roughly 3,000-4,000 tokens.
    # This leaves plenty of room in a 16k context window for instructions and thought.
    if len(final_text) > 12000:
        print(f"[SCRAPER_UTILS] WARNING: Capping output from {len(final_text)} to 12000 chars.")
        return final_text[:12000] + "\n\n... [TRUNCATED FOR CONTEXT SAFETY] ..."

    return final_text

def clean_search_snippets(soup):
    """Cleans up raw search result pages for summarized views."""
    results = []
    for result in soup.find_all('div', class_='result'):
        title = result.find('a', class_='result__a')
        snippet = result.find('a', class_='result__snippet')
        if title and snippet:
            results.append(f"Title: {title.text}\nURL: {title.get('href')}\nSummary: {snippet.text.strip()}")
    return "\n\n".join(results[:5])
