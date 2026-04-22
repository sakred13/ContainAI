import os
import requests
from markdownify import markdownify as md
from langchain_core.messages import HumanMessage

# We use late imports for 'llm' to avoid circular dependency
# from common.llm_client import llm

import uuid
CONTEXT_BASE_DIR = "/context"

def save_page_markdown(url: str, conversation_id: str, title: str, content: str = "") -> str:
    """Reads a webpage (or uses provided content) and writes it as clean markdown."""
    if not conversation_id or conversation_id in ["default", "unknown", "<current conversation ID>", "None"]:
        conversation_id = str(uuid.uuid4())
        
    print(f"\n[RESEARCH][{conversation_id}] Saving page: {title} ({url})", flush=True)
    try:
        import bs4
        if not content:
            response = requests.get(url, timeout=10)
            content = response.text
        
        # Pre-clean with BeautifulSoup to remove obviously useless tags
        soup = bs4.BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
            
        markdown_content = md(str(soup), heading_style="ATX")
        print(f"[RESEARCH] Raw Markdown length: {len(markdown_content)} chars", flush=True)
        
        # Clean title for filename
        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_', '-')]).strip()
        safe_title = safe_title.replace(' ', '_')[:50]
        
        dir_path = os.path.join(CONTEXT_BASE_DIR, conversation_id)
        os.makedirs(dir_path, exist_ok=True)
        
        file_path = os.path.join(dir_path, f"{safe_title}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"[RESEARCH][{conversation_id}] Saved to: {file_path}", flush=True)
        return file_path
    except Exception as e:
        print(f"Error in save_page_markdown: {e}")
        return ""

def compact_page_markdown(file_path: str, conversation_id: str = "unknown") -> None:
    """Uses LLM to compact file content to only useful information and updates the file."""
    print(f"[RESEARCH][{conversation_id}] Compacting: {os.path.basename(file_path)}", flush=True)
    if not os.path.exists(file_path):
        print(f"[RESEARCH][{conversation_id}] ERROR: File not found {file_path}", flush=True)
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return
        from common.llm_client import llm
        prompt = (
            "SYSTEM: You are an expert Intelligence Analyst. Your task is to process raw data from multiple research sources into a comprehensive, high-fidelity research dossier.\n"
            "STRICT RULES:\n"
            "1. EXHAUSTIVE DETAIL: Preserve EVERY unique factual detail: dates, locations, full names, event titles, scientific results, technical specs, and specific affiliations found in the text.\n"
            "2. STRUCTURE: Do NOT summarize into a short paragraph. Use structured, nested markdown (## Sections, ### Sub-sections, * Bullet points) that logically organizes the specific subject matter being analyzed.\n"
            "3. NO DATA LOSS: If the source lists multiple distinct items, examples, or entries, you MUST list them ALL. Do not merge distinct facts into vague generalizations or omit details for the sake of brevity.\n"
            "4. ELIMINATE ONLY true noise: navigation bars, social media buttons, ads, and generic 'bot detection' warnings. Do NOT attribute website operator branding or legal boilerplate to the subject of the research.\n"
            "5. OUTPUT ONLY the refined markdown. No conversational filler.\n\n"
            f"RAW DATA FOR ANALYSIS:\n{content[:25000]}"
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        compacted_content = response.content.strip()
        print(f"[RESEARCH][{conversation_id}] LLM raw output length: {len(compacted_content)} chars", flush=True)
        
        # Clean up common LLM artifacts
        if "```" in compacted_content:
            parts = compacted_content.split("```")
            compacted_content = parts[1] if len(parts) >= 3 else parts[-1]
            if compacted_content.startswith("markdown"):
                compacted_content = compacted_content[8:]
            compacted_content = compacted_content.strip()
            
        # Post-process: Remove lines that are just single country names (removes leftover region lists)
        lines = compacted_content.splitlines()
        filtered_lines = []
        for line in lines:
            clean = line.strip().strip('-').strip('*').strip()
            # Simple heuristic: if a line is just one word and that word is in a list of country-like noise
            if clean in ["Argentina", "Australia", "Austria", "Belgium", "Bulgaria", "Brazil", "Chile", "China"]:
                continue
            filtered_lines.append(line)
        compacted_content = "\n".join(filtered_lines)
        
        print(f"[RESEARCH][{conversation_id}] Final compacted length: {len(compacted_content)} chars", flush=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(compacted_content)
    except Exception as e:
        print(f"[RESEARCH][{conversation_id}] ERROR in compact_page_markdown: {e}", flush=True)

def summarize_search_results(file_paths: list, conversation_id: str) -> str:
    """Combines compacted MD files with '----' delimiter, saves to a summary file, and compacts it."""
    if not conversation_id or conversation_id in ["default", "unknown", "<current conversation ID>", "None"]:
        conversation_id = str(uuid.uuid4())
        
    print(f"[RESEARCH][{conversation_id}] Summarizing {len(file_paths)} files into raw context", flush=True)
    try:
        combined_content = []
        has_snippets = any("Search_Snippets" in p for p in file_paths)
        
        for path in file_paths:
            if os.path.exists(path):
                # Optimization: If we have high-quality snippets, exclude the noisy Search Index
                if has_snippets and "Search_Index" in path:
                    continue
                    
                print(f"[RESEARCH][{conversation_id}] Appending: {os.path.basename(path)}", flush=True)
                with open(path, "r", encoding="utf-8") as f:
                    # Truncate each file to 2500 chars. This is usually the sweet spot for 
                    # the main content without hitting navigation boilerplate.
                    combined_content.append(f"### SOURCE: {os.path.basename(path)}\n{f.read()[:2500]}")
        
        intro = "RESEARCH DATA FOUND (Use this to provide an exhaustive, detailed answer):\n"
        summary_text = intro + "\n\n" + "\n\n---\n\n".join(combined_content)
        
        # Save the full concatenated log for reference (limited to 40k chars)
        dir_path = os.path.join(CONTEXT_BASE_DIR, conversation_id)
        os.makedirs(dir_path, exist_ok=True)
        summary_file = os.path.join(dir_path, "search_raw_concat.md")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_text[:40000])
            
        # Return a tight 12k char window to the agent (approx 3k tokens)
        print(f"[RESEARCH][{conversation_id}] Concatenation complete. Returning {min(len(summary_text), 12000)} chars to agent.", flush=True)
        return summary_text[:12000] 
    except Exception as e:
        print(f"[RESEARCH][{conversation_id}] ERROR in summarize_search_results: {e}", flush=True)
        return f"Error gathering results: {e}"

def get_context_tools():
    """Returns a list of tools for context management. Currently empty as requested."""
    return []
