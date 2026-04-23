import os
import json
import re
from datetime import datetime, timedelta
from common.db_client import db_client
from common.llm_client import llm, agent_executor
from models.enums import ConversationStatus, AgentID, SystemPrompt
from langchain_core.messages import HumanMessage

GLOBAL_DOCS_DIR = "/context/global_docs"

def prepare_librarian_context(convo_id):
    """Fetches requirement and answers from DB."""
    state_json = db_client.get_state(convo_id)
    if not state_json:
        return "No requirement data found."
    
    req = state_json.get("state", {}).get("requirement", "No specific requirement.")
    q_a = [f"Q: {q.get('question')}\nA: {q.get('answer')}" 
           for q in state_json.get("state", {}).get("userQuestions", []) if q.get("answered")]
    
    return f"OVERALL REQUIREMENT: {req}\n\nSPECIFICS GATHERED:\n" + "\n\n".join(q_a)

APPROVAL_PROMPT = (
    "You are a Technical Librarian. Given a list of candidate libraries and a user message, "
    "identify which libraries from the CANDIDATES list the user has approved or requested to use.\n\n"
    "CANDIDATES:\n{candidates}\n\n"
    "USER MESSAGE: \"{user_input}\"\n\n"
    "OUTPUT RULE: Respond ONLY with a JSON list of the EXACT names from the CANDIDATES list that are approved. "
    "If the user says 'all', include every name from the CANDIDATES list.\n"
    "Example: [\"beautifulsoup4\", \"markdownify\"]"
)



def run_librarian_workflow(convo_id, agent_executor, model_name, user_input=None):
    """
    Main entry point for Librarian workflow. 
    Handles state transitions:
    1. ELICITATION_COMPLETE -> LISTED_DEPENDENCIES (Planning/Discovery)
    2. LISTED_DEPENDENCIES -> DEPENDENCIES_FETCHED (Fetching after user approval)
    """
    state_json = db_client.get_full_state(convo_id)

    if not state_json:
        return "Database state not found for this conversation."

    current_status = state_json.get("status")

    # --- STEP 1: PLANNING (LISTING) ---
    if current_status == ConversationStatus.ELICITATION_COMPLETE.value:
        return _run_listing_phase(convo_id, agent_executor, model_name, state_json)

    # --- STEP 2: FETCHING (AFTER APPROVAL) ---
    if current_status == ConversationStatus.LISTED_DEPENDENCIES.value:
        return _run_fetching_phase(convo_id, agent_executor, model_name, state_json, user_input)

    return f"Librarian: Current status is '{current_status}'. No action taken."

def _run_listing_phase(convo_id, agent_executor, model_name, state_json):
    print(f"[LIBRARIAN][{convo_id}] Phase 1: Planning/Decomposing sub-tasks...", flush=True)
    results = prepare_librarian_context(convo_id)
    plan_prompt = SystemPrompt.get_prompt(AgentID.LIBRARIAN, {"elicitation_results": results, "mode": "plan"})
    
    # Use agent_executor to allow the model to search for best libraries per sub-task
    plan_res = agent_executor.invoke({"messages": [HumanMessage(content=plan_prompt)]})
    res_text = plan_res["messages"][-1].content if "messages" in plan_res else str(plan_res)
    
    plan_data = _extract_json_obj(res_text)
    sub_tasks = plan_data.get("sub_tasks", []) if plan_data else []
    
    if not sub_tasks:
        # Fallback if decomposition fails
        return f"Librarian: I couldn't automatically break down the project. Could you list the technical tasks you need help with? Raw output: {res_text[:150]}"

    # Map to dependencies list for DB state
    dependencies = []
    for st in sub_tasks:
        dependencies.append({
            "task": st.get("task"),
            "name": st.get("library"),
            "reason": st.get("reason"),
            "approved": False,
            "fetched": False
        })

    state_json["dependencies"] = dependencies
    # Clean up top-level DB fields from the JSON object before saving
    clean_state = {k: v for k, v in state_json.items() if k not in ["status", "agent_id", "model_name"]}
    
    db_client.upsert_state(
        convo_id=convo_id,
        state_json=clean_state,
        agent_id=AgentID.LIBRARIAN.value,
        model_name=model_name,
        status=ConversationStatus.LISTED_DEPENDENCIES.value
    )


    resp = "### 📚 Recommended Stack\n"
    resp += "I've broken your project into technical sub-tasks and found the best-in-class libraries for each:\n\n"
    for d in dependencies:
        resp += f"- **{d['task']}**: `{d['name']}`\n  *Rationale: {d['reason']}*\n"
    resp += "\n**Which of these would you like to use?** (e.g., 'Use all', 'Only X and Y', or '/invoke librarian Use X')."
    return resp

def _run_fetching_phase(convo_id, agent_executor, model_name, state_json, user_input):
    print(f"[LIBRARIAN][{convo_id}] Phase 2: Processing user approval and fetching...", flush=True)
    
    dependencies = state_json.get("dependencies", [])
    if not dependencies:
        return "Internal Error: Dependencies list missing from state."

    # --- INTELLIGENT LLM-BASED APPROVAL ---
    print(f"[LIBRARIAN][{convo_id}] Parsing approval from user input: {user_input[:50]}...", flush=True)
    
    # 1. Greedy Check: If user says "Approved", "All", or similar, just approve everything.
    user_input_low = user_input.lower().strip()
    is_greedy = any(x in user_input_low for x in ["approved", "all", "yes", "proceed", "go", "implement"])
    
    if is_greedy:
        print(f"[LIBRARIAN][{convo_id}] Greedy approval detected. Approving all candidates.", flush=True)
        for d in dependencies:
            d["approved"] = True
        approval_granted = True
    else:
        candidates_str = "\n".join([f"- {d['name']}" for d in dependencies])
        approval_res = llm.invoke([
            HumanMessage(content=APPROVAL_PROMPT.format(candidates=candidates_str, user_input=user_input))
        ])
        print(f"[LIBRARIAN][{convo_id}] LLM Approval Reasoning: {approval_res.content}", flush=True)
        
        approved_names = _extract_json_list(approval_res.content)
        approval_granted = False
        
        for d in dependencies:
            name_low = d["name"].lower()
            if any(name_low in n.lower() or n.lower() in name_low for n in approved_names):
                d["approved"] = True
                approval_granted = True

    if not approval_granted:
        return "I didn't catch which libraries you approved (or I couldn't match them). Please clarify which ones I should implement."



    approved_libs = [d for d in dependencies if d.get("approved")]
    os.makedirs(GLOBAL_DOCS_DIR, exist_ok=True)
    cache_threshold = datetime.now() - timedelta(days=60)

    for lib in approved_libs:
        lib_name = lib["name"].strip()
        safe_name = lib_name.lower().replace('-', '_')
        lib_dir = os.path.join(GLOBAL_DOCS_DIR, safe_name)
        meta_path = os.path.join(lib_dir, "metadata.json")
        
        print(f"[LIBRARIAN][{convo_id}] Checking cache: {meta_path}", flush=True)
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 50:
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                scraped_str = meta.get("scraped_at", "1970-01-01T00:00:00").replace("Z", "")
                scraped_at = datetime.fromisoformat(scraped_str)
                if scraped_at > cache_threshold:
                    print(f"[LIBRARIAN][{convo_id}] Cache HIT for {lib_name}. Skipping re-fetch.", flush=True)
                    lib["fetched"] = True
                    continue
            except Exception as e:
                print(f"[LIBRARIAN][{convo_id}] Cache check error for {lib_name}: {str(e)}", flush=True)


        # --- FETCH (RESEARCH) ---
        print(f"[LIBRARIAN][{convo_id}] Cache MISS for {lib_name}. Researching via tools...", flush=True)
        
        # PHASE A: RAW GATHERING (Let the agent use tools freely)
        gather_prompt = (
            f"Independent Research Task: Gather all technical details for the Python library '{lib_name}'.\n"
            "1. Use 'get_pypi_details' for metadata.\n"
            "2. Use 'search_cheat_sheet' for examples.\n"
            "3. Use 'internet_search' or 'get_local_pydoc' for API signatures.\n"
            "GATHER AS MUCH AS POSSIBLE. Do not summarize yet."
        )
        gather_res = agent_executor.invoke({"messages": [HumanMessage(content=gather_prompt)]})
        raw_dump = gather_res["messages"][-1].content if "messages" in gather_res else str(gather_res)
        
        # PHASE B: SYNTHESIS (Force JSON format from the raw dump)
        synth_prompt = SystemPrompt.get_prompt(AgentID.LIBRARIAN, {"library_name": lib_name, "mode": "research"})
        synth_prompt += f"\n\nRAW RESEARCH DATA:\n{raw_dump}"
        
        synth_res = llm.invoke([HumanMessage(content=synth_prompt)])
        doc_data = _extract_json_obj(synth_res.content)
        
        if doc_data:
            # Check for quality before writing
            has_quality = len(str(doc_data.get("usage_examples", ""))) > 50 or len(str(doc_data.get("api_reference", ""))) > 50
            
            if has_quality:
                os.makedirs(lib_dir, exist_ok=True)
                
                # 1. Save Meta (Always JSON)
                with open(meta_path, "w") as f:
                    metadata = doc_data.get("metadata", {})
                    if not isinstance(metadata, dict): metadata = {"raw": str(metadata)}
                    metadata["scraped_at"] = datetime.now().isoformat()
                    json.dump(metadata, f, indent=2)
                
                # 2. Save Usage Examples
                examples = doc_data.get("usage_examples", "")
                if isinstance(examples, dict): examples = json.dumps(examples, indent=2)
                with open(os.path.join(lib_dir, "usage_examples.md"), "w") as f:
                    f.write(str(examples))
                
                # 3. Save API Reference
                api_ref = doc_data.get("api_reference", "")
                if isinstance(api_ref, dict): api_ref = json.dumps(api_ref, indent=2)
                with open(os.path.join(lib_dir, "api_reference.md"), "w") as f:
                    f.write(str(api_ref))

                lib["fetched"] = True
                print(f"[LIBRARIAN][{convo_id}] Successfully fetched and cached {lib_name}.", flush=True)
            else:
                print(f"[LIBRARIAN][{convo_id}] Research for {lib_name} returned insufficient data. Skipping write to protect cache.", flush=True)



    state_json["dependencies"] = dependencies
    # Clean up top-level DB fields from the JSON object before saving
    clean_state = {k: v for k, v in state_json.items() if k not in ["status", "agent_id", "model_name"]}

    # ONLY promote status if ALL approved libraries were successfully fetched
    all_approved_ready = approved_libs and all(l.get("fetched") for l in approved_libs)
    new_status = ConversationStatus.DEPENDENCIES_FETCHED.value if all_approved_ready else ConversationStatus.LISTED_DEPENDENCIES.value

    db_client.upsert_state(
        convo_id=convo_id,
        state_json=clean_state,
        agent_id=AgentID.LIBRARIAN.value,
        model_name=model_name,
        status=new_status
    )

    if all_approved_ready:
        return f"Done! I have fetched and cached the technical documentation for: " + ", ".join([l['name'] for l in approved_libs]) + "."
    else:
        return f"Partial success. Some libraries could not be fully documented or were not approved. Please review the selection or try again."


def _extract_json_list(text):
    """Robustly extracts a JSON list from text."""
    try:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        return json.loads(match.group(1)) if match else []
    except: return []

def _extract_json_obj(text):

    """Robustly extracts a JSON object from text, handling trailing/leading garbage."""
    try:
        # Look for the first '{' and the last '}'
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
             return json.loads(match.group(1))
        return None
    except: return None
