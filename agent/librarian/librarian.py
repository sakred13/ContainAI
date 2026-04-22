import os
import json
import re
from common.db_client import db_client
from common.llm_client import llm, agent_executor
from models.enums import ConversationStatus, AgentID, SystemPrompt
from langchain_core.messages import HumanMessage

def prepare_librarian_context(convo_id):
    """Fetches requirement and answers from DB."""
    state_json = db_client.get_state(convo_id)
    if not state_json:
        return "No requirement data found."
    
    req = state_json.get("state", {}).get("requirement", "No specific requirement.")
    q_a = [f"Q: {q.get('question')}\nA: {q.get('answer')}" 
           for q in state_json.get("state", {}).get("userQuestions", []) if q.get("answered")]
    
    return f"OVERALL REQUIREMENT: {req}\n\nSPECIFICS GATHERED:\n" + "\n\n".join(q_a)

def run_librarian_workflow(convo_id, agent_executor, model_name):
    """
    Drives the multi-step, context-safe Librarian research pipeline.
    Plan (Raw LLM) -> Loop(Research Single Library via Agent) -> Finalize
    """
    try:
        # --- PHASE 1: PLANNING ---
        print(f"[LIBRARIAN][{convo_id}] Phase 1: Planning libraries (Raw LLM)...", flush=True)
        results = prepare_librarian_context(convo_id)
        plan_prompt = SystemPrompt.get_prompt(AgentID.LIBRARIAN, {"elicitation_results": results, "mode": "plan"})
        
        # USE RAW LLM: Forces JSON output and avoids accidental tool calling
        plan_res = llm.invoke([HumanMessage(content=plan_prompt)])
        plan_text = plan_res.content
        
        print(f"[LIBRARIAN][{convo_id}] Plan result: {plan_text[:100]}...", flush=True)
        
        libraries = _extract_json_list(plan_text)
        if not libraries:
            return f"Librarian could not identify any specific libraries from model output: {plan_text[:150]}..."

        # --- PHASE 2: ITERATIVE RESEARCH ---
        print(f"[LIBRARIAN][{convo_id}] Phase 2: Researching {len(libraries)} libraries independently...", flush=True)
        all_processed_deps = []
        convo_dir = os.path.join("/context", str(convo_id))
        os.makedirs(convo_dir, exist_ok=True)

        for lib_name in libraries:
            print(f"[LIBRARIAN][{convo_id}] Researching {lib_name}...", flush=True)
            research_prompt = SystemPrompt.get_prompt(AgentID.LIBRARIAN, {"library_name": lib_name, "mode": "research"})
            
            # FRESH CALL: This clears previous tool outputs from context
            research_res = agent_executor.invoke({"messages": [HumanMessage(content=research_prompt)]})
            res_text = research_res["messages"][-1].content if "messages" in research_res else str(research_res)
            
            dep_data = _extract_json_obj(res_text)
            if dep_data:
                # Save specialized documentation to disk
                file_name = f"{lib_name.lower().replace('-', '_')}.md"
                file_path = os.path.join(convo_dir, file_name)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Documentation: {dep_data.get('library')}\n\n")
                    f.write(f"Source: {dep_data.get('sourceUrl')}\n\n")
                    f.write(dep_data.get("technicalContract", ""))
                
                # Update with local path and add to list
                dep_data["usageDoc"] = file_path
                # Remove the bulk markdown from the DB object to keep DB light
                if "technicalContract" in dep_data: del dep_data["technicalContract"]
                
                all_processed_deps.append(dep_data)

        # --- PHASE 3: FINALIZE ---
        state_json = db_client.get_state(convo_id)
        if state_json:
            state_json["dependencies"] = all_processed_deps
            db_client.upsert_state(
                convo_id=convo_id,
                state_json=state_json,
                agent_id=AgentID.LIBRARIAN.value,
                model_name=model_name,
                status=ConversationStatus.READY.value
            )

        return f"Librarian complete. Researched {len(all_processed_deps)} libraries. Context saved to {convo_dir}."

    except Exception as e:
        print(f"[LIBRARIAN PIPELINE ERROR]: {e}")
        return f"Research pipeline failed: {str(e)}"

def _extract_json_list(text):
    try:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        return json.loads(match.group(1)) if match else []
    except: return []

def _extract_json_obj(text):
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        return json.loads(match.group(1)) if match else None
    except: return None
