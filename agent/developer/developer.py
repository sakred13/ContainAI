import os
import re
from datetime import datetime
from common.db_client import db_client
from common.llm_client import llm
from models.enums import ConversationStatus, AgentID, SystemPrompt
from langchain_core.messages import HumanMessage

GLOBAL_DOCS_DIR = "/context/global_docs"

def _get_sandbox_path(convo_id):
    """Returns the isolated sandbox path for a conversation."""
    path = os.path.join("/context/sandbox", convo_id)
    os.makedirs(path, exist_ok=True)
    return path

def _get_documentation_context(state_json):
    """Aggregates all approved library documentation into a single context string."""
    dependencies = state_json.get("dependencies", [])
    doc_context = ""
    
    for lib in dependencies:
        if lib.get("approved"):
            lib_name = lib["name"].lower().replace('-', '_')
            lib_path = os.path.join(GLOBAL_DOCS_DIR, lib_name)
            
            if os.path.exists(lib_path):
                doc_context += f"\n--- DOCUMENTATION: {lib['name']} ---\n"
                # API Reference
                api_path = os.path.join(lib_path, "api_reference.md")
                if os.path.exists(api_path):
                    with open(api_path, "r") as f:
                        doc_context += f"API REFERENCE:\n{f.read()}\n"
                
                # Usage Examples
                usage_path = os.path.join(lib_path, "usage_examples.md")
                if os.path.exists(usage_path):
                    with open(usage_path, "r") as f:
                        doc_context += f"USAGE EXAMPLES:\n{f.read()}\n"
    
    return doc_context or "No documentation found in cache."

def run_developer_workflow(convo_id, model_name, user_input=None):
    """
    Developer Agent Workflow:
    1. DEPENDENCIES_FETCHED -> Generate Implementation Plan.
    2. PLAN_GENERATED -> Wait for 'Approved' or feedback.
    3. PLAN_APPROVED -> Write code to sandbox.
    """
    state_json = db_client.get_full_state(convo_id)
    if not state_json:
        return "Conversation state not found."

    current_status = state_json.get("status")
    
    # Check if we are currently "Implementing" (busy state)
    if state_json.get("is_working"):
         return "The developer agent is working on your code. Please wait."

    # --- PHASE 1: GENERATE PLAN ---
    if current_status == ConversationStatus.DEPENDENCIES_FETCHED.value:
        return _step_generate_plan(convo_id, model_name, state_json)

    # --- PHASE 2: HANDLE APPROVAL / FEEDBACK ---
    if current_status == ConversationStatus.PLAN_GENERATED.value:
        return _step_handle_approval(convo_id, model_name, state_json, user_input)

    # --- PHASE 3: IMPLEMENT CODE ---
    if current_status in [ConversationStatus.PLAN_APPROVED.value, ConversationStatus.COMMENTS_POSTED.value]:
        return _step_implement_code(convo_id, model_name, state_json)

    return f"Developer: Current status is '{current_status}'. No action taken."

def _step_generate_plan(convo_id, model_name, state_json):
    requirements = state_json.get("state", {}).get("requirement", "No requirements.")
    docs = _get_documentation_context(state_json)
    print(f"[DEVELOPER][{convo_id}] Documentation Context Loaded: {len(docs)} characters", flush=True)

    
    prompt = SystemPrompt.get_prompt(AgentID.DEVELOPER, {
        "mode": "plan",
        "requirement_context": requirements,
        "documentation_context": docs
    })
    
    res = llm.invoke([HumanMessage(content=prompt)])
    plan_text = res.content
    
    # Update local state_json
    state_json["implementationPlan"] = plan_text
    
    # Save to DB
    clean_state = {k: v for k, v in state_json.items() if k not in ["status", "agent_id", "model_name"]}
    db_client.upsert_state(
        convo_id=convo_id,
        state_json=clean_state,
        agent_id=AgentID.DEVELOPER.value,
        model_name=model_name,
        status=ConversationStatus.PLAN_GENERATED.value
    )
    
    return plan_text

def _step_handle_approval(convo_id, model_name, state_json, user_input):
    if not user_input:
        return "I am waiting for your approval of the plan. Please say 'Approved' or provide feedback."

    user_input_low = user_input.lower().strip()
    
    if "approved" in user_input_low or "approve" in user_input_low:
        # Move to PLAN_APPROVED
        clean_state = {k: v for k, v in state_json.items() if k not in ["status", "agent_id", "model_name"]}
        db_client.upsert_state(
            convo_id=convo_id,
            state_json=clean_state,
            agent_id=AgentID.DEVELOPER.value,
            model_name=model_name,
            status=ConversationStatus.PLAN_APPROVED.value
        )
        # Immediately trigger implementation
        return _step_implement_code(convo_id, model_name, state_json)
    
    else:
        # feedback loop. Update requirement and regenerate plan.
        current_req = state_json.get("state", {}).get("requirement", "")
        updated_req = f"{current_req}\n\nUSER FEEDBACK ON PLAN: {user_input}"
        
        if "state" not in state_json: state_json["state"] = {}
        state_json["state"]["requirement"] = updated_req
        
        # Regenerate plan immediately
        return _step_generate_plan(convo_id, model_name, state_json)

def _step_implement_code(convo_id, model_name, state_json):
    # Set "working" flag to prevent concurrent prompts
    db_client.upsert_state(
        convo_id=convo_id,
        state_json={**state_json, "is_working": True},
        agent_id=AgentID.DEVELOPER.value,
        model_name=model_name,
        status=ConversationStatus.PLAN_APPROVED.value
    )

    plan = state_json.get("implementationPlan", "No plan found.")
    requirements = state_json.get("state", {}).get("requirement", "")
    docs = _get_documentation_context(state_json)
    db_client.add_thought(convo_id, AgentID.DEVELOPER.value, f"Starting implementation. Loaded documentation context ({len(docs)} chars). Processing plan: {plan[:100]}...")

    print(f"[DEVELOPER][{convo_id}] Implementation Phase: Docs Loaded: {len(docs)} characters", flush=True)
    print(f"[DEVELOPER][{convo_id}] DOCS PREVIEW: {docs[:500]}...", flush=True)


    review_ctx = ""
    if state_json.get("review_comments"):
        review_ctx = f"\n\nBUG REPORT FROM REVIEWER:\n{state_json['review_comments']}\n\nLAST CRASH LOG:\n{state_json.get('last_crash_log', '')}"

    prompt = SystemPrompt.get_prompt(AgentID.DEVELOPER, {
        "mode": "implement",
        "implementation_plan": plan,
        "full_context": f"REQUIREMENTS:\n{requirements}\n\nDOCUMENTATION:\n{docs}{review_ctx}"
    })
    
    res = llm.invoke([HumanMessage(content=prompt)])
    code_text = res.content
    db_client.add_thought(convo_id, AgentID.DEVELOPER.value, "LLM has generated code response. Extracting Python block and preparing for sandbox deployment.")
    
    # Extract code from Markdown
    code_match = re.search(r"```python\n(.*?)```", code_text, re.DOTALL)
    actual_code = code_match.group(1) if code_match else code_text
    
    # Write to sandbox with isolation
    sandbox_dir = _get_sandbox_path(convo_id)
    file_name = f"app_{convo_id}.py"
    file_path = os.path.join(sandbox_dir, file_name)
    
    with open(file_path, "w") as f:
        f.write(actual_code)

    db_client.add_thought(convo_id, AgentID.DEVELOPER.value, f"Code successfully saved to {file_path}. Transitioning to READY_FOR_REVIEW.")
    print(f"[DEVELOPER][{convo_id}] Saved isolated code to {file_path}", flush=True)
    
    # Final state update
    state_json["fileLocation"] = f"/sandbox/{file_name}"
    state_json["is_working"] = False
    
    clean_state = {k: v for k, v in state_json.items() if k not in ["status", "agent_id", "model_name"]}
    db_client.upsert_state(
        convo_id=convo_id,
        state_json=clean_state,
        agent_id=AgentID.DEVELOPER.value,
        model_name=model_name,
        status=ConversationStatus.READY_FOR_REVIEW.value
    )
    
    return f"Code has been written to the sandbox!\n\n**File Location**: `/sandbox/{file_name}`\n\n```python\n{actual_code[:500]}...\n```"
