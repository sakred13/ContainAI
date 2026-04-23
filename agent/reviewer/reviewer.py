import os
import re
import docker
import json
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from common.db_client import DBClient
from models.enums import AgentID, ConversationStatus, SystemPrompt


# Constants
SANDBOX_CONTAINER = "containai_sandbox"
CONTEXT_DIR = "/context"
def _get_sandbox_path(convo_id):
    """Returns the isolated sandbox path for a conversation."""
    return os.path.join("/context/sandbox", convo_id)

db_client = DBClient()
docker_client = docker.from_env()

def run_reviewer_task(convo_id, model_name="llama3.2"):
    print(f"[REVIEWER][{convo_id}] Starting code review and testing...", flush=True)
    
    state = db_client.get_full_state(convo_id)
    if not state:
        return {"error": "Conversation state not found."}
    
    state_json = state # get_full_state returns a combined dict
    db_client.add_thought(convo_id, AgentID.REVIEWER.value, "Retrieving conversation state and preparing sandbox for execution loop.")

    # 2. Find the script to run
    # The developer saves it as /context/sandbox/app_<convo_id>.py
    script_name = f"app_{convo_id}.py"
    script_path = os.path.join(SANDBOX_DIR, script_name)
    
    if not os.path.exists(script_path):
        return {"error": f"Generated script {script_name} not found in sandbox folder."}

    with open(script_path, "r") as f:
        source_code = f.read()

    # 3. Execution Loop (with auto-provisioning)
    max_attempts = 3
    attempts = 0
    test_passed = False
    crash_log = ""

    # Check for test input in isolated dir
    sandbox_dir = _get_sandbox_path(convo_id)
    test_input = os.path.join(sandbox_dir, "test.html")
    cmd = f"python {script_name} {os.path.basename(test_input) if os.path.exists(test_input) else ''}"

    while attempts < max_attempts:
        attempts += 1
        print(f"[REVIEWER][{convo_id}] Attempt {attempts}: Running {cmd}...", flush=True)
        db_client.add_thought(convo_id, AgentID.REVIEWER.value, f"Attempt {attempts}: Executing script inside the isolated container sandbox.")
        
        container = docker_client.containers.get(SANDBOX_CONTAINER)
        # Run inside the isolated subfolder within the container
        workdir = f"/sandbox/{convo_id}"
        exec_res = container.exec_run(cmd, workdir=workdir)
        exit_code = exec_res.exit_code
        output = exec_res.output.decode("utf-8")
        
        if exit_code == 0:
            print(f"[REVIEWER][{convo_id}] DEBUG: exit_code=0, output_len={len(output)}", flush=True)
            print(f"[REVIEWER][{convo_id}] RAW_OUTPUT_START\n{output}\nRAW_OUTPUT_END", flush=True)
            # Check if the output contains a caught exception message


            if "Error" in output and "successful" not in output.lower():
                print(f"[REVIEWER][{convo_id}] Caught exception detected in logs despite exit code 0.", flush=True)
                test_passed = False
                crash_log = output
                break
            
            print(f"[REVIEWER][{convo_id}] Execution successful!", flush=True)
            test_passed = True
            crash_log = output
            break

        else:
            print(f"[REVIEWER][{convo_id}] Execution failed with exit code {exit_code}.", flush=True)
            crash_log = output
            
            # Check for ModuleNotFoundError
            if "ModuleNotFoundError" in output:
                match = re.search(r"No module named '([^']+)'", output)
                if match:
                    lib_name = match.group(1)
                    print(f"[REVIEWER][{convo_id}] Missing library detected: {lib_name}. Installing...", flush=True)
                    container.exec_run(f"pip install {lib_name}")
                    continue # Try again after install
            
            # If not a simple import error, or we reached limit, stop
            break

    # 4. Finalize Results
    if test_passed:
        db_client.update_status(convo_id, ConversationStatus.COMPLETED.value)
        return {
            "status": "SUCCESS",
            "message": "Code ran successfully in the sandbox.",
            "output": crash_log
        }
    else:
        # LLM Analysis
        print(f"[REVIEWER][{convo_id}] Analyzing failure via {model_name}...", flush=True)
    db_client.add_thought(convo_id, AgentID.REVIEWER.value, f"Execution failed. Requesting root cause analysis and refactor suggestions from {model_name}.")
        llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL"))
        
        analysis_prompt = SystemPrompt.get_prompt(AgentID.REVIEWER, {
            "crash_log": crash_log,
            "source_code": source_code,
            "mode": "analyze"
        })
        
        res = llm.invoke([HumanMessage(content=analysis_prompt)])
        comments = res.content
        
        # Save comments to DB state
        state_json = state # get_full_state already returned the state dict

        state_json["review_comments"] = comments
        state_json["last_crash_log"] = crash_log
        
        db_client.upsert_state(
            convo_id=convo_id,
            state_json=state_json,
            agent_id=AgentID.REVIEWER.value,
            model_name=model_name,
            status=ConversationStatus.COMMENTS_POSTED.value
        )
        
        return {
            "status": "FAILED",
            "message": "Reviewer found bugs. Comments posted.",
            "comments": comments
        }
