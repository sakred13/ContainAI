"""
orchestrator_agent/routes.py
-----------------------------
Flask Blueprint for /orchestrate and /interject endpoints.

The orchestrator parses a scenario prompt, spawns virtual agents with role-
aware system prompts, and runs a multi-round conversation loop until all
agents signal [CONCLUSION_MET] or one signals APPROVED.

Interjection state is kept in a module-level defaultdict so it is shared
across all requests within the same process.
"""

import os
import json
import re
import time
import threading

import requests
from collections import defaultdict
from flask import Blueprint, request, Response, jsonify
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from common.llm_client import llm, agent_executor, tools, MODEL_NAME, app, OLLAMA_BASE_URL
from common.db_client import db_client
from models.enums import AgentID, ModelName, SystemPrompt
from models.schemas import InvokeRequest
from elicitor import process_elicitor_response
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Model Registry for Dispatch
# ---------------------------------------------------------------------------
MODEL_URLs = {
    ModelName.LLAMA32.value: os.getenv("AGENT_LLAMA_URL", "http://agent_llama:5000"),
    ModelName.LLAMA31_8B.value: os.getenv("AGENT_LLAMA8B_URL", "http://agent_llama_8b:5000"),
    ModelName.GEMMA3_4B.value: os.getenv("AGENT_GEMMA_URL", "http://agent_gemma:5000"),
    ModelName.QWEN25_CODER.value: os.getenv("AGENT_QWEN_URL", "http://agent_qwen:5000"),
    ModelName.DOLPHIN_MISTRAL.value: os.getenv("AGENT_DOLPHIN_URL", "http://agent_dolphin:5000"),
    ModelName.DEEPSEEK_R1_7B.value: os.getenv("AGENT_DEEPSEEK_URL", "http://agent_deepseek:5000"),
}

MODEL_STRENGTHS = ModelName.get_all_strengths_formatted()

# ---------------------------------------------------------------------------
# Blueprint & shared interjection store
# ---------------------------------------------------------------------------
orchestrator_bp = Blueprint("orchestrator", __name__)

# Maps convo_id → list of pending user interjection strings
sim_interjections: defaultdict[str, list] = defaultdict(list)

# ---------------------------------------------------------------------------
# Scenario parsing
# ---------------------------------------------------------------------------
PARSE_PROMPT = (
    "Extract simulation details as JSON. You MUST output ONLY a valid JSON object with the following keys:\n"
    "1. 'agents': A list of objects with 'name', 'personality', 'role_objective', and 'model'.\n"
    "   - 'role_objective': Specific directions extracted for THIS agent ONLY from the prompt.\n"
    "   - 'model' assignment guide: " + MODEL_STRENGTHS + "\n"
    "2. 'rounds': Number of rounds (integer).\n"
    "3. 'scenario': A detailed 1-sentence summary of the interaction.\n"
    "4. 'opener': The EXACT first line the first agent should say to trigger the task.\n\n"
    "EXAMPLE OUTPUT:\n"
    '{"agents": [{"name": "ALICE", "personality": "hacker", "role_objective": "Hack the firewall.", "model": "dolphin-mistral"}, '
    '{"name": "BOB", "personality": "lawyer", "role_objective": "Defend the company in court.", "model": "gemma3:4b"}], '
    '"rounds": 5, "scenario": "A hacker tries to breach a law firm.", "opener": "I am in the main frame."}\n\n'
    "Now extract from this: "
)


def parse_scenario_fallback(prompt: str) -> dict | None:
    """Regex-based fallback parser when the LLM fails to produce valid JSON."""
    name_pattern = r"\b([A-Z]{2,}[A-Z]*)\b"
    names = list(set(re.findall(name_pattern, prompt)))
    stop_words = {
        "THE", "AND", "FOR", "WANT", "BETWEEN", "EACH", "OTHER",
        "ROUND", "ROUNDS", "YEAR", "OLD", "WHO",
    }
    names = [n for n in names if n not in stop_words]

    rounds_match = re.search(r"(\d+)\s*round", prompt, re.IGNORECASE)
    rounds = int(rounds_match.group(1)) if rounds_match else 5

    if len(names) < 2:
        return None

    agents = []
    for name in names:
        personality_match = re.search(
            rf"{name}\s+(?:is\s+)?(.{{10,60}}?)(?:\.|,|and\s+[A-Z]|$)",
            prompt,
            re.IGNORECASE,
        )
        personality = (
            personality_match.group(1).strip()
            if personality_match
            else f"a participant named {name}"
        )
        # Default fallback to Llama 8B
        agents.append({"name": name, "personality": personality, "model": "llama3.1:8b"})

    return {
        "agents": agents,
        "rounds": rounds,
        "scenario": prompt,
        "opener": "Alright, let's get this started!",
    }


def _parse_scenario(scenario_prompt: str):
    """
    Try LLM-based JSON parsing first, then fall back to regex.
    Returns (config_dict | None, raw_llm_output_str).
    """
    parse_response = llm.invoke([HumanMessage(content=PARSE_PROMPT + scenario_prompt)])
    raw = parse_response.content.strip()

    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    config = None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        pass

    if not config or len(config.get("agents", [])) < 2:
        config = parse_scenario_fallback(scenario_prompt)

    return config, raw


# ---------------------------------------------------------------------------
# Per-agent turn logic
# ---------------------------------------------------------------------------
TOOL_ROLES = {"REVIEWER", "TESTER", "QA", "VERIFIER", "VALIDATOR", "EXECUTOR"}


def _build_agent_system(agent: dict, agents: list, scenario: str, i: int) -> str:
    agent_name = agent.get("name", f"Agent_{i+1}")
    personality = agent.get("personality", "A participant in the scenario")
    role_objective = agent.get("role_objective", "Participate in the scenario.")

    other_agents = [
        a.get("name", f"Agent_{j+1}") for j, a in enumerate(agents) if j != i
    ]
    is_tool_agent = agent_name.upper() in TOOL_ROLES

    system = (
        f"CRITICAL ROLEPLAY INSTRUCTION: You are strictly playing the role of {agent_name}.\n\n"
        f"YOUR PERSONALITY AND EXPERTISE:\n{personality}\n\n"
        f"YOUR SPECIFIC OBJECTIVE FOR THIS SCENARIO:\n{role_objective}\n\n"
        f"You are interacting with: {', '.join(other_agents)}.\n\n"
        f"RULES:\n"
        f"1. You MUST act, speak, and think entirely as {agent_name}.\n"
        f"2. You MUST focus ONLY on your specific objective. Do not try to do the jobs of the other agents.\n"
        f"3. Respond ONLY in the FIRST PERSON. Do NOT describe your actions or narrate.\n"
        f"4. Do NOT drop character or break the fourth wall. Only speak the exact words your agent would say in a natural conversation.\n"
        f"5. Do NOT spiral into endless discussion. Put your thoughts forward efficiently and actively push for a conclusion.\n\n"
        "IMPORTANT TERMINATION CLAUSE: If you have reached an agreement, if you have nothing new to add, "
        "if you are just thanking the others or wrapping up, OR if you notice the conversation "
        "is going in circles, you MUST append the exact signal [CONCLUSION_MET] to your response to end the scenario."
    )

    if "code" in personality.lower() or "programmer" in personality.lower() or "coder" in personality.lower():
        system += "\n\nCRITICAL CODING RULE: Whenever you write or share code, you MUST wrap it in Markdown code blocks (```python ... ```)."


    if is_tool_agent:
        system += (
            "\n\nIMPORTANT: You have access to the 'execute_python_code' tool. "
            "When code is presented to you for review, you MUST use the execute_python_code tool "
            "to actually run it and verify the output is correct. "
            "If the code runs correctly and produces the expected output, you can output APPROVED."
        )

    return system, agent_name, is_tool_agent


def _dispatch_agent_turn(agent_name: str, model_key: str, system_msg: str, conversation_history: list, round_num: int, instruction: str, convo_id: str):
    """
    Send the turn to the specific model's /chat endpoint.
    Returns (reply_text, list_of_json_events).
    """
    target_url = MODEL_URLs.get(model_key, MODEL_URLs["llama3.1:8b"])
    
    # Format messages for the worker's /chat endpoint
    payload_msgs = [{"role": "system", "content": system_msg}]
    for entry in conversation_history:
        role = "assistant" if entry["agent"] == agent_name else "user"
        content = entry["text"]
        if entry["agent"] == "HUMAN":
            content = f"***[HUMAN USER OVERRIDE]***: {content}"
        elif role == "user":
            content = f"[{entry['agent']}]: {content}"
            
        # Merge consecutive messages of the same role to strictly alternate user/assistant
        if payload_msgs[-1]["role"] == role:
            payload_msgs[-1]["content"] += f"\n\n{content}"
        else:
            payload_msgs.append({"role": role, "content": content, "name": entry.get("agent", "Unknown")})
    
    # Add the specific turn instruction (e.g. Opener or 'Your turn')
    if payload_msgs[-1]["role"] == "user":
        payload_msgs[-1]["content"] += f"\n\n{instruction}"
    else:
        payload_msgs.append({"role": "user", "content": instruction})
    
    # --- LOG THE REQUEST TO CONSOLE ---
    print(f"\n{'='*60}", flush=True)
    print(f"ROUND {round_num} - DISPATCHING TO: {agent_name} ({model_key})", flush=True)
    print(f"TARGET URL: {target_url}/chat", flush=True)
    print(f"--- PAYLOAD SENT ---", flush=True)
    print(json.dumps(payload_msgs, indent=2), flush=True)
    
    events = []
    full_reply = ""
    
    try:
        resp = requests.post(f"{target_url}/chat", json={"messages": payload_msgs, "convo_id": convo_id}, stream=True, timeout=180)
        for line in resp.iter_lines():
            if not line: continue
            data = json.loads(line)
            if data["type"] == "token":
                full_reply += data["content"]
            elif data["type"] == "tool_start":
                events.append(json.dumps({
                    "type": "tool_event",
                    "agent": f"{agent_name} ({model_key})",
                    "round": round_num,
                    "content": f"🛠️ **{agent_name}** is starting tool `{data['name']}` with input: `{data['input']}`"
                }) + "\n")
            elif data["type"] == "tool_end":
                events.append(json.dumps({
                    "type": "tool_event",
                    "agent": f"{agent_name} ({model_key})",
                    "round": round_num,
                    "content": f"✅ **Tool output:**\n```\n{data['output']}\n```"
                }) + "\n")
                
        # --- LOG THE RESPONSE TO CONSOLE ---
        print(f"\n--- CONTINUOUS TOOL EVENTS ---", flush=True)
        for e in events: 
            print(e.strip(), flush=True)
        print(f"\n--- FINAL TEXT OUTPUT ---\n{full_reply.strip()}", flush=True)
        print(f"{'='*60}\n", flush=True)

        return full_reply.strip(), events
    except Exception as e:
        error_msg = f"Error calling model {model_key}: {str(e)}"
        print(f"\n--- CRASH/ERROR ---\n{error_msg}\n{'='*60}\n", flush=True)
        return error_msg, []


def _try_manual_tool(reply: str, agent_name: str, round_num: int):
    """
    Scan reply for a bare JSON tool call block (`{"name": ..., "arguments": ...}`).
    If found, execute the tool and return (updated_reply, extra_events).
    """
    tool_map = {t.name: t for t in tools}
    extra_events = []

    start_idx = reply.find("{")
    end_idx = reply.rfind("}")
    if start_idx == -1 or end_idx == -1:
        return reply, extra_events

    tool_data = None
    current_end = end_idx
    while current_end > start_idx:
        try:
            tool_data = json.loads(reply[start_idx:current_end + 1])
            break
        except Exception:
            current_end = reply.rfind("}", start_idx, current_end)

    if not (isinstance(tool_data, dict) and "name" in tool_data and "arguments" in tool_data):
        return reply, extra_events

    try:
        t_name = tool_data.get("name")
        t_args = tool_data.get("arguments", tool_data)
        if "name" in t_args:
            del t_args["name"]

        # Guard: if the model passed a JSON schema dict instead of a scalar value,
        # extract the 'description' field as the actual argument value.
        def _coerce(v):
            if isinstance(v, dict) and ("type" in v or "description" in v):
                return v.get("description", str(v))
            return v
        t_args = {k: _coerce(v) for k, v in t_args.items()}

        if t_name not in tool_map:
            return reply, extra_events

        t_input_str = str(list(t_args.values())[0]) if t_args else ""
        extra_events.append(json.dumps({
            "type": "tool_event",
            "agent": agent_name,
            "round": round_num,
            "content": (
                f"🛠️ **{agent_name}** is manually running tool `{t_name}` with input:\n"
                f"```\n{t_input_str[:200]}\n```"
            ),
        }) + "\n")

        tool_result = tool_map[t_name].invoke(t_args)

        extra_events.append(json.dumps({
            "type": "tool_event",
            "agent": agent_name,
            "round": round_num,
            "content": f"✅ **Tool output:**\n```\n{str(tool_result)[:500]}\n```",
        }) + "\n")

        reply += f"\n\n*(Agent executed tool `{t_name}`. Output: {str(tool_result)[:500]})*"
    except Exception as pe:
        print("Failed to execute parsed tool:", pe)

    return reply, extra_events


# ---------------------------------------------------------------------------
# Pacing / HITL wait helper
# ---------------------------------------------------------------------------
def _pacing_wait(
    waiting_for_user: bool,
    convo_id: str,
    agent_name: str,
    round_num: int,
):
    """Block until user input arrives (if needed) or ~2.5 s have elapsed."""
    pacing_steps = 0
    max_pacing = 25  # × 0.1 s = 2.5 s

    while True:
        has_pause = any("PAUSE" in str(x).upper() for x in sim_interjections[convo_id])

        if waiting_for_user or has_pause:
            time.sleep(0.5)
            if sim_interjections[convo_id]:
                waiting_for_user = False
                sim_interjections[convo_id] = [
                    x for x in sim_interjections[convo_id]
                    if "PAUSE" not in str(x).upper()
                ]
                break
            continue
        else:
            time.sleep(0.1)
            pacing_steps += 1
            if pacing_steps >= max_pacing or sim_interjections[convo_id]:
                break


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@orchestrator_bp.route("/interject", methods=["POST"])
def interject():
    """Push a user message into a running simulation by convo_id."""
    data = request.get_json(force=True, silent=True) or {}
    convo_id = data.get("convo_id")
    text = data.get("text")
    if convo_id and text:
        sim_interjections[convo_id].append(text)
    return {"status": "ok"}


@orchestrator_bp.route("/orchestrate", methods=["POST"])
def orchestrate():
    """Parse a scenario prompt, create agents, and run a multi-round conversation."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return {"error": "Request body must be valid JSON"}, 400

    scenario_prompt = data.get("prompt", "")
    convo_id = data.get("convo_id", "default_sim")

    def generate():
        # ------------------------------------------------------------------ #
        # Step 1: Parse the scenario
        # ------------------------------------------------------------------ #
        yield json.dumps({"type": "status", "content": "🎭 Parsing scenario..."}) + "\n"

        config, raw = _parse_scenario(scenario_prompt)
        print(f"[orchestrate] LLM parse output: {raw}", flush=True)

        if not config or len(config.get("agents", [])) < 2:
            yield json.dumps({
                "type": "error",
                "content": (
                    f"Could not identify at least 2 agents. Try using ALL CAPS names "
                    f"(e.g. ALICE, BOB).\n\nLLM returned:\n```\n{raw}\n```"
                ),
            }) + "\n"
            return

        agents = config.get("agents", [])
        
        # User requested to forcefully assign Llama to tool-using agents
        for a in agents:
            if a.get("name", "").upper() in TOOL_ROLES:
                a["model"] = "llama3.1:8b"
                
        rounds = config.get("rounds", 5)

        if not isinstance(rounds, int):
            rounds = 5
        scenario = config.get("scenario", "")
        opener = config.get("opener", "Let's get started!")

        if len(agents) < 2:
            yield json.dumps({
                "type": "error",
                "content": "Need at least 2 agents for a multi-agent scenario.",
            }) + "\n"
            return

        agent_names = [a["name"] for a in agents]

        yield json.dumps({
            "type": "status",
            "content": (
                f"🎭 Setting up **{len(agents)} agents**: {', '.join(agent_names)}\n\n"
                f"**Scenario:** {scenario}\n\n"
                "*Simulation runs until all agents agree the conclusion is met.*"
            ),
        }) + "\n"

        # ------------------------------------------------------------------ #
        # Step 2: Multi-agent loop
        # ------------------------------------------------------------------ #
        conversation_history: list[dict] = []
        approved = False
        round_num = 1
        last_replies: dict[str, str] = {}
        WRAP_UP_PHRASES = [
            "agree to disagree", "thank you for this discussion",
            "thanks for the conversation", "no further points", "was a pleasure",
            "end of discussion", "nothing more to add", "my final word",
            "let's politely end", "completely agree with you",
            "couldn't agree more", "we are in agreement", "i agree with you",
        ]

        while True:
            if approved:
                break

            conclusions_this_round = 0

            for i, agent in enumerate(agents):
                if approved:
                    break

                system, agent_name, is_tool_agent = _build_agent_system(agent, agents, scenario, i)

                # ---- Flush pending user interjections ---- #
                while sim_interjections[convo_id]:
                    user_text = sim_interjections[convo_id].pop(0)
                    conversation_history.append(
                        {"agent": "HUMAN", "text": user_text, "round": round_num}
                    )
                    yield json.dumps({
                        "type": "agent_message",
                        "agent": "🧑‍💻 YOU",
                        "round": round_num,
                        "content": user_text,
                    }) + "\n"

                # ---- Generate reply via Dispatch ----
                if not conversation_history:
                    turn_instruction = f"SYSTEM INSTRUCTION: You are the first to speak. Kick off the scenario by making your opening statement. Provide any code, data, or context required by your {agent.get('personality')} persona to start the task."
                else:
                    turn_instruction = f"SYSTEM INSTRUCTION: It is your turn to speak. React to the conversation history as {agent_name} based on your {agent.get('personality')} persona. Do not offer to help like an AI assistant. Just play the role."

                model_key = agent.get("model", "llama3.1:8b")
                reply, dispatch_events = _dispatch_agent_turn(
                    agent_name, model_key, system, conversation_history, round_num, turn_instruction, convo_id
                )
                for event in dispatch_events:
                    yield event

                # Catch unformatted bare JSON tool calls that bypassed worker XML parsing
                reply, extra_events = _try_manual_tool(reply, agent_name, round_num)
                for event in extra_events:
                    yield event

                # Strip self-prefix (e.g. "[ALICE]: " or "ALICE: ")
                for prefix in (f"[{agent_name}]:", f"{agent_name}:"):
                    if reply.startswith(prefix):
                        reply = reply[len(prefix):].strip()
                        break

                # ---- Conclusion detection ---- #
                is_concluding = "[CONCLUSION_MET]" in reply.upper()
                if not is_concluding and any(p in reply.lower() for p in WRAP_UP_PHRASES):
                    is_concluding = True
                    reply += " [CONCLUSION_MET]"

                prev_reply = last_replies.get(agent_name, "")
                if (
                    not is_concluding
                    and prev_reply
                    and len(reply) > 10
                    and (reply in prev_reply or prev_reply in reply)
                ):
                    is_concluding = True
                    reply += " [CONCLUSION_MET] (Repetition Detected)"

                last_replies[agent_name] = reply
                if is_concluding:
                    conclusions_this_round += 1

                conversation_history.append({"agent": agent_name, "text": reply, "round": round_num})

                yield json.dumps({
                    "type": "agent_message",
                    "agent": agent_name,
                    "model": model_key,
                    "persona": agent.get("personality", "Participant"),
                    "round": round_num,
                    "content": reply,
                }) + "\n"

                if "APPROVED" in reply.upper():
                    approved = True
                    break

                # ---- HITL pacing ---- #
                # We only pause for the human if the agent explicitly tags @USER
                waiting_for_user = "@USER" in reply.upper()
                if waiting_for_user:
                    yield json.dumps({
                        "type": "status",
                        "content": f"⏸️ **{agent_name}** is waiting for your input. Use the interject box above to reply!",
                    }) + "\n"

                _pacing_wait(waiting_for_user, convo_id, agent_name, round_num)

            # Did every agent conclude this round?
            if conclusions_this_round == len(agents):
                yield json.dumps({
                    "type": "status",
                    "content": "\n\n🛑 **All agents have signaled the conclusion is met.**",
                }) + "\n"
                break

            round_num += 1

        if approved:
            yield json.dumps({
                "type": "status",
                "content": (
                    f"\n\n✅ **{agent_names[-1]} APPROVED the result!** "
                    f"Simulation ended after {round_num} rounds."
                ),
            }) + "\n"
        else:
            yield json.dumps({
                "type": "status",
                "content": (
                    f"\n\n🏁 **Simulation complete!** Natural conclusion reached after "
                    f"{round_num} rounds between {', '.join(agent_names)}."
                ),
            }) + "\n"
    return Response(generate(), mimetype="application/x-ndjson")


def run_agent_task(task_id, convo_id, agent_executor, model_name, agent_id, user_input=None):

    """Worker function to run the agent logic in a background thread."""
    db_client.update_task(task_id, "RUNNING")
    try:
        from librarian import run_librarian_workflow
        from developer.developer import run_developer_workflow
        
        if agent_id == "LIBRARIAN":
            # Initialize state if missing (starting fresh from Coding Mode)
            if not db_client.get_full_state(convo_id):
                print(f"[ORCHESTRATOR][{convo_id}] Initializing state for new coding session.", flush=True)
                db_client.upsert_state(
                    convo_id=convo_id,
                    state_json={"state": {"requirement": user_input}},
                    status="ELICITATION_COMPLETE",
                    agent_id="SYSTEM",
                    model_name=model_name
                )
                # Ensure the first user message is logged
                db_client.add_message(convo_id, "USER", user_input, receiver="LIBRARIAN")

            final_response = run_librarian_workflow(
                convo_id=convo_id,
                agent_executor=agent_executor,
                model_name=model_name,
                user_input=user_input
            )
            db_client.add_message(convo_id, "LIBRARIAN", final_response, receiver="DEVELOPER")
        elif agent_id == "DEVELOPER":
            final_response = run_developer_workflow(
                convo_id=convo_id,
                model_name=model_name,
                user_input=user_input
            )
            db_client.add_message(convo_id, "DEVELOPER", final_response, receiver="REVIEWER" if "READY_FOR_REVIEW" in db_client.get_status(convo_id) else "USER")
        elif agent_id == "REVIEWER":
            from reviewer import run_reviewer_task
            res = run_reviewer_task(convo_id=convo_id, model_name=model_name)
            final_response = res.get("message", str(res))
            if "comments" in res:
                final_response += f"\n\n{res['comments']}"
            db_client.add_message(convo_id, "REVIEWER", final_response, receiver="DEVELOPER" if res.get("status") == "COMMENTS_POSTED" else "USER")
        # --- AUTO-CHAINING LOGIC ---
        new_status = db_client.get_status(convo_id)
        
        # 1. Developer -> Reviewer (Autonmous Testing)
        if agent_id == "DEVELOPER" and new_status == "READY_FOR_REVIEW":
            print(f"[ORCHESTRATOR][{convo_id}] Auto-triggering REVIEWER for testing...", flush=True)
            from reviewer import run_reviewer_task
            res = run_reviewer_task(convo_id=convo_id, model_name=model_name)
            final_response += f"\n\n--- REVIEWER RESULTS ---\n{res.get('message', '')}"
            if "comments" in res:
                final_response += f"\n\n{res['comments']}"
            
            # Re-check status after reviewer run
            new_status = db_client.get_status(convo_id)

        # 2. Reviewer -> Developer (Autonomous Refactor)
        if new_status == "COMMENTS_POSTED":
             # We stop here to let the user see the failure or we could auto-loop.
             # User said: "The coder should address the comments and put it back to READY_FOR_REVIEW"
             # Let's auto-trigger once to show the loop.
             print(f"[ORCHESTRATOR][{convo_id}] Auto-triggering DEVELOPER for refactor...", flush=True)
             refactor_res = run_developer_workflow(convo_id=convo_id, model_name=model_name)
             final_response += f"\n\n--- REFACTORING (Attempt 1) ---\n{refactor_res}"



        db_client.update_task(task_id, "COMPLETED", result_json={"response": final_response})
    except Exception as e:
        print(f"[TASK ERROR][{task_id}]: {e}")
        db_client.update_task(task_id, "FAILED", error_message=str(e))

@orchestrator_bp.route("/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Returns the current status and result of a background task."""
    task = db_client.get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@orchestrator_bp.route("/invoke", methods=["POST"])
def invoke():
    """
    Generic Raw LLM usage API call.
    Expects: {"agentId": str, "model": str, "prompt": str, "convoId": str}
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return {"error": "Request body must be valid JSON"}, 400

    try:
        req_model = InvokeRequest(**data)
    except ValidationError as e:
        return {"error": e.errors()}, 400

    # 1. SPECIALIZED AGENTS (Background Thread)
    if req_model.agentId in [AgentID.LIBRARIAN, AgentID.DEVELOPER, AgentID.REVIEWER]:
        try:
            # Create a task record
            task_id = db_client.create_task(req_model.convoId, req_model.agentId.value)
            
            # Start background thread
            thread = threading.Thread(
                target=run_agent_task,
                args=(task_id, req_model.convoId, agent_executor, req_model.model.value, req_model.agentId.value, req_model.prompt)
            )



            thread.daemon = True
            thread.start()
            
            return {
                "task_id": task_id,
                "status": "PENDING",
                "message": "Research task started asynchronously. Poll /task/<task_id> for results."
            }, 202
        except Exception as e:
            return {"error": f"Failed to start Librarian task: {str(e)}"}, 500

    # Determine System Prompt for other agents
    if req_model.agentId == AgentID.ELICITOR:
        from elicitor import prepare_elicitor_context
        elicitor_context = prepare_elicitor_context(req_model.convoId, req_model.prompt)
        
        # If already complete, skip LLM
        if elicitor_context.get("mode") == "complete":
            return {
                "agentId": req_model.agentId.value,
                "model": req_model.model.value,
                "convoId": req_model.convoId,
                "response": "Elicitation is already complete! Requirement: " + elicitor_context.get("requirement")
            }
        
        system_msg = SystemPrompt.get_prompt(req_model.agentId, elicitor_context)
    else:
        system_msg = SystemPrompt.get_prompt(req_model.agentId)

    # 2. RAW API call for other agents (Elicitor, etc.)
    # This bypasses the worker agent's /chat endpoint (and its tool instructions)
    payload = {
        "model": req_model.model.value,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": req_model.prompt}
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384
        }
    }

    try:
        print(f"[INVOKE] Raw call to Ollama for model: {req_model.model.value}", flush=True)
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        
        raw_data = resp.json()
        result_text = raw_data.get("message", {}).get("content", "").strip()

        # ELICITOR specific logic: Delegate to elicitor.py
        final_response = result_text
        if req_model.agentId == AgentID.ELICITOR:
            final_response = process_elicitor_response(
                convo_id=req_model.convoId,
                model_name=req_model.model.value,
                agent_id=req_model.agentId.value,
                result_text=result_text,
                mode=elicitor_context.get("mode", "initial")
            )

        return {
            "agentId": req_model.agentId.value,
            "model": req_model.model.value,
            "convoId": req_model.convoId,
            "response": final_response
        }
    except Exception as e:
        return {"error": str(e)}, 500


# ---------------------------------------------------------------------------
# Coding Mode API Routes
# ---------------------------------------------------------------------------

@orchestrator_bp.route("/coding/conversations", methods=["GET"])
def list_coding_conversations():
    """Lists all conversation IDs that have messages (effectively active sessions)."""
    try:
        convoids = db_client.list_all_convo_ids()
        results = []
        for cid in convoids:
            # For now, just return them all. In a real app, filter for 'code' type.
            results.append({
                "id": cid,
                "status": "active",
                "updated_at": datetime.now().isoformat()
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/coding/conversation/<convo_id>", methods=["GET"])
def get_coding_messages(convo_id):
    """Fetches all messages and thoughts for a specific coding conversation."""
    try:
        messages = db_client.get_messages(convo_id)
        return jsonify({"messages": messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/coding/sandbox/<convo_id>/files", methods=["GET"])
def list_sandbox_files(convo_id):
    """Lists all files in the sandbox for a specific conversation."""
    from developer.developer import _get_sandbox_path
    try:
        sandbox_path = _get_sandbox_path(convo_id)
        files = []
        for root, _, filenames in os.walk(sandbox_path):
            for f in filenames:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, sandbox_path)
                files.append({
                    "name": f,
                    "path": rel_path
                })
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/coding/sandbox/<convo_id>/file", methods=["GET", "POST"])
def manage_sandbox_file(convo_id):
    """Reads or writes a specific file in the sandbox."""
    from developer.developer import _get_sandbox_path
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Path parameter is required"}), 400

    try:
        sandbox_path = _get_sandbox_path(convo_id)
        file_path = os.path.join(sandbox_path, path)

        if request.method == "GET":
            if not os.path.exists(file_path):
                return jsonify({"error": "File not found"}), 404
            with open(file_path, "r", encoding="utf-8") as f:
                return jsonify({"content": f.read()})
        else:
            data = request.json
            content = data.get("content", "")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
