import os
import json
import time
import requests as req
from flask import Flask, request, Response
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from tools.my_tools import get_all_tools

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

app = Flask(__name__)

# ---- Wait for Ollama & pull model ----
def startup():
    print(f"[{MODEL_NAME}] Waiting for Ollama...", flush=True)
    for _ in range(30):
        try:
            res = req.get(OLLAMA_BASE_URL)
            if res.status_code == 200:
                print(f"[{MODEL_NAME}] Ollama is ready.", flush=True)
                break
        except req.exceptions.ConnectionError:
            pass
        time.sleep(2)

    print(f"[{MODEL_NAME}] Pulling model...", flush=True)
    response = req.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": MODEL_NAME}, stream=True)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'status' in data:
                    print(data['status'], flush=True)
    print(f"[{MODEL_NAME}] Ready to serve.", flush=True)

# ---- Build agent ----
tools = get_all_tools()
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
agent_executor = create_react_agent(llm, tools)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.route("/chat", methods=["POST"])
def chat():
    """Accepts a list of messages, streams back JSON-lines events."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return {"error": "Request body must be valid JSON"}, 400
    messages = data.get("messages", [])

    system_prompt = (
        f"You are a helpful AI assistant powered by {MODEL_NAME}. "
        "\n\nCRITICAL RULE: DO NOT USE TOOLS FOR GREETINGS OR CASUAL CONVERSATION. "
        "If a user says 'Hi', 'Hello', or 'How are you?', you MUST respond with text ONLY. "
        "\n\nDETAILED TOOL RULES:\n"
        "- ONLY use tools when the user's request GENUINELY requires external information you do not already know.\n"
        "- For greetings, opinions, creative writing, general knowledge, coding questions, or math — respond directly. DO NOT call any tools.\n"
        "- Use 'robust_internet_search' ONLY for live data or current news.\n"
        "- Use 'fetch_website_content' ONLY for specific URLs provided by the user.\n"
        "- Use 'wikipedia_search' ONLY for factual encyclopedia lookups.\n"
        "- Use 'execute_python_code' ONLY when explicitly asked to run or test code.\n"
        "- Use 'get_current_time' ONLY when the user explicitly asks for the current time or date.\n"
        "\nEXAMPLE OF WRONG BEHAVIOR:\n"
        "User: 'Hi'\n"
        "Assistant: [Calling get_current_time] <--- WRONG! Respond with 'Hello! How can I help you?' instead."
    )

    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    def generate():
        # Use LangGraph's synchronous .stream() API — no asyncio, no event loops,
        # no httpx client lifecycle issues across Flask threads.
        # stream_mode="messages" yields (message_chunk, metadata) tuples:
        #   node="agent" → AIMessageChunk tokens (or tool-call intent)
        #   node="tools" → ToolMessage with the tool result
        try:
            for msg_chunk, metadata in agent_executor.stream(
                {"messages": lc_messages},
                stream_mode="messages"
            ):
                node = metadata.get("langgraph_node", "")

                if node == "tools":
                    # Tool result coming back from the tool executor node
                    content = getattr(msg_chunk, "content", "")
                    if content:
                        yield json.dumps({"type": "tool_end", "output": str(content)[:500]}) + "\n"

                elif node == "agent":
                    # Check if the AI is initiating a tool call
                    tool_call_chunks = getattr(msg_chunk, "tool_call_chunks", [])
                    if tool_call_chunks:
                        for tc in tool_call_chunks:
                            if tc.get("name"):  # name only present on the first chunk
                                yield json.dumps({"type": "tool_start", "name": tc["name"], "input": ""}) + "\n"
                    else:
                        # Regular streamed text token
                        content = getattr(msg_chunk, "content", "")
                        if isinstance(content, str) and content:
                            yield json.dumps({"type": "token", "content": content}) + "\n"

        except Exception as e:
            if "does not support tools" in str(e):
                # Sync fallback for models that reject tool binding (e.g. deepseek-r1)
                for chunk in llm.stream(lc_messages):
                    if chunk.content:
                        yield json.dumps({"type": "token", "content": chunk.content}) + "\n"
            else:
                yield json.dumps({"type": "token", "content": f"\n\n[Agent Error: {str(e)}]"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")

# ---- MULTI-AGENT ORCHESTRATION ----

import re

PARSE_PROMPT = """Extract agent details from this scenario as JSON. Output ONLY the JSON object, nothing else.

Example input: "A debate between ALICE and BOB about cats vs dogs for 3 rounds. ALICE loves cats and BOB loves dogs."
Example output: {"agents": [{"name": "ALICE", "personality": "loves cats, argues passionately for cats"}, {"name": "BOB", "personality": "loves dogs, argues passionately for dogs"}], "rounds": 3, "scenario": "debate about cats vs dogs", "opener": "Let me tell you why cats are superior!"}

Now extract from this scenario: """

def parse_scenario_fallback(prompt):
    """Regex-based fallback parser when LLM fails to produce valid JSON."""
    # Extract agent names (look for ALL CAPS names or names after "agents"/"between")
    name_pattern = r'\b([A-Z]{2,}[A-Z]*)\b'
    names = list(set(re.findall(name_pattern, prompt)))
    # Filter out common words
    stop_words = {"THE", "AND", "FOR", "WANT", "BETWEEN", "EACH", "OTHER", "ROUND", "ROUNDS", "YEAR", "OLD", "WHO"}
    names = [n for n in names if n not in stop_words]
    
    # Extract rounds
    rounds_match = re.search(r'(\d+)\s*round', prompt, re.IGNORECASE)
    rounds = int(rounds_match.group(1)) if rounds_match else 5
    
    if len(names) < 2:
        return None
    
    # Build agent configs from context clues in the prompt
    agents = []
    for name in names:
        # Try to find personality description near the name
        personality_match = re.search(rf'{name}\s+(?:is\s+)?(.{{10,60}}?)(?:\.|,|and\s+[A-Z]|$)', prompt, re.IGNORECASE)
        personality = personality_match.group(1).strip() if personality_match else f"a participant named {name}"
        agents.append({"name": name, "personality": personality})
    
    return {
        "agents": agents,
        "rounds": rounds,
        "scenario": prompt,
        "opener": f"Alright, let's get this started!"
    }

from collections import defaultdict
sim_interjections = defaultdict(list)

@app.route("/interject", methods=["POST"])
def interject():
    """Allows a user to push a message into a running simulation."""
    data = request.get_json(force=True, silent=True) or {}
    convo_id = data.get("convo_id")
    text = data.get("text")
    if convo_id and text:
        sim_interjections[convo_id].append(text)
    return {"status": "ok"}

@app.route("/orchestrate", methods=["POST"])
def orchestrate():
    """Parse a scenario prompt, create agents, and run a multi-round conversation."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return {"error": "Request body must be valid JSON"}, 400
    
    scenario_prompt = data.get("prompt", "")
    convo_id = data.get("convo_id", "default_sim")
    
    def generate():
        # Step 1: Parse the scenario using the LLM
        yield json.dumps({"type": "status", "content": "🎭 Parsing scenario..."}) + "\n"
        
        parse_response = llm.invoke([HumanMessage(content=PARSE_PROMPT + scenario_prompt)])
        raw = parse_response.content.strip()
        print(f"[orchestrate] LLM parse output: {raw}", flush=True)
        
        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        
        config = None
        try:
            config = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[orchestrate] JSON parse failed, trying fallback...", flush=True)
        
        # Fallback: regex-based parsing
        if not config or len(config.get("agents", [])) < 2:
            print(f"[orchestrate] Using regex fallback parser", flush=True)
            config = parse_scenario_fallback(scenario_prompt)
        
        if not config or len(config.get("agents", [])) < 2:
            yield json.dumps({"type": "error", "content": f"Could not identify at least 2 agents. Try using ALL CAPS names (e.g. ALICE, BOB).\n\nLLM returned:\n```\n{raw}\n```"}) + "\n"
            return
        
        agents = config.get("agents", [])
        rounds = config.get("rounds", 5)
        scenario = config.get("scenario", "")
        opener = config.get("opener", "Let's get started!")
        
        if len(agents) < 2:
            yield json.dumps({"type": "error", "content": "Need at least 2 agents for a multi-agent scenario."}) + "\n"
            return
        
        agent_names = [a["name"] for a in agents]
        is_interactive = any(word in scenario.lower() for word in ["ask", "question", "input", "interact", "opinion", "debate"])
        
        # Boost rounds for interactive sessions if not explicitly low
        if is_interactive and rounds < 3:
            rounds = 10
            
        yield json.dumps({"type": "status", "content": f"🎭 Setting up **{len(agents)} agents**: {', '.join(agent_names)} for up to **{rounds} rounds**\n\n**Scenario:** {scenario}"}) + "\n"
        
        # If interactive, wait for the user to provide the first question/topic
        if is_interactive:
            yield json.dumps({"type": "status", "content": "👋 **Interaction required!** Please provide your first question or topic using the interject box."}) + "\n"
            while not sim_interjections[convo_id]:
                time.sleep(0.5)
        
        # Step 2: Detect agent roles for tool-aware prompting
        TOOL_ROLES = {"REVIEWER", "TESTER", "QA", "VERIFIER", "VALIDATOR"}
        
        # Step 3: Run the multi-agent loop
        conversation_history = []
        approved = False
        
        for round_num in range(1, rounds + 1):
            if approved:
                break
                
            for i, agent in enumerate(agents):
                if approved:
                    break
                    
                agent_name = agent.get("name", f"Agent_{i+1}")
                if "personality" in agent:
                    personality = agent["personality"]
                else:
                    details = ", ".join([f"{k}: {v}" for k, v in agent.items() if k != "name"])
                    personality = details if details else f"A participant in the scenario named {agent_name}."
                other_agents = [a.get("name", f"Agent_{j+1}") for j, a in enumerate(agents) if j != i]
                is_tool_agent = agent_name.upper() in TOOL_ROLES
                
                # Build role-aware system prompt
                system = (
                    f"You are {agent_name}. {personality}. "
                    f"You are in a scenario: {scenario}. "
                    f"You are interacting with: {', '.join(other_agents)}. "
                    f"This is round {round_num} of {rounds}. "
                    f"Respond ONLY as {agent_name} in the FIRST PERSON. Do NOT describe your actions or narrate. "
                    f"Speak directly to others. Stay in character even if you disagree with them."
                )
                
                if is_tool_agent:
                    system += (
                        f"\n\nIMPORTANT: You have access to the 'execute_python_code' tool. "
                        f"When code is presented to you for review, you MUST use the execute_python_code tool to actually run it and verify the output is correct. "
                        f"If the code runs correctly and produces the expected output, include the word APPROVED in your response. "
                        f"If it fails or produces wrong output, explain what went wrong and ask for fixes."
                    )
                
                # --- Process pending user interjections BEFORE this agent generates ---
                while sim_interjections[convo_id]:
                    user_text = sim_interjections[convo_id].pop(0)
                    conversation_history.append({"agent": "HUMAN", "text": user_text, "round": round_num})
                    yield json.dumps({
                        "type": "agent_message",
                        "agent": "🧑‍💻 YOU",
                        "round": round_num,
                        "content": user_text
                    }) + "\n"
                
                lc_msgs = [SystemMessage(content=system)]
                
                if not conversation_history:
                    lc_msgs.append(HumanMessage(content=f"Start the scenario. Your opening line: {opener}"))
                else:
                    for entry in conversation_history:
                        if entry["agent"] == agent_name:
                            lc_msgs.append(AIMessage(content=entry["text"]))
                        elif entry["agent"] == "HUMAN":
                            lc_msgs.append(HumanMessage(content=f"***[OVERRIDE INSTRUCTION FROM HUMAN USER]***: {entry['text']}\n(Pay attention and adapt your response to this)"))
                        else:
                            lc_msgs.append(HumanMessage(content=f"[{entry['agent']}]: {entry['text']}"))
                    lc_msgs.append(HumanMessage(content=f"It's your turn. Respond as {agent_name}."))
                
                # Use agent_executor (with tools) for tool-enabled agents, raw LLM for others
                if is_tool_agent:
                    result = agent_executor.invoke({"messages": lc_msgs})
                    all_msgs = result["messages"]
                    input_count = len(lc_msgs)
                    new_msgs = all_msgs[input_count:]
                    
                    # Extract tool events and final reply
                    reply_parts = []
                    for msg in new_msgs:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield json.dumps({
                                    "type": "tool_event",
                                    "agent": agent_name,
                                    "round": round_num,
                                    "content": f"🛠️ **{agent_name}** is running tool `{tc['name']}` with input:\n```\n{tc['args'].get('code', tc['args'])}\n```"
                                }) + "\n"
                        elif msg.type == 'tool':
                            tool_output = msg.content[:500]
                            yield json.dumps({
                                "type": "tool_event",
                                "agent": agent_name,
                                "round": round_num,
                                "content": f"✅ **Tool output:**\n```\n{tool_output}\n```"
                            }) + "\n"
                        elif msg.type == 'ai' and msg.content:
                            reply_parts.append(msg.content)
                    
                    reply = reply_parts[-1].strip() if reply_parts else "(No response)"
                else:
                    response = llm.invoke(lc_msgs)
                    reply = response.content.strip()
                
                conversation_history.append({"agent": agent_name, "text": reply, "round": round_num})
                
                yield json.dumps({
                    "type": "agent_message",
                    "agent": agent_name,
                    "round": round_num,
                    "content": reply
                }) + "\n"
                
                # Check for approval
                if "APPROVED" in reply.upper():
                    approved = True
                    break

                # --- HITL Check: Should we wait for the user? ---
                waiting_for_user = "@USER" in reply.upper() or "WAITING" in reply.upper()
                
                if waiting_for_user:
                    yield json.dumps({"type": "status", "content": f"⏸️ **{agent_name}** is waiting for your input. Use the interject box above to reply!"}) + "\n"
                
                # Intelligent Pacing & Wait Logic
                # Loop for a while to allow user to read, but BLOCK if we are waiting_for_user
                pacing_steps = 0
                max_pacing = 25 # 2.5 seconds default pacing
                
                while True:
                    # If user interjects with "PAUSE", we enter a wait loop
                    has_pause = any("PAUSE" in str(x).upper() for x in sim_interjections[convo_id])
                    
                    if waiting_for_user or has_pause:
                        # Hard Block: Wait for ANY new input
                        time.sleep(0.5)
                        if sim_interjections[convo_id]:
                             # New input received! Break the block.
                             waiting_for_user = False
                             # Clear 'PAUSE' if it was there
                             sim_interjections[convo_id] = [x for x in sim_interjections[convo_id] if "PAUSE" not in str(x).upper()]
                             break
                        continue
                    else:
                        # Standard Pacing: Wait ~2.5s then continue
                        time.sleep(0.1)
                        pacing_steps += 1
                        if pacing_steps >= max_pacing or sim_interjections[convo_id]:
                            break
        
        if approved:
            yield json.dumps({"type": "status", "content": f"\n\n✅ **{agent_names[-1]} APPROVED the result!** Simulation ended after {round_num} rounds."}) + "\n"
        else:
            yield json.dumps({"type": "status", "content": f"\n\n🏁 **Simulation complete!** {rounds} rounds between {', '.join(agent_names)}."}) + "\n"
    
    return Response(generate(), mimetype="application/x-ndjson")

if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, threaded=True)
