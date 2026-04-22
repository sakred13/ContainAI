"""
worker_agent/routes.py
----------------------
Flask Blueprint for the /chat endpoint.

Tool-calling strategy per model family:
  • NATIVE path  — LangGraph create_react_agent + bind_tools (Ollama native API).
                    Works for: llama3.x, mistral, gemma3, qwen2.5-coder
  • MANUAL path  — text-based <tool_call> / <observation> loop.
                    Used for: deepseek-r1 (reasoning model, no native binding),
                               dolphin-mistral (uncensored fine-tune, unreliable binding)
"""

import json
import re

from flask import Blueprint, request, Response
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from common.llm_client import llm, agent_executor, tools, MODEL_NAME

worker_bp = Blueprint("worker", __name__)

# Models that do NOT support Ollama's native tool-binding API reliably.
# These are routed directly to the text-based <tool_call> fallback.
MANUAL_TOOL_MODELS = {
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "dolphin-mistral",
    "mistral",
    "gemma3:4b",
    "qwen2.5-coder",
    "llama3.1:8b",
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_system_prompt(convo_id: str = "default") -> str:
    return (
        f"You are a helpful AI assistant powered by {MODEL_NAME}. "
        f"\n\nCURRENT CONVERSATION ID: {convo_id}"
        "\n\nCRITICAL RULE: DO NOT USE TOOLS FOR GREETINGS OR CASUAL CONVERSATION. "
        "If a user says 'Hi', 'Hello', or 'How are you?', you MUST respond with text ONLY. "
        "\n\nDETAILED TOOL RULES:\n"
        "- ONLY use tools when the user's request GENUINELY requires external information you do not already know.\n"
        "When responding, ask yourself if a tool is needed. If not, don't use a tool - respond with what you know.\n"
        "- For greetings, opinions, creative writing, general knowledge, coding questions, or math — respond directly. DO NOT call any tools.\n"
        "- Use 'internet_search' for all web lookups. By default, it performs a quick search. Set 'deep_search=True' ONLY for detailed research, full article content, or when looking up private individuals likely to require deep verification. ALWAYS pass the current conversation ID.\n"
        "- Use 'fetch_website_content' ONLY for a specific URL explicitly provided by the user.\n"
        "- Use 'wikipedia_search' ONLY for widely recognized subjects: famous people, historical events, scientific concepts, countries, or established organizations. Never use it for private individuals or regular people.\n"
        "- Use 'execute_python_code' ONLY when explicitly asked to run or test code.\n"
        "- Use 'get_current_time' ONLY when the user explicitly asks for the current time or date. DO NOT look up news or events unless specifically requested.\n"
        "\nFALLBACK RULE: If 'internet_search' returns no results, it will automatically fallback to a deep search. You HONESTLY ADMIT when information cannot be found.\n"
        "PERSON LOOKUP RULE: If asked 'Who is [person]?' and they are not globally famous, skip wikipedia_search and go straight to 'internet_search' with 'deep_search=True'. Summarize findings (job, location, school) directly.\n"
        "NO NARRATION RULE: NEVER say 'I will search', 'Let me look that up', 'Performing a search', or any similar phrase. If you need a tool, you MUST output the <tool_call> block IMMEDIATELY and output NOTHING ELSE. No text before it, no explanation after it. Just the tool call.\n"
        "STRICT FOCUS RULE: If the user asks a simple question (e.g., 'What time is it?', 'What timezone is X in?'), and you retrieve the answer, STOP. Do NOT perform extra unprompted searches for news, weather, or events to 'add context'. Provide the answer concisely.\n"
        "NO HALLUCINATION RULE: If your tools return 'No search results found' or other errors, you MUST HONESTLY ADMIT that you cannot find the information. NEVER guess, NEVER provide 'estimates', and NEVER say 'according to various sources' if your tools found nothing. It is better to say 'I don't know' than to provide random or potentially incorrect data.\n"
        "\nEXAMPLES OF WRONG BEHAVIOR:\n"
        "User: 'Who is Saketh Poduvu?'\n"
        "Assistant: 'I am not sure, let me search for him.' <tool_call>... <--- WRONG! The introductory text is banned. Use the tool call ONLY.\n"
        "User: 'How far is St. Louis from Kansas City?'\n"
        "Tool Observation: 'No search results found.'\n"
        "Assistant: 'It is approximately 310 miles.' <--- WRONG! You must admit you found no information.\n"
        "User: 'Hi'\n"
        "Assistant: [Calling get_current_time] <--- WRONG! Respond with 'Hello! How can I help you?' instead."
    )


def _lc_messages_from_request(messages: list, convo_id: str = "default") -> list:
    """Convert raw request message dicts to LangChain message objects."""
    lc_messages = []
    
    # Check if a custom system message was provided
    has_system = any(msg["role"] == "system" for msg in messages)
    
    if not has_system:
        lc_messages.append(SystemMessage(content=_build_system_prompt(convo_id)))
        
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"], name=msg.get("name")))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"], name=msg.get("name")))

    return lc_messages


def _native_stream(lc_messages: list):
    """
    Attempt to stream using LangGraph's native tool-binding.
    Yields JSON-lines events.

    tool_call_chunks arrive in multiple pieces: the first chunk carries the
    tool name, subsequent chunks carry the (JSON-encoded) arguments string.
    We accumulate both so we can emit tool_start with the complete input just
    before the tool result (tool_end) arrives.
    """
    # pending_tool: {"name": str, "args": str}  – filled while chunks arrive
    pending_tool: dict | None = None

    for msg_chunk, metadata in agent_executor.stream(
        {"messages": lc_messages},
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")

        if node == "tools":
            content = getattr(msg_chunk, "content", "")
            if content:
                # Emit tool_start now that we have the complete args string
                if pending_tool:
                    try:
                        # args is a JSON string like '{"query": "..."}'
                        args_obj = json.loads(pending_tool["args"])
                        input_str = str(list(args_obj.values())[0]) if args_obj else pending_tool["args"]
                    except Exception:
                        input_str = pending_tool["args"]
                    yield json.dumps({
                        "type": "tool_start",
                        "name": pending_tool["name"],
                        "input": input_str[:200],
                    }) + "\n"
                    pending_tool = None
                yield json.dumps({"type": "tool_end", "output": str(content)[:500]}) + "\n"

        elif node == "agent":
            tool_call_chunks = getattr(msg_chunk, "tool_call_chunks", [])
            if tool_call_chunks:
                for tc in tool_call_chunks:
                    if tc.get("name"):
                        # First chunk — begin accumulating
                        pending_tool = {"name": tc["name"], "args": tc.get("args", "")}
                    elif pending_tool and tc.get("args"):
                        # Subsequent chunks — append args fragment
                        pending_tool["args"] += tc["args"]
            else:
                content = getattr(msg_chunk, "content", "")
                if isinstance(content, str) and content:
                    yield json.dumps({"type": "token", "content": content}) + "\n"


def _manual_tool_stream(lc_messages: list, convo_id: str = "default"):
    """
    Text-based <tool_call> / <observation> fallback loop for models that
    reject native tool binding.
    Yields JSON-lines events.
    """
    tool_map = {t.name: t for t in tools}
    tool_desc = ""
    for t in tools:
        # Show args as {"arg_name": "<type>"} so the model knows to pass a plain value
        if hasattr(t, "args") and t.args:
            args_hint = ", ".join(
                f'"{k}": "<{v.get("type", "string")}>"'
                for k, v in t.args.items()
            )
            args_hint = "{" + args_hint + "}"
        else:
            args_hint = "{}"
        tool_desc += f"- {t.name}: {t.description} (arguments: {args_hint})\n"

    sys_msg = lc_messages[0].content
    sys_msg += (
        "\n\n[CRITICAL TOOL INSTRUCTIONS]\n"
        "Your environment uses text-based tool calling. Tools are strictly OPTIONAL. ONLY use a tool if you genuinely need external information.\n"
        "To use a tool, you MUST use this XML wrapper around the JSON:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"arg_name": "value"}}\n'
        "</tool_call>\n\n"
        "EXAMPLE OF CORRECT TOOL INVOCATION:\n"
        "<tool_call>\n"
        '{"name": "deep_internet_search", "arguments": {"query": "Search query here"}}\n'
        "</tool_call>\n\n"
        "CRITICAL: NEVER output raw JSON without the <tool_call> tags!\n"
        f"Available tools:\n{tool_desc}\n"
        "When you output this, STOP GENERATING. You will then receive an <observation> block. You can call tools multiple times if needed."
    )

    my_messages = [SystemMessage(content=sys_msg)] + lc_messages[1:]
    print(f"[WORKER] Calling LLM with {len(my_messages)} messages. Last msg: {my_messages[-1].content[:100]}...", flush=True)
    MAX_STEPS = 5

    for _step in range(MAX_STEPS):
        response_text = ""
        buffer = ""
        is_tool_call = False

        for chunk in llm.stream(my_messages):
            if not chunk.content:
                continue
            # Strip DeepSeek reasoning tokens before processing
            chunk_text = _THINK_RE.sub("", chunk.content)
            if not chunk_text:
                continue
            buffer += chunk_text
            response_text += chunk_text

            if "<tool_call>" in response_text:
                is_tool_call = True

            if not is_tool_call:
                if "<" in buffer:
                    # Keep everything from the first '<' in the buffer to check for tool calls
                    idx = buffer.find("<")
                    prefix = buffer[idx:]
                    if "<tool_call>".startswith(prefix.lower()):
                        if idx > 0:
                            yield json.dumps({"type": "token", "content": buffer[:idx]}) + "\n"
                            buffer = prefix
                    else:
                        yield json.dumps({"type": "token", "content": buffer}) + "\n"
                        buffer = ""
                else:
                    yield json.dumps({"type": "token", "content": buffer}) + "\n"
                    buffer = ""

        if buffer and not is_tool_call:
            yield json.dumps({"type": "token", "content": buffer}) + "\n"

        my_messages.append(AIMessage(content=response_text))

        if is_tool_call:
            try:
                start_idx = response_text.find("<tool_call>") + len("<tool_call>")
                end_idx = response_text.find("</tool_call>")
                tool_json_str = (
                    response_text[start_idx:end_idx].strip()
                    if end_idx != -1
                    else response_text[start_idx:].strip()
                )
                tool_json_str = tool_json_str.replace("```json", "").replace("```", "").strip()
                tool_data = json.loads(tool_json_str)

                t_name = tool_data.get("name")
                t_args = tool_data.get("arguments", tool_data)
                if "name" in t_args:
                    del t_args["name"]

                # Guard: if the model passed a JSON schema dict instead of a
                # scalar value (e.g. {"type": "string", "description": "..."}
                # instead of the actual string), extract the description field.
                def _coerce(v):
                    if isinstance(v, dict) and ("type" in v or "description" in v):
                        return v.get("description", str(v))
                    return v
                t_args = {k: _coerce(v) for k, v in t_args.items()}

                if t_name in tool_map:
                    # Auto-inject conversation_id if missing but the tool expects it
                    if convo_id and t_name == "deep_internet_search" and "conversation_id" not in t_args:
                        t_args["conversation_id"] = convo_id

                    t_input_str = str(list(t_args.values())[0]) if t_args else ""
                    yield json.dumps({"type": "tool_start", "name": t_name, "input": t_input_str[:200]}) + "\n"

                    tool_result = tool_map[t_name].invoke(t_args)
                    yield json.dumps({"type": "tool_end", "output": str(tool_result)[:500]}) + "\n"

                    my_messages.append(
                        HumanMessage(
                            content=f"<observation>\n{tool_result}\n</observation>\nIf you have the answer, output it directly to the user now. Do NOT call another tool unless absolutely necessary."
                        )
                    )
                    continue
                else:
                    my_messages.append(
                        HumanMessage(content=f"Error: Tool '{t_name}' not found. Check spelling.")
                    )
                    continue
            except Exception as parse_e:
                my_messages.append(
                    HumanMessage(content=f"Error parsing tool JSON: {parse_e}. Output exact format!")
                )
                continue
        else:
            break  # Done — no more tool calls


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@worker_bp.route("/chat", methods=["POST"])
def chat():
    """Accept a list of messages, stream back JSON-lines events."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return {"error": "Request body must be valid JSON"}, 400

    messages = data.get("messages", [])
    convo_id = data.get("convo_id")
    if not convo_id or convo_id in ["default", "unknown", "None", "<current conversation ID>"]:
        import uuid
        convo_id = str(uuid.uuid4())
    print(f"[WORKER] Incoming convo_id: {convo_id}", flush=True)
    lc_messages = _lc_messages_from_request(messages, convo_id)

    def generate():
        # Using manual tool stream uniformly for all models.
        # Ollama's native tool binding frequently injects JSON schema dicts
        # instead of scalar values, causing Pydantic to crash in the native path.
        yield from _manual_tool_stream(lc_messages, convo_id)

    return Response(generate(), mimetype="application/x-ndjson")
