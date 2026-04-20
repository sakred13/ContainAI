import gradio as gr
import requests
import json
import os
import uuid
import glob
import tempfile
from datetime import datetime

# ---- MODEL → AGENT CONTAINER MAP ----
MODEL_REGISTRY = {
    "llama3.2 — Fast & Balanced": os.getenv("AGENT_LLAMA_URL", "http://agent_llama:5000"),
    "llama3.1:8b — Versatile 8B": os.getenv("AGENT_LLAMA8B_URL", "http://agent_llama_8b:5000"),
    "gemma3:4b — Strict Safety": os.getenv("AGENT_GEMMA_URL", "http://agent_gemma:5000"),
    "qwen2.5-coder — Python Expert": os.getenv("AGENT_QWEN_URL", "http://agent_qwen:5000"),
    "dolphin-mistral — Uncensored & Cooperative": os.getenv("AGENT_DOLPHIN_URL", "http://agent_dolphin:5000"),
    "deepseek-r1:7b — Deep Reasoning Logic": os.getenv("AGENT_DEEPSEEK_URL", "http://agent_deepseek:5000"),
}

# ---- CONVERSATION STORAGE ----
CONVERSATIONS_DIR = os.getenv("CONVERSATIONS_DIR", "/data/conversations")
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

def new_conversation_id():
    return str(uuid.uuid4())[:8]

def save_conversation(convo_id, model, history, convo_type="chat"):
    """Save conversation to a JSON file."""
    filepath = os.path.join(CONVERSATIONS_DIR, f"{convo_id}.json")
    data = {
        "id": convo_id,
        "model": model,
        "type": convo_type,
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }
    for msg in history:
        data["messages"].append({"role": msg["role"], "content": msg["content"]})
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_conversation(convo_id):
    """Load a conversation from its JSON file."""
    filepath = os.path.join(CONVERSATIONS_DIR, f"{convo_id}.json")
    if not os.path.exists(filepath):
        return None, None, "chat", []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = [{"role": m["role"], "content": m["content"]} for m in data.get("messages", [])]
    return data.get("id"), data.get("model"), data.get("type", "chat"), history

def list_conversations():
    """List all saved conversations as dropdown choices."""
    files = sorted(glob.glob(os.path.join(CONVERSATIONS_DIR, "*.json")), key=os.path.getmtime, reverse=True)
    choices = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            convo_id = data.get("id", "?")
            model = data.get("model", "?")
            ctype = data.get("type", "chat")
            updated = data.get("updated_at", "")[:16].replace("T", " ")
            preview = ""
            for m in data.get("messages", []):
                if m["role"] == "user":
                    preview = m["content"][:40]
                    break
            icon = "🎭" if ctype == "simulation" else "💬"
            choices.append(f"{convo_id} | {icon} {model} | {preview}... | {updated}")
        except Exception:
            pass
    return choices

def get_text_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return " ".join([c.get('text', '') if isinstance(c, dict) else str(c) for c in content])
    elif isinstance(content, tuple) and len(content) > 0:
        return str(content[0])
    return str(content)

def stream_from_agent(messages, model_name):
    """Stream chat response from the selected model's agent container."""
    agent_url = MODEL_REGISTRY.get(model_name)
    if not agent_url:
        yield {"type": "token", "content": f"Error: Unknown model {model_name}"}
        return

    # Build clean message list for the agent API
    payload_messages = []
    for msg in messages:
        payload_messages.append({"role": msg["role"], "content": get_text_content(msg["content"])})
    
    payload = {"messages": payload_messages}
    try:
        response = requests.post(f"{agent_url}/chat", json=payload, stream=True, timeout=120)
        for line in response.iter_lines():
            if line:
                event = json.loads(line)
                yield event
    except requests.exceptions.ConnectionError:
        yield {"type": "token", "content": f"⚠️ Agent container for **{model_name}** is not reachable. Make sure to pull the model first."}
    except Exception as e:
        yield {"type": "token", "content": f"Error: {str(e)}"}

def submit_message(message, history, selected_model, convo_id):
    if not message or not message.strip():
        yield history, "", convo_id
        return
    
    # Auto-create conversation ID if none exists
    if not convo_id:
        convo_id = new_conversation_id()
    
    # Add user message
    history = history + [{"role": "user", "content": message}]
    yield history, "", convo_id
    
    # Stream response from agent
    output = ""
    for event in stream_from_agent(history, selected_model):
        etype = event.get("type")
        if etype == "token":
            output += event["content"]
            yield history + [{"role": "assistant", "content": output}], "", convo_id
        elif etype == "tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("input", "")
            output += f"\n\n> **🛠️ Calling Tool:** `{tool_name}`\n> **Input:** `{tool_input}`\n\n"
            yield history + [{"role": "assistant", "content": output}], "", convo_id
        elif etype == "tool_end":
            tool_output = event.get("output", "")
            content_safe = str(tool_output).replace("<", "&lt;").replace(">", "&gt;")
            output += f"<details>\n<summary><b>✅ Tool Result ({len(tool_output)} chars) - Click to expand</b></summary>\n\n<pre><code>{content_safe}</code></pre>\n\n</details>\n\n"
            yield history + [{"role": "assistant", "content": output}], "", convo_id
    
    # Finalize: add assistant message to history and save
    if output:
        history = history + [{"role": "assistant", "content": output}]
    save_conversation(convo_id, selected_model, history)
    yield history, "", convo_id

def export_history(history, selected_model):
    md_content = f"# Chat Export\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_content += f"Model Used: {selected_model} (LangChain Agent)\n\n"
    md_content += "---\n\n"
    
    for msg in history:
        if msg["role"] == "user":
            md_content += f"### 👉 User\n{get_text_content(msg['content'])}\n\n"
        else:
            md_content += f"### 🤖 Agent\n{get_text_content(msg['content'])}\n\n"
                
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    with open(temp_file.name, "w", encoding="utf-8") as f:
        f.write(md_content)
    return temp_file.name

def check_agent_status(model_name):
    agent_url = MODEL_REGISTRY.get(model_name)
    try:
        res = requests.get(f"{agent_url}/health", timeout=3)
        if res.status_code == 200:
            return f"✅ **{model_name}** is online and ready"
    except Exception:
        pass
    return f"⚠️ **{model_name}** is not responding — it may still be pulling the model"

# ---- UI ----

URL_SYNC_JS = """
(val) => {
    if (val) {
        const url = new URL(window.location);
        url.searchParams.set('id', val);
        window.history.replaceState({}, '', url);
    }
    return val;
}
"""

with gr.Blocks(title="ContainAI") as demo:
    gr.Markdown("# 🐋 ContainAI")
    
    url_updater = gr.Textbox(visible=False, elem_id="url_updater")
    url_updater.change(None, inputs=[url_updater], js=URL_SYNC_JS)
    
    with gr.Tabs(elem_id="main_tabs") as tabs:
        # ============ TAB 1: CHAT ============
        with gr.Tab("💬 Chat", id="tab_chat"):
            convo_id_state = gr.State("")
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    value=list(MODEL_REGISTRY.keys())[0],
                    label="🧠 Select Model",
                    interactive=True,
                    scale=2
                )
                convo_dropdown = gr.Dropdown(
                    choices=list_conversations(),
                    value=None,
                    label="📂 Load Conversation",
                    interactive=True,
                    scale=3
                )
                status_btn = gr.Button("🔍 Status", variant="secondary", scale=1)
            
            model_status = gr.Markdown(f"Selected: **{list(MODEL_REGISTRY.keys())[0]}** — New conversation")
            
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                new_convo_btn = gr.Button("➕ New Chat", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear")
            
            with gr.Row():
                export_btn = gr.DownloadButton("💾 Download History (.md)", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
            
            # ---- Chat Event Handlers ----
            def update_status(model, convo_id):
                if convo_id:
                    return f"Selected: **{model}** — Conversation `{convo_id}`"
                return f"Selected: **{model}** — New conversation"
            
            def start_new_conversation():
                new_id = new_conversation_id()
                return [], "", new_id, f"Started new conversation `{new_id}`"
            
            def load_selected_conversation(selection):
                if not selection:
                    return [], list(MODEL_REGISTRY.keys())[0], "", "No conversation selected"
                convo_id = selection.split(" | ")[0].strip()
                loaded_id, model, ctype, history = load_conversation(convo_id)
                if loaded_id:
                    status = f"Loaded conversation `{loaded_id}` — Model: **{model}**"
                    return history, model or list(MODEL_REGISTRY.keys())[0], loaded_id, status
                return [], list(MODEL_REGISTRY.keys())[0], "", "Failed to load conversation"
            
            def refresh_convo_list():
                return gr.Dropdown(choices=list_conversations(), value=None)
            
            model_dropdown.change(update_status, inputs=[model_dropdown, convo_id_state], outputs=[model_status])
            status_btn.click(check_agent_status, inputs=[model_dropdown], outputs=[model_status])
            
            submit_btn.click(
                submit_message,
                inputs=[msg, chatbot, model_dropdown, convo_id_state],
                outputs=[chatbot, msg, convo_id_state]
            ).then(lambda x: x, inputs=[convo_id_state], outputs=[url_updater])
            
            msg.submit(
                submit_message,
                inputs=[msg, chatbot, model_dropdown, convo_id_state],
                outputs=[chatbot, msg, convo_id_state]
            ).then(lambda x: x, inputs=[convo_id_state], outputs=[url_updater])
            
            new_convo_btn.click(start_new_conversation, outputs=[chatbot, msg, convo_id_state, model_status]).then(lambda x: x, inputs=[convo_id_state], outputs=[url_updater])
            convo_dropdown.change(load_selected_conversation, inputs=[convo_dropdown], outputs=[chatbot, model_dropdown, convo_id_state, model_status]).then(lambda x: x, inputs=[convo_id_state], outputs=[url_updater])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
            export_btn.click(export_history, inputs=[chatbot, model_dropdown], outputs=[export_btn])
            refresh_btn.click(refresh_convo_list, outputs=[convo_dropdown])

        # ============ TAB 2: MULTI-AGENT SIMULATION ============
        with gr.Tab("🎭 Multi-Agent", id="tab_sim"):
            sim_convo_id_state = gr.State("")
            gr.Markdown("### Multi-Agent Simulation\nDescribe a scenario with multiple agents. They'll interact with each other autonomously.")
            gr.Markdown(
                '**Example:** *"I want a comedic standoff between agents ALICE and BOB to crack jokes on each other for 10 rounds. '
                'ALICE is a 22 year old girl and BOB is a 25 year old boy"*'
            )
            
            with gr.Row():
                sim_model = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    value=list(MODEL_REGISTRY.keys())[0],
                    label="🧠 Model to Power Agents",
                    interactive=True,
                    scale=3
                )
            
            sim_prompt = gr.Textbox(
                placeholder="Describe your multi-agent scenario here...",
                label="📜 Scenario Prompt",
                lines=3
            )
            
            sim_chatbot = gr.Chatbot(type="messages")
            
            with gr.Row():
                sim_interject_msg = gr.Textbox(placeholder="🗣️ Step in as HUMAN USER to intervene...", show_label=False, scale=4)
                sim_interject_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                sim_run_btn = gr.Button("🚀 Run Simulation", variant="primary")
                sim_new_btn = gr.Button("➕ New Simulation", variant="secondary")
                sim_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            def start_new_sim():
                new_id = new_conversation_id()
                return [], "", new_id
            
            def run_simulation(prompt, model_name, convo_id):
                if not prompt or not prompt.strip():
                    yield [], convo_id
                    return
                    
                if not convo_id:
                    convo_id = new_conversation_id()
                    
                agent_url = MODEL_REGISTRY.get(model_name)
                if not agent_url:
                    yield [{"role": "assistant", "content": f"Error: Unknown model {model_name}"}], convo_id
                    return
                
                history = [{"role": "user", "content": prompt}]
                yield history, convo_id
                save_conversation(convo_id, model_name, history, "simulation")
                
                try:
                    response = requests.post(
                        f"{agent_url}/orchestrate", 
                        json={"prompt": prompt, "convo_id": convo_id}, 
                        stream=True, 
                        timeout=300
                    )
                    for line in response.iter_lines():
                        if line:
                            event = json.loads(line)
                            etype = event.get("type")
                            
                            if etype == "status":
                                history.append({"role": "assistant", "content": event["content"]})
                                save_conversation(convo_id, model_name, history, "simulation")
                                yield history, convo_id
                            elif etype == "error":
                                history.append({"role": "assistant", "content": f"❌ {event['content']}"})
                                save_conversation(convo_id, model_name, history, "simulation")
                                yield history, convo_id
                            elif etype == "tool_event":
                                agent_name = event.get("agent", "")
                                round_num = event.get("round", "")
                                content = event["content"]
                                history.append({
                                    "role": "assistant",
                                    "content": f"**[Round {round_num}] {agent_name} — Tool:**\n\n{content}"
                                })
                                save_conversation(convo_id, model_name, history, "simulation")
                                yield history, convo_id
                            elif etype == "agent_message":
                                agent_name = event["agent"]
                                round_num = event["round"]
                                content = event["content"]
                                history.append({
                                    "role": "assistant",
                                    "content": f"**[Round {round_num}] {agent_name}:**\n\n{content}"
                                })
                                save_conversation(convo_id, model_name, history, "simulation")
                                yield history, convo_id
                except Exception as e:
                    history.append({"role": "assistant", "content": f"❌ Simulation error: {str(e)}"})
                    save_conversation(convo_id, model_name, history, "simulation")
                    yield history, convo_id
            def prep_sim_id(convo_id):
                if not convo_id:
                    return new_conversation_id()
                return convo_id

            sim_run_btn.click(
                prep_sim_id, inputs=[sim_convo_id_state], outputs=[sim_convo_id_state]
            ).then(
                run_simulation, inputs=[sim_prompt, sim_model, sim_convo_id_state], outputs=[sim_chatbot, sim_convo_id_state]
            ).then(
                lambda x: x, inputs=[sim_convo_id_state], outputs=[url_updater]
            )
            sim_new_btn.click(start_new_sim, outputs=[sim_chatbot, sim_prompt, sim_convo_id_state]).then(lambda x: x, inputs=[sim_convo_id_state], outputs=[url_updater])
            sim_clear_btn.click(lambda: [], outputs=[sim_chatbot])
            
            def handle_interjection(text, model_name, convo_id):
                if not text or not text.strip() or not convo_id:
                    return ""
                agent_url = MODEL_REGISTRY.get(model_name)
                if agent_url:
                    try:
                        requests.post(f"{agent_url}/interject", json={"convo_id": convo_id, "text": text}, timeout=3)
                    except Exception:
                        pass
                return ""
                
            sim_interject_btn.click(handle_interjection, inputs=[sim_interject_msg, sim_model, sim_convo_id_state], outputs=[sim_interject_msg])
            sim_interject_msg.submit(handle_interjection, inputs=[sim_interject_msg, sim_model, sim_convo_id_state], outputs=[sim_interject_msg])
            
    # ---- URL Loading Logic ----
    def on_page_load(request: gr.Request):
        params = request.query_params
        convo_id = params.get("id")
        if not convo_id:
            return (
                gr.Tabs(), # No tab switch
                "",        # chat convo_id
                [],        # chat history
                gr.Dropdown(), # chat model
                "",        # sim convo_id
                [],        # sim history 
                gr.Dropdown(), # sim model
                gr.Textbox()   # sim prompt
            )
            
        _, model, ctype, history = load_conversation(convo_id)
        if not history:
            return gr.Tabs(), "", [], gr.Dropdown(), "", [], gr.Dropdown(), gr.Textbox()
            
        model_val = model if model else list(MODEL_REGISTRY.keys())[0]
        
        if ctype == "simulation":
            prompt = history[0]["content"] if history else ""
            return (
                gr.Tabs(selected="tab_sim"),
                "",                   # chat convo_id
                [],                   # chat history
                gr.Dropdown(),        # chat model
                convo_id,             # sim convo_id
                history,              # sim history
                gr.Dropdown(value=model_val), # sim model
                gr.Textbox(value=prompt) # sim prompt
            )
        else:
            return (
                gr.Tabs(selected="tab_chat"),
                convo_id,             # chat convo_id
                history,              # chat history
                gr.Dropdown(value=model_val), # chat model
                "",                   # sim convo_id
                [],                   # sim history
                gr.Dropdown(),        # sim model
                gr.Textbox()          # sim prompt
            )
            
    demo.load(
        on_page_load, 
        outputs=[tabs, convo_id_state, chatbot, model_dropdown, sim_convo_id_state, sim_chatbot, sim_model, sim_prompt]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)

