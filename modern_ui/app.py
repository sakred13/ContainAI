import os
import json
import requests
import glob
from datetime import datetime
from flask import Flask, request, Response, send_from_directory, jsonify

app = Flask(__name__, static_folder='static')

# ---- CONFIGURATION ----
CONVERSATIONS_DIR = os.getenv("CONVERSATIONS_DIR", "/data/conversations")
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
MODEL_REGISTRY = {
    "llama3.2": os.getenv("AGENT_LLAMA_URL", "http://agent_llama:5000"),
    "llama3.1:8b": os.getenv("AGENT_LLAMA8B_URL", "http://agent_llama_8b:5000"),
    "gemma3:4b": os.getenv("AGENT_GEMMA_URL", "http://agent_gemma:5000"),
    "qwen2.5-coder": os.getenv("AGENT_QWEN_URL", "http://agent_qwen:5000"),
    "dolphin-mistral": os.getenv("AGENT_DOLPHIN_URL", "http://agent_dolphin:5000"),
    "deepseek-r1:7b": os.getenv("AGENT_DEEPSEEK_URL", "http://agent_deepseek:5000"),
}

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    if path and (path.startswith('api/') or '.' in path):
        return "Not Found", 404
    return send_from_directory('.', 'index.html')

@app.route('/api/models')
def get_models():
    return jsonify(list(MODEL_REGISTRY.keys()))

@app.route('/api/conversations')
def get_conversations():
    files = sorted(glob.glob(os.path.join(CONVERSATIONS_DIR, "*.json")), key=os.path.getmtime, reverse=True)
    history = []
    for f in files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
                history.append({
                    "id": data.get("id"),
                    "model": data.get("model"),
                    "type": data.get("type", "chat"),
                    "updated_at": data.get("updated_at"),
                    "preview": data.get("messages", [{}])[0].get("content", "")[:50]
                })
        except: continue
        
    # Append coding sessions from DB
    orchestrator_url = MODEL_REGISTRY.get('llama3.2')
    try:
        r = requests.get(f"{orchestrator_url}/coding/conversations", timeout=2)
        if r.status_code == 200:
            for convo in r.json():
                history.append({
                    "id": convo["id"],
                    "model": "qwen2.5-coder",
                    "type": "code",
                    "updated_at": convo["updated_at"],
                    "preview": f"Session status: {convo['status']}"
                })
    except: pass
    
    return jsonify(history)

@app.route('/api/conversation/<convo_id>')
def get_conversation(convo_id):
    path = os.path.join(CONVERSATIONS_DIR, f"{convo_id}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Not found"}), 404

def save_convo(convo_id, model, messages, ctype="chat"):
    path = os.path.join(CONVERSATIONS_DIR, f"{convo_id}.json")
    data = {
        "id": convo_id,
        "model": model,
        "type": ctype,
        "updated_at": datetime.now().isoformat(),
        "messages": messages
    }
    with open(path, 'w') as f:
        json.dump(data, f)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model')
    messages = data.get('messages')
    convo_id = data.get('convo_id')
    
    agent_url = MODEL_REGISTRY.get(model)
    if not agent_url:
        return jsonify({"error": f"Invalid model '{model}'. Valid models: {list(MODEL_REGISTRY.keys())}"}), 400

    print(f"[ROUTE] model='{model}' → {agent_url}", flush=True)

    def generate():
        try:
            # Proxying the streaming response from the agent
            resp = requests.post(f"{agent_url}/chat", json={"messages": messages}, stream=True, timeout=300)
            for line in resp.iter_lines():
                if line:
                    yield line + b'\n'
        except Exception as e:
            yield json.dumps({"type": "token", "content": f"Backend Proxy Error: {str(e)}"}).encode() + b'\n'
        finally:
            save_convo(convo_id or f"chat_{datetime.now().strftime('%H%M%S')}", model, messages)

    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/api/orchestrate', methods=['POST'])
def orchestrate():
    data = request.json
    model = data.get('model', "llama3.2")
    prompt = data.get('prompt')
    convo_id = data.get('convo_id')
    
    agent_url = MODEL_REGISTRY.get(model)
    if not agent_url:
        return jsonify({"error": f"Orchestrator model ({model}) not online"}), 500

    def generate():
        history_buffer = [{"role": "user", "content": prompt}]
        try:
            resp = requests.post(
                f"{agent_url}/orchestrate", 
                json={"prompt": prompt, "convo_id": convo_id}, 
                stream=True, 
                timeout=300
            )
            for line in resp.iter_lines():
                if line:
                    event = json.loads(line)
                    if event.get("type") in ["agent_message", "status"]:
                        history_buffer.append({"role": "assistant", "content": event["content"]})
                    yield line + b'\n'
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}).encode() + b'\n'
        finally:
            save_convo(convo_id, model, history_buffer, "simulation")

    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/api/interject', methods=['POST'])
def interject():
    data = request.json
    model = data.get('model')
    convo_id = data.get('convo_id')
    text = data.get('text')
    
    agent_url = MODEL_REGISTRY.get(model)
    if not agent_url:
        return jsonify({"error": "Invalid model"}), 400

    try:
        requests.post(f"{agent_url}/interject", json={"convo_id": convo_id, "text": text}, timeout=5)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    # Default to the llama3.2 agent for status checks
    agent_url = MODEL_REGISTRY.get('llama3.2')
    try:
        resp = requests.get(f"{agent_url}/task/{task_id}", timeout=10)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/invoke', methods=['POST'])
def invoke():
    data = request.json
    model = data.get('model', 'llama3.2')
    
    agent_url = MODEL_REGISTRY.get(model)
    if not agent_url:
        return jsonify({"error": f"Invalid model '{model}'"}), 400

    try:
        # Forward the invoke request to the agent
        resp = requests.post(f"{agent_url}/invoke", json=data, timeout=120)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/coding/<path:subpath>', methods=['GET', 'POST'])
def coding_proxy(subpath):
    # Use Llama 3.2 as the default orchestrator address
    orchestrator_url = MODEL_REGISTRY.get('llama3.2')
    url = f"{orchestrator_url}/coding/{subpath}"
    
    try:
        if request.method == 'GET':
            resp = requests.get(url, params=request.args, timeout=10)
        else:
            resp = requests.post(url, json=request.json, timeout=10)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health():
    results = {}
    for name, url in MODEL_REGISTRY.items():
        try:
            r = requests.get(f"{url}/health", timeout=1)
            results[name] = "online" if r.status_code == 200 else "offline"
        except:
            results[name] = "offline"
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
