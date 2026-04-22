"""
common/llm_client.py
--------------------
Shared LLM client, agent executor, Flask app instance, and Ollama startup
logic. All other modules import from here so that there is a single source
of truth for the model name, base URL, tools list, and Flask app.
"""

import os
import json
import time

import requests as req
from flask import Flask
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from tools.my_tools import get_all_tools

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Tools & agent executor
# ---------------------------------------------------------------------------
tools = get_all_tools()
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, num_ctx=16384, timeout=300)
# create_react_agent (LangGraph) returns a compiled graph
agent_executor = create_react_agent(llm, tools)

# ---------------------------------------------------------------------------
# Ollama startup helper
# ---------------------------------------------------------------------------
def startup():
    """Wait for Ollama to become available, then pull the required model."""
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
    response = req.post(
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"name": MODEL_NAME},
        stream=True,
    )
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "status" in data:
                    print(data["status"], flush=True)
    print(f"[{MODEL_NAME}] Ready to serve.", flush=True)
