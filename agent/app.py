"""
app.py
------
Thin application entrypoint.

Responsibility: register blueprints on the shared Flask app and expose the
/health route. All business logic lives in the sub-packages:

  common/            – shared LLM client, Ollama startup, Flask app instance
  worker_agent/      – /chat endpoint (single-user conversation + tool calling)
  orchestrator_agent/– /orchestrate and /interject endpoints (multi-agent sim)
"""

from common.llm_client import app, startup, MODEL_NAME
from worker_agent.routes import worker_bp
from orchestrator_agent.routes import orchestrator_bp

# Register blueprints
app.register_blueprint(worker_bp)
app.register_blueprint(orchestrator_bp)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, threaded=True)
