# ContainAI 🐋 Local Agentic AI

This project runs a completely local, private large language model (LLM) agent using Docker and [Ollama](https://ollama.com/). It includes a sleek Gradio-based web interface to chat with your model without needing an external internet connection or paid API keys.

## Features
- **Ollama Engine**: Automatically pulls and runs the specified AI model (defaults to `llama3.2`).
- **Gradio Chat UI**: A responsive, ChatGPT-like web frontend.
- **Dockerized Environment**: The entire stack is containerized for cross-platform compatibility, isolating dependencies from your host machine.
- **Persistent Data**: Downloaded models are safely stored in a local Docker volume, so you never have to re-download massive model files on reboot.

---

## Prerequisites
Before you start, ensure you have the following installed on your machine:
- [Docker](https://docs.docker.com/get-docker/)
- Docker Compose (usually included with Docker Desktop)

---

## 🚀 How to Run

1. **Open your terminal** and navigate to this folder:
   ```bash
   cd c:\Users\Reddy\Desktop\code\Locallama
   ```

2. **Start the containers** in detached mode (background):
   ```bash
   docker-compose up --build -d
   ```

3. **Wait for startup**. On the very first run, the background container may take a few minutes to download the massive AI model (~1.5GB+).

4. **Access the Chat Interface**: Open your favorite web browser and go to:
   **👉 [http://localhost:7860](http://localhost:7860)**

---

## 🛠️ API Access (Optional)
You don't have to use the web application! The Ollama backend exposes a standard REST API on port `11434`. You can interact with it via Postman, cURL, LangChain, or your own code.

**Example cURL request to invoke the model directly:**
```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Explain quantum computing in one sentence.",
  "stream": false
}'
```

---

## ⚙️ Configuration
**Changing the AI Model**: 
If you want to use a different model (like `gemma`, `mistral`, or `phi3`):
1. Open `base_agent/app.py`.
2. Change the `MODEL_NAME = "llama3.2"` variable near the top to your preferred model.
3. Restart the environment:
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

## 🛑 Stopping the Agent
To shut down the infrastructure gracefully and free up your system resources, run:
```bash
docker-compose down
```
*(Your pulled models will be saved safely for the next time you boot up!)*
