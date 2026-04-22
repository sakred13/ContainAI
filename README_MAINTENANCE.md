# ContainAI Maintenance

## ⚠️ Important Note
The `/api/invoke` endpoint in `modern_ui/app.py` has been added temporarily for testing purposes. It should be removed once the final orchestration logic is finalized and client applications are updated to use the direct agent endpoints or the consolidated orchestrator.

### Tasks:
- [ ] Remove `/api/invoke` from `modern_ui/app.py`
- [ ] Update documentation for client apps.
- [x] Remove debug port mapping `5005:5000` from `agent_llama` in `docker-compose.yml` after Postman testing is complete.

### **Asynchronous Task System (implemented 2026-04-22)**
*   **Architecture**: `/invoke` now returns a `task_id` (UUID) immediately and processes Librarian research in a background thread.
*   **Endpoints**: Added `GET /task/<task_id>` (Agent) and `GET /api/task/<task_id>` (UI Proxy) for polling status.
*   **Persistence**: Added `agent_tasks` table to PostgreSQL with status tracking (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`).
*   **Technical Scraping**: Upgraded `scrape_technical_docs` to be "greedy" (captures all code/sigs/plain-text signatures) and forced "Deep Link Discovery" in the Researcher prompt.
