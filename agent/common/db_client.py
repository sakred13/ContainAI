import os
import json
import psycopg2
from psycopg2.extras import Json
from datetime import datetime

class DBClient:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "db")
        self.user = os.getenv("POSTGRES_USER", "user")
        self.password = os.getenv("POSTGRES_PASSWORD", "password")
        self.dbname = os.getenv("POSTGRES_DB", "containai")
        self._conn = None

    def _get_connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                dbname=self.dbname
            )
        return self._conn

    def upsert_state(self, convo_id, state_json, status="active", agent_id=None, model_name=None):
        """Insert or update the state for a given conversation ID."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_states (convo_id, agent_id, model_name, status, state_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (convo_id) DO UPDATE SET
                    agent_id = EXCLUDED.agent_id,
                    model_name = EXCLUDED.model_name,
                    status = EXCLUDED.status,
                    state_json = EXCLUDED.state_json,
                    updated_at = EXCLUDED.updated_at;
                """,
                (convo_id, agent_id, model_name, status, Json(state_json), datetime.now())
            )
        conn.commit()

    def get_state(self, convo_id):
        """Fetch only the state_json for a given conversation ID."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT state_json FROM conversation_states WHERE convo_id = %s", (convo_id,))
            result = cur.fetchone()
            return result[0] if result else None

    def get_status(self, convo_id):
        """Fetch only the status for a given conversation ID."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM conversation_states WHERE convo_id = %s", (convo_id,))
            result = cur.fetchone()
            return result[0] if result else None

    def delete_state(self, convo_id):
        """Remove a conversation state from the database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_states WHERE convo_id = %s", (convo_id,))
        conn.commit()

    def list_all_convo_ids(self):
        """List all conversation IDs in the database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT convo_id FROM conversation_states ORDER BY updated_at DESC")
            return [row[0] for row in cur.fetchall()]

    def create_task(self, convo_id, agent_id):
        """Creates a new task and returns the task_id (UUID)."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_tasks (convo_id, agent_id) VALUES (%s, %s) RETURNING task_id",
                (convo_id, agent_id)
            )
            task_id = cur.fetchone()[0]
        conn.commit()
        return str(task_id)

    def update_task(self, task_id, status, result_json=None, error_message=None):
        """Updates the status and result/error of a task."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_tasks 
                SET status = %s, result_json = %s, error_message = %s, updated_at = %s
                WHERE task_id = %s
                """,
                (status, Json(result_json) if result_json else None, error_message, datetime.now(), task_id)
            )
        conn.commit()

    def get_task(self, task_id):
        """Fetches the full task details."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT task_id, convo_id, agent_id, status, result_json, error_message, updated_at FROM agent_tasks WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            if row:
                return {
                    "task_id": str(row[0]),
                    "convo_id": row[1],
                    "agent_id": row[2],
                    "status": row[3],
                    "result": row[4],
                    "error": row[5],
                    "updated_at": row[6].isoformat() if row[6] else None
                }
            return None

# Singleton instance for shared usage
db_client = DBClient()
