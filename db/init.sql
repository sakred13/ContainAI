CREATE TABLE IF NOT EXISTS conversation_states (
    convo_id TEXT PRIMARY KEY,
    agent_id TEXT,
    model_name TEXT,
    status TEXT,
    state_json JSONB,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster JSON lookups if needed later
CREATE INDEX IF NOT EXISTS idx_convo_id ON conversation_states(convo_id);

CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    convo_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    result_json JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
