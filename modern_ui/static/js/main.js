document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const modelSelect = document.getElementById('model-select');
    const historyList = document.getElementById('history-list');
    const modeChat = document.getElementById('mode-chat');
    const modeSim = document.getElementById('mode-sim');

    let currentMode = 'chat'; // 'chat' or 'sim'
    let currentConvoId = Date.now().toString().slice(-8);
    let currentConversation = [];

    // --- HISTORY MGMT ---
    const fetchHistory = async () => {
        const res = await fetch('/api/conversations');
        const history = await res.json();
        
        historyList.innerHTML = '<label>Recent Chats</label>';
        
        // Add "New Conversation" button
        const newConvo = document.createElement('div');
        newConvo.className = 'history-item active';
        newConvo.textContent = '➕ New Conversation';
        newConvo.onclick = () => setMode(currentMode);
        historyList.appendChild(newConvo);

        history.forEach(item => {
            const div = document.createElement('div');
            div.className = `history-item ${item.id === currentConvoId ? 'active' : ''}`;
            const icon = item.type === 'simulation' ? '🎭' : '💬';
            div.innerHTML = `
                <div style="font-weight: 600; font-size: 0.8rem;">${icon} ${item.id}</div>
                <div style="font-size: 0.7rem; opacity: 0.7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.preview || 'No messages'}</div>
            `;
            div.onclick = () => loadConversation(item.id);
            historyList.appendChild(div);
        });
    };

    const loadConversation = async (id) => {
        const res = await fetch(`/api/conversation/${id}`);
        const data = await res.json();
        if (data.error) return;

        currentConvoId = id;
        currentConversation = data.messages || [];
        modelSelect.value = data.model || modelSelect.value;
        currentMode = data.type || 'chat';
        
        modelSelect.disabled = (currentMode === 'simulation' || currentMode === 'sim');

        modeChat.classList.toggle('active', currentMode === 'chat');
        modeSim.classList.toggle('active', currentMode === 'sim');

        chatBox.innerHTML = "";
        currentConversation.forEach(msg => {
            const role = msg.role === 'user' ? 'user' : 'ai';
            addMessage(role, msg.content, msg.role === 'human' ? 'HUMAN OVERRIDE' : '');
        });

        // Highlight active
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        fetchHistory(); 
    };

    // --- MODE SWITCH ---
    const setMode = (mode) => {
        currentMode = mode;
        modeChat.classList.toggle('active', mode === 'chat');
        modeSim.classList.toggle('active', mode === 'sim');
        
        if (mode === 'sim') {
            modelSelect.value = "llama3.1:8b";
            modelSelect.disabled = true;
        } else {
            modelSelect.disabled = false;
        }

        // Clear UI for new mode
        chatBox.innerHTML = `
            <div style="text-align: center; margin-top: 20%; color: var(--text-muted);">
                <h2 style="color: var(--text-main); margin-bottom: 0.5rem;">${mode === 'chat' ? 'Welcome to Chat' : 'Simulation Mode'}</h2>
                <p>${mode === 'chat' ? 'Start a new agentic session.' : 'Describe a scenario for multiple agents to interact.'}</p>
            </div>
        `;
        currentConversation = [];
        currentConvoId = Math.random().toString(36).substring(7);
        fetchHistory();
    };

    modeChat.addEventListener('click', () => setMode('chat'));
    modeSim.addEventListener('click', () => setMode('sim'));

    // --- UTILS ---
    const addMessage = (role, content, agentName = '') => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        let label = "";
        if (agentName) {
            label = `<div style="font-size: 0.7rem; color: var(--accent); margin-bottom: 4px; font-weight: bold;">${agentName}</div>`;
        }

        bubble.innerHTML = label + marked.parse(content);
        
        msgDiv.appendChild(bubble);
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return bubble;
    };

    const addStatus = (content) => {
        const div = document.createElement('div');
        div.style.textAlign = 'center';
        div.style.fontSize = '0.8rem';
        div.style.color = 'var(--text-muted)';
        div.style.margin = '1rem 0';
        div.innerHTML = marked.parse(content);
        chatBox.appendChild(div);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const addToolLog = (name, input) => {
        const toolDiv = document.createElement('div');
        toolDiv.className = 'tool-call';
        toolDiv.innerHTML = `
            <div class="tool-header">
                <span>🛠️</span> <strong>${name}</strong>
                <em class="tool-chevron">▼</em>
            </div>
            <div class="tool-body">
                <div class="tool-body-inner">
                    <div class="tool-section-label">Input</div>
                    <div class="tool-output">${input || '…'}</div>
                    <div class="tool-section-label result-label" style="display:none">Result</div>
                    <div class="tool-result"></div>
                </div>
            </div>
        `;
        toolDiv.querySelector('.tool-header').addEventListener('click', () => {
            toolDiv.classList.toggle('open');
        });
        chatBox.appendChild(toolDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return {
            toolDiv,
            resultEl: toolDiv.querySelector('.tool-result'),
            resultLabel: toolDiv.querySelector('.result-label'),
        };
    };

    // --- API CALLS ---
    const streamChat = async (message) => {
        currentConversation.push({ role: 'user', content: message });
        
        addMessage('user', message);
        // AI bubble is created lazily on the first token so that tool
        // call blocks (tool_start) are always inserted above it.
        let aiBubble = null;
        let aiContent = "";
        let lastTool = null; // track current tool block for tool_end

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: modelSelect.value,
                messages: currentConversation,
                convo_id: currentConvoId
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.type === 'token') {
                        // Create the bubble on the very first token (after any tool blocks)
                        if (!aiBubble) aiBubble = addMessage('ai', '');
                        aiContent += data.content;
                        aiBubble.innerHTML = marked.parse(aiContent);
                        chatBox.scrollTop = chatBox.scrollHeight;
                    } else if (data.type === 'tool_start') {
                        // Appended now — before the AI bubble exists
                        lastTool = addToolLog(data.name, data.input || '...');
                    } else if (data.type === 'tool_end') {
                        if (lastTool) {
                            lastTool.resultEl.textContent = data.output || '';
                            lastTool.resultLabel.style.display = 'block';
                            // rAF ensures the browser paints the collapsed state first,
                            // so the grid-template-rows transition fires correctly.
                            requestAnimationFrame(() => lastTool.toolDiv.classList.add('open'));
                        }
                    }
                } catch (e) {}
            }
        }
        
        // Guard: if the model returned no tokens at all, add an empty bubble
        if (!aiBubble) addMessage('ai', '(No response)');
        currentConversation.push({ role: 'assistant', content: aiContent });
        fetchHistory(); // Refresh sidebar
    };

    const runSimulation = async (prompt) => {
        addMessage('user', prompt);
        
        const response = await fetch('/api/orchestrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: modelSelect.value,
                prompt: prompt,
                convo_id: currentConvoId
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.type === 'status') {
                        addStatus(data.content);
                    } else if (data.type === 'agent_message') {
                        const modelTag = data.model ? `(${data.model}) ` : "";
                        const personaTag = data.persona ? ` — *${data.persona}*` : "";
                        addMessage('ai', data.content, `🎭 ${data.agent} ${modelTag}${personaTag} (Round ${data.round})`);
                    } else if (data.type === 'tool_event') {
                        // content already includes both input and output — show expanded
                        const tool = addToolLog(data.agent, data.content);
                        requestAnimationFrame(() => tool.toolDiv.classList.add('open'));
                    } else if (data.type === 'error') {
                        addStatus(`❌ Error: ${data.content}`);
                    }
                } catch (e) {}
            }
        }
    };

    const handleInterjection = async (text) => {
        addMessage('user', text, 'HUMAN USER OVERRIDE');
        await fetch('/api/interject', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: currentMode === 'sim' ? 'llama3.1:8b' : modelSelect.value,
                convo_id: currentConvoId,
                text: text
            })
        });
    };

    // --- EVENTS ---
    sendBtn.addEventListener('click', () => {
        const text = userInput.value.trim();
        if (!text) return;
        userInput.value = "";
        userInput.style.height = 'auto';

        if (currentMode === 'chat') {
            streamChat(text);
        } else {
            // If conversation hasn't started, run orchestrate. Otherwise, interject.
            if (currentConversation.length === 0) {
                currentConversation.push({role: 'user', content: text});
                runSimulation(text);
            } else {
                handleInterjection(text);
            }
        }
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendBtn.click();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });

    // Populate models
    const init = async () => {
        const res = await fetch('/api/models');
        const models = await res.json();
        modelSelect.innerHTML = models.map(m => `<option value="${m}">${m.split(' — ')[0]}</option>`).join('');
        fetchHistory();
    };

    init();
});
