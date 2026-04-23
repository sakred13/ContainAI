const codingModule = (() => {
    const mainChat = document.getElementById('main-chat');
    const coderWorkspace = document.getElementById('coder-workspace');
    const coderChatBox = document.getElementById('coder-chat-box');
    const sandboxExplorer = document.getElementById('sandbox-files');
    const contextExplorer = document.getElementById('context-files');
    const editorModal = document.getElementById('editor-modal');
    const modalEditor = document.getElementById('modal-editor');
    const modalFilename = document.getElementById('modal-filename');
    const saveBtn = document.getElementById('save-file-btn');
    const closeModal = document.querySelector('.close-modal');

    let pollInterval = null;
    let currentFile = null;

    const getConvoId = () => {
        let id = window.location.pathname.slice(1);
        if (id === 'index.html' || !id) return null;
        return id;
    };

    const setMode = (mode) => {
        if (mode === 'code') {
            mainChat.style.display = 'none';
            coderWorkspace.style.display = 'flex';
            
            // Check if we have a convo ID in the URL, if not, generate one
            let id = getConvoId();
            if (!id) {
                id = 'code-' + Math.random().toString(36).substring(7);
                window.history.pushState({ id }, '', `/${id}`);
            }
            
            startPolling();
        } else {
            mainChat.style.display = 'flex';
            coderWorkspace.style.display = 'none';
            stopPolling();
        }
    };

    const startPolling = () => {
        if (pollInterval) clearInterval(pollInterval);
        refreshUI();
        pollInterval = setInterval(refreshUI, 2000);
    };

    const stopPolling = () => {
        clearInterval(pollInterval);
    };

    const refreshUI = async () => {
        const convoId = getConvoId();
        if (!convoId) return;

        try {
            // 1. Fetch Messages & Thoughts
            const res = await fetch(`/api/coding/conversation/${convoId}`);
            if (!res.ok) return;
            const data = await res.json();
            if (data.messages) {
                renderMessages(data.messages);
            }

            // 2. Fetch Files
            const fileRes = await fetch(`/api/coding/sandbox/${convoId}/files`);
            const files = await fileRes.json();
            renderFiles(files, convoId);

        } catch (e) {
            console.error("Polling error:", e);
        }
    };

    const renderMessages = (messages) => {
        coderChatBox.innerHTML = "";
        messages.forEach(msg => {
            // Render Message
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${msg.sender === 'USER' ? 'user' : 'ai'}`;
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.innerHTML = `<div style="font-size: 0.7rem; color: var(--accent); margin-bottom: 4px; font-weight: bold;">${msg.sender}</div>` + marked.parse(msg.content);
            msgDiv.appendChild(bubble);
            coderChatBox.appendChild(msgDiv);

            // Render Thoughts
            if (msg.thoughts && msg.thoughts.length > 0) {
                msg.thoughts.forEach(thought => {
                    const tDiv = document.createElement('div');
                    tDiv.className = 'thought-container';
                    tDiv.textContent = `💭 ${thought}`;
                    coderChatBox.appendChild(tDiv);
                });
            }
        });
        coderChatBox.scrollTop = coderChatBox.scrollHeight;
    };

    const renderFiles = (files, convoId) => {
        sandboxExplorer.innerHTML = "";
        files.forEach(file => {
            const div = document.createElement('div');
            div.className = 'file-item';
            div.innerHTML = `<span>📄</span> ${file.name}`;
            div.onclick = () => openEditor(convoId, file.path);
            sandboxExplorer.appendChild(div);
        });
    };

    const openEditor = async (convoId, path) => {
        const res = await fetch(`/api/coding/sandbox/${convoId}/file?path=${path}`);
        const data = await res.json();
        modalFilename.textContent = path;
        modalEditor.value = data.content || "";
        currentFile = { convoId, path };
        editorModal.style.display = 'flex';
    };

    const saveFile = async () => {
        if (!currentFile) return;
        const res = await fetch(`/api/coding/sandbox/${currentFile.convoId}/file?path=${currentFile.path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: modalEditor.value })
        });
        if (res.ok) {
            alert("File saved!");
            editorModal.style.display = 'none';
        }
    };

    saveBtn.onclick = saveFile;
    closeModal.onclick = () => editorModal.style.display = 'none';

    const sendCommand = async () => {
        const input = document.getElementById('coder-user-input');
        const text = input.value.trim();
        if (!text) return;
        
        let convoId = getConvoId();
        const model = document.getElementById('model-select').value;
        
        // Ensure convoId exists
        if (!convoId) {
            convoId = 'code-' + Math.random().toString(36).substring(7);
            window.history.pushState({ id: convoId }, '', `/${convoId}`);
        }

        input.value = "";
        
        const msgRes = await fetch(`/api/invoke`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agentId: 'LIBRARIAN',
                model: model,
                convoId: convoId,
                prompt: text
            })
        });

        if (msgRes.ok) {
            console.log("Agentic task invoked successfully");
            refreshUI();
        }
    };

    return { setMode, refreshUI, sendCommand };
})();

// Hook into the main mode switcher
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('mode-code');
    const sendBtn = document.getElementById('coder-send-btn');
    const userInput = document.getElementById('coder-user-input');

    if (btn) {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            codingModule.setMode('code');
        });
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', codingModule.sendCommand);
    }

    if (userInput) {
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                codingModule.sendCommand();
            }
        });
    }
});

