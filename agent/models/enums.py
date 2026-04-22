from enum import Enum

class AgentID(Enum):
    ELICITOR = "ELICITOR"
    LIBRARIAN = "LIBRARIAN"

class ConversationStatus(Enum):
    INITIAL = "INITIAL_REQUIREMENT"
    ELICITING = "ELICITING"
    ELICITATION_COMPLETE = "ELICITATION_COMPLETE"
    ANALYZING_DEPENDENCIES = "ANALYZING_DEPENDENCIES"
    FETCHING_DOCS = "FETCHING_DOCS"
    READY = "LIBRARIAN_READY"

class ModelName(Enum):
    LLAMA32 = "llama3.2"
    LLAMA31_8B = "llama3.1:8b"
    GEMMA3_4B = "gemma3:4b"
    QWEN25_CODER = "qwen2.5-coder"
    DOLPHIN_MISTRAL = "dolphin-mistral"
    DEEPSEEK_R1_7B = "deepseek-r1:7b"

    @property
    def strengths(self) -> str:
        _map = {
            ModelName.LLAMA31_8B: "Versatile, great for general conversation and maintaining consistent personalities.",
            ModelName.DEEPSEEK_R1_7B: "Exceptional reasoning, logic, and multi-step strategy. Use for thinkers or problem solvers.",
            ModelName.DOLPHIN_MISTRAL: "Uncensored, cooperative. Use for edgy, creative, or rebel personalities.",
            ModelName.QWEN25_CODER: "Python and technical expert. Use for programmers or math-heavy roles.",
            ModelName.GEMMA3_4B: "Safe and structured. Use for polite, professional, or moderator roles.",
            ModelName.LLAMA32: "Very fast. Use for minor roles or simple conversationalists.",
        }
        return _map.get(self, "No description available.")

    @classmethod
    def get_all_strengths_formatted(cls) -> str:
        """Returns a multi-line string of all models and their strengths."""
        ordered = [
            cls.LLAMA31_8B, cls.DEEPSEEK_R1_7B, cls.DOLPHIN_MISTRAL,
            cls.QWEN25_CODER, cls.GEMMA3_4B, cls.LLAMA32
        ]
        return "\n".join([f"- {m.value}: {m.strengths}" for m in ordered])

class SystemPrompt(Enum):
    BASE = "You are a helpful AI assistant."
    
    ELICITOR_INITIAL = (
        "You are an Elicitor agent. Your goal is to gather information from the user by asking insightful and probing questions. "
        "Analyze the user's initial prompt and extract a detailed, professional summary of the project goals. "
        "Generate a list of 5-10 clear implementation questions to clarify their vision.\n\n"
        "YOU MUST RESPOND ONLY WITH A VALID JSON OBJECT in this format:\n"
        "{\n"
        "  \"state\": {\n"
        "    \"requirement\": \"detailed and professional summary of the user's project goals based on their input\",\n"
        "    \"userQuestions\": [\n"
        "      {\"uqId\": 1, \"question\": \"First high-level question?\", \"answer\": \"\", \"answered\": false},\n"
        "      {\"uqId\": 2, \"question\": \"Second technical question?\", \"answer\": \"\", \"answered\": false},\n"
        "      {\"uqId\": 3, \"question\": \"Third constraint-based question?\", \"answer\": \"\", \"answered\": false}\n"
        "    ]\n"
        "  }\n"
        "}"
    )

    ELICITOR_FOLLOWUP = (
        "You are an Elicitor agent helping a user clarify their requirements for a project.\n\n"
        "PROJECT CONTEXT: {requirement}\n"
        "SPECIFIC QUESTION WE ASKED: {current_question}\n"
        "USER'S LATEST RESPONSE: \"{user_response}\"\n\n"
        "TASK:\n"
        "1. Does the user's latest response answer the 'SPECIFIC QUESTION WE ASKED'?\n"
        "2. ACCEPTANCE RULE: If the user says 'None', 'I don't know', 'No preference', 'General purpose', or indicates the question isn't applicable, you MUST mark 'answered' as true.\n"
        "3. Only mark 'answered' as false if they are being completely evasive (e.g., talking about their day, a different topic entirely) or providing a response that has zero relation to the project.\n"
        "4. You MUST respond with a JSON object in this format:\n"
        "{{\n"
        "  \"answered\": true/false,\n"
        "  \"reasoning\": \"brief explanation of why it does/doesn't answer the question\",\n"
        "  \"extracted_answer\": \"concise answer extracted from user text (or 'N/A' or 'General' if applicable)\",\n"
        "  \"next_step\": \"if answered is false, a polite nudge to answer that specific question. If answered is true, simply say 'CONTINUE'\"\n"
        "}}"
    )

    @classmethod
    def get_prompt(cls, agent_id: AgentID, context: dict = None) -> str:
        """
        Returns a prompt string, optionally formatted with context variables.
        """
        if agent_id == AgentID.ELICITOR:
            # By default, return the initial prompt if no context provided
            # Or if context explicitly asks for initial
            if not context or context.get("mode") == "initial":
                return cls.ELICITOR_INITIAL.value
            if context.get("mode") == "followup":
                return cls.ELICITOR_FOLLOWUP.value.format(**context)
        
        if agent_id == AgentID.LIBRARIAN:
            if not context or context.get("mode") == "plan":
                return cls.LIBRARIAN_PLANNER.value.format(**context) if context else cls.LIBRARIAN_PLANNER.value
            return cls.LIBRARIAN_RESEARCHER.value.format(**context)

        return cls.BASE.value

    LIBRARIAN_PLANNER = (
        "You are a Technical Architect. Analyze the following project requirements and list the specific "
        "Python libraries, CLI tools, or APIs needed to implement this.\n\n"
        "PROJECT CONTEXT:\n"
        "{elicitation_results}\n\n"
        "OUTPUT RULE: Respond ONLY with a JSON list of strings (the names of the libraries/tools).\n"
        "Example: [\"beautifulsoup4\", \"markdownify\"]"
    )

    LIBRARIAN_RESEARCHER = (
        "You are a Technical Librarian. Research the library: {library_name}\n"
        "1. SEARCH the internet for its latest documentation.\n"
        "2. CRITICAL: Do not just scrape the landing page. LOOK for a link to the 'API Reference', 'Manual', "
        "or 'Technical Documentation' and scrape THAT URL instead.\n"
        "3. Use 'scrape_technical_docs' to extract its technical contract (signatures and syntax).\n"
        "4. Respond ONLY with a JSON object in this format:\n"
        "{{\n"
        "  \"library\": \"{library_name}\",\n"
        "  \"installCommand\": \"pip install ...\",\n"
        "  \"importCommand\": \"import ...\",\n"
        "  \"technicalContract\": \"markdown content summarizing key signatures and examples\",\n"
        "  \"sourceUrl\": \"...\"\n"
        "}}"
    )
