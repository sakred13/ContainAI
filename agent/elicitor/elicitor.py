import json
import re
from common.db_client import db_client
from models.enums import ConversationStatus

def prepare_elicitor_context(convo_id, user_input):
    """
    Determines the state of the conversation and returns necessary context 
    for building the ELICITOR system prompt.
    """
    state_json = db_client.get_state(convo_id)
    
    if not state_json:
        # No state exists yet - Initial requirement phase
        return {"mode": "initial"}
    
    # Existing state found - Follow-up phase
    # Find the first unanswered question
    questions = state_json.get("state", {}).get("userQuestions", [])
    requirement = state_json.get("state", {}).get("requirement", "No summary yet.")
    
    current_q = None
    for q in questions:
        if not q.get("answered"):
            current_q = q
            break
    
    if not current_q:
        # All questions answered already?
        return {"mode": "complete", "requirement": requirement}

    return {
        "mode": "followup",
        "requirement": requirement,
        "current_question": current_q.get("question"),
        "user_response": user_input
    }

def process_elicitor_response(convo_id, model_name, agent_id, result_text, mode="initial"):
    """
    Processes the LLM output based on the mode (initial gathering or follow-up validation).
    """
    try:
        # Extract JSON
        json_match = re.search(r"(\{.*\})", result_text, re.DOTALL)
        if not json_match:
            return result_text

        resp_obj = json.loads(json_match.group(1))

        if mode == "initial":
            return _handle_initial_save(convo_id, model_name, agent_id, resp_obj)
        else:
            return _handle_followup_save(convo_id, model_name, agent_id, resp_obj)

    except Exception as e:
        print(f"[ELICITOR PROCESSOR ERROR]: {e}")
        return result_text

def _handle_initial_save(convo_id, model_name, agent_id, state_obj):
    """Saves the initial list of questions and returns Q1."""
    if "state" in state_obj and "userQuestions" in state_obj["state"]:
        db_client.upsert_state(
            convo_id=convo_id,
            state_json=state_obj,
            agent_id=agent_id,
            model_name=model_name,
            status=ConversationStatus.ELICITING.value
        )
        questions = state_obj["state"]["userQuestions"]
        if questions:
            return f"Great! Here is my first question:\n\n{questions[0].get('question')}"
    return "I've processed your requirement, but I'm having trouble generating questions. Could you provide more detail?"

def _handle_followup_save(convo_id, model_name, agent_id, validation_obj):
    """Updates an existing state based on validation results and returns the next step."""
    state_json = db_client.get_state(convo_id)
    if not state_json:
        return "Internal error: State lost during follow-up."

    questions = state_json.get("state", {}).get("userQuestions", [])
    
    # 1. Did the user answer the current question?
    if not validation_obj.get("answered", False):
        # User didn't answer properly — return the nudge provided by the LLM
        return validation_obj.get("next_step", "I'm sorry, that didn't quite answer my question. could you please clarify?")

    # 2. They answered! Update the current question in the list
    answered_count = 0
    next_question = None
    
    for q in questions:
        if not q.get("answered"):
            # This was the one we were asking
            q["answered"] = True
            q["answer"] = validation_obj.get("extracted_answer")
            answered_count += 1
            break
        answered_count += 1

    # 3. Find the NEXT question
    for q in questions:
        if not q.get("answered"):
            next_question = q.get("question")
            break

    # 4. Save updated state
    status = ConversationStatus.ELICITING.value if next_question else ConversationStatus.COMPLETE.value
    db_client.upsert_state(
        convo_id=convo_id,
        state_json=state_json,
        agent_id=agent_id,
        model_name=model_name,
        status=status
    )

    if next_question:
        return f"Got it. Next question:\n\n{next_question}"
    else:
        return "Excellent! I have gathered all the initial requirements. We are ready to proceed!"
