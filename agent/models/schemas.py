import time
from pydantic import BaseModel, Field
from .enums import AgentID, ModelName

class InvokeRequest(BaseModel):
    agentId: AgentID = Field(..., description="The ID of the agent to invoke")
    model: ModelName = Field(..., description="The model name to use for invocation")
    prompt: str = Field(..., description="The prompt or user input for the agent")
    convoId: str = Field(default_factory=lambda: f"invoke_{int(time.time())}", description="Unique ID for the conversation")
