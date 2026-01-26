from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Core — follows agent3 pattern with add_messages for automatic accumulation
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Identity — set at WebSocket connect, passed through all nodes
    current_user: str       # username
    user_id: int            # users.id from database.py
    session_id: str         # UUID for this conversation session

    # User preferences (loaded once at connect from DB)
    user_location: str      # e.g. "Kathmandu" — for weather
    user_interests: str     # e.g. "technology,business" — for news

    # Action gating (confirmation flow)
    pending_confirmation: Optional[str]   # human-readable description of pending action
    pending_action: Optional[dict]        # serialized action dict to execute after confirm

    # Draft state
    draft_email: Optional[dict]           # {to, subject, body}

    # Voice/UI state — pushed to frontend via WebSocket
    voice_state: str        # "listening" | "thinking" | "speaking" | "idle"

    # Final spoken output
    final_response: Optional[str]
    error: Optional[str]
