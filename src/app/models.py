from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint.

    From the API consumer's perspective we only expose the final,
    verified answer plus some metadata (e.g. context snippets).
    Internal draft answers remain inside the agent pipeline.
    """

    answer: str
    context: str
    citations: dict[str, dict] | None


class VercelChatRequest(BaseModel):
    """
    Vercel AI SDK request format.
    Accepts the standard UI messages format and transforms internally.
    """
    id: str  # Conversation ID from frontend
    messages: List[Dict[str, Any]]  # UI messages array
    trigger: str  # "submit-message" or other triggers
    thread_id: Optional[str] = None  # Optional override for thread_id
    resume: Optional[bool] = False  # Whether resuming from interrupt

