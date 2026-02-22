"""LangGraph orchestration for the linear multi-agent QA flow."""

from functools import lru_cache
from typing import Any, Dict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from ...utils.langgraph_vercel_adapter import stream_langgraph_to_vercel

from typing import AsyncGenerator


from .agents import retrieval_node, summarization_node, verification_node, planning_node
from .state import QAState


def create_qa_graph() -> Any:
    """Create and compile the linear multi-agent QA graph.

    The graph executes in order:
    1. Retrieval Agent: gathers context from vector store
    2. Summarization Agent: generates draft answer from context
    3. Verification Agent: verifies and corrects the answer

    Returns:
        Compiled graph ready for execution.
    """
    builder = StateGraph(QAState)

    # Add nodes for each agent
    builder.add_node("planning", planning_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("summarization", summarization_node)
    builder.add_node("verification", verification_node)

    # Define linear flow: START -> planning -> retrieval -> summarization -> verification -> END
    builder.add_edge(START, "planning")
    builder.add_edge("planning", "retrieval")
    builder.add_edge("retrieval", "summarization")
    builder.add_edge("summarization", "verification")
    builder.add_edge("verification", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_qa_graph() -> Any:
    """Get the compiled QA graph instance (singleton via LRU cache)."""
    return create_qa_graph()


async def run_qa_flow(message: str, thread_id: str, resume: bool = False):
    """Run the complete multi-agent QA flow for a question.

    This is the main entry point for the QA system. It:
    1. Initializes the graph state with the question
    2. Executes the linear agent flow (Retrieval -> Summarization -> Verification)
    3. Extracts and returns the final results

    Args:
        question: The user's question about the vector databases paper.

    Returns:
        Dictionary with keys:
        - `answer`: Final verified answer
        - `draft_answer`: Initial draft answer from summarization agent
        - `context`: Retrieved context from vector store
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_qa_graph()
    
    if resume:
        # Resume execution with user input
        initial_state = Command(resume=message)
    else:
        initial_state = QAState(
        messages=[HumanMessage(content=message)],
        question=message,
        context=None,
        draft_answer=None,
        answer=None,
        plan=None,
        sub_questions=None,
        citations=None,
    )
    

    # final_state = graph.invoke(initial_state)
    
    # Stream using the pluggable adapter!
    # No need to specify stream_mode or graph-specific logic
    # Configure custom data fields to stream alongside messages
    async for event in stream_langgraph_to_vercel(
        graph= graph,
        initial_state=initial_state,
        config=config,
        custom_data_fields=["plan","retrieval","answer"],
    ):
        yield event

