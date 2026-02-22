"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""
from typing import List
import json
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""

def _extract_plan_and_subquestions(messages: List[object]) -> tuple[str, list[str]]:
    """Extract plan and sub-questions from the last AIMessage.
    
    Returns:
        Tuple of (plan_text, list_of_subquestions)
    """
    # Get the last AI message content
    content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = str(msg.content)
            break
    
    if not content:
        return "", []
    
    # Split into plan and sub-questions sections

    plan_text = ""
    sub_questions = []
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('plan:'):
            current_section = 'plan'
            continue
        elif line.lower().startswith('sub-questions:') or line.lower().startswith('sub questions:'):
            current_section = 'subquestions'
            continue
        
        if current_section == 'plan' and line:
            plan_text += line + '\n'
        elif current_section == 'subquestions' and line:
            if line.startswith('-') or line.startswith('•'):
                query = line.lstrip('-•').strip().strip('"').strip("'")
                if query:
                    sub_questions.append(query)
    
    return plan_text.strip(), sub_questions


def _extract_citation_map(messages: List[object]) -> dict:
    """Extract citation map from ToolMessage artifacts.
    
    Returns:
        Dictionary mapping citation IDs to metadata
    """
    citation_map = {}
    for msg in messages:
        if hasattr(msg, 'artifact') and isinstance(msg.artifact, dict):
            if 'citation_map' in msg.artifact:
                citation_map.update(msg.artifact['citation_map'])
    return citation_map


# Define agents at module level for reuse
planning_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=PLANNING_SYSTEM_PROMPT,
)

retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


def planning_node(state: QAState) -> QAState:
    """Planning Agent node: analyzes the question to guide retrieval.
    
    This node passes the question through the planning agent
    for any initial analysis before retrieval.
    """
    print('PLANNINGGG')
    question = state["question"]

    result = planning_agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    plan, sub_questions = _extract_plan_and_subquestions(messages)
    
    return {
    "plan": plan,
    "sub_questions": sub_questions,
    "messages": [
        AIMessage(
            content=f"__META__:{json.dumps({'plan': plan, 'sub_questions': sub_questions})}__STEP__:plan__LABEL__:🧠 Research plan generated.",
            name="plan",
        )
    ]
}


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Sends the user's question to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Stores the consolidated context string in `state["context"]`.
    """
    question = state["question"]
    plan = state.get("plan", "")
    sub_questions = state.get("sub_questions", [])
    
    retrieval_prompt = f"""Original Question: {question}

    Search Plan:
    {plan}

    Sub-questions to address:
    {chr(10).join(f"{i}. {sq}" for i, sq in enumerate(sub_questions, 1))}
    """
   
    result = retrieval_agent.invoke({"messages": [HumanMessage(content=retrieval_prompt)]})
    messages = result.get("messages", [])
    context = ""

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            break

    citation_map = _extract_citation_map(messages)
    print("retrievallll")
    print(citation_map)

    return {
    "context": context,
    "citations": citation_map,
    "messages": [
        AIMessage(
            content=f"__META__:{json.dumps({'citations': citation_map})}__STEP__:retrieval__LABEL__:📚 Retrieved supporting sources.",
            name="retrieval",
        )
    ]
}


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer with citations.

    This node:
    - Sends question + plan + sub-questions + context (with citation IDs) to the Summarization Agent.
    - Agent responds with a draft answer that includes citation IDs ([C1], [C2], etc.).
    - Stores the draft answer in `state["draft_answer"]`.
    - Citation map is already in state from retrieval node.
    """
    question = state["question"]
    plan = state.get("plan", "")
    sub_questions = state.get("sub_questions", [])
    context = state.get("context", "")
    citation_map = state.get("citations", {})
  
    sub_questions_formatted = "\n".join(f"{i}. {sq}" for i, sq in enumerate(sub_questions, 1))
    
    user_content = f"""Original Question:
{question}

Answer Structure Plan:
{plan}

Sub-questions to Address:
{sub_questions_formatted}

Retrieved Context (with Citation IDs):
{context}

Citation:
{citation_map}
"""

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)
       
    return {
        "messages": [AIMessage(
            content="__STEP__:drafting__LABEL__:✍️ Drafting answer...",
            name="summarization",
        )],
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies citations and corrects the draft answer.

    This node:
    - Sends question + plan + context + draft_answer + citation_map to the Verification Agent.
    - Agent verifies that each citation ID is accurate and properly used.
    - Agent checks that the answer follows the plan structure.
    - Agent verifies all sub-questions are addressed.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer (with citations) in `state["answer"]`.
    """
    question = state["question"]
    plan = state.get("plan", "")
    sub_questions = state.get("sub_questions", [])
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")
    citation_map = state.get("citations", {})
    
    print(f"\n📋 Original Question:\n{question}\n")
    
    draft_citations = [cid for cid in citation_map.keys() if f"[{cid}]" in draft_answer]
    
    print(f"📝 DRAFT ANSWER TO VERIFY:")
    print("-" * 80)
    print(draft_answer)
    print("-" * 80)
    print(f"Draft length: {len(draft_answer)} characters")
    print(f"Citations in draft: {len(draft_citations)} ({', '.join(draft_citations)})\n")
    
    sub_questions_formatted = "\n".join(f"{i}. {sq}" for i, sq in enumerate(sub_questions, 1))
    
    citation_reference = "\n".join(
        f"{cid}: Page {meta['page']}, Source: {meta['source']}"
        for cid, meta in citation_map.items()
    )
    
    user_content = f"""Original Question:
{question}

Expected Answer Structure (Plan):
{plan}

Sub-questions That Should Be Addressed:
{sub_questions_formatted}

Retrieved Context (with Citation IDs):
{context}

Citation Reference Map:
{citation_reference}

Draft Answer to Verify:
{draft_answer}
"""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)
    
    final_citations = [cid for cid in citation_map.keys() if f"[{cid}]" in answer]
    
    print(f"\n✅ FINAL VERIFIED ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    print(f"Citations in final: {len(final_citations)} ({', '.join(final_citations)})\n")
    
    added_citations = set(final_citations) - set(draft_citations)
    removed_citations = set(draft_citations) - set(final_citations)

    print("📊 COMPARISON:")
    print(f"   Draft length:     {len(draft_answer)} chars / ~{len(draft_answer.split())} words")
    print(f"   Final length:     {len(answer)} chars / ~{len(answer.split())} words")
    print(f"   Change:           {len(answer) - len(draft_answer):+d} chars")
    print(f"   Draft citations:  {len(draft_citations)} {draft_citations}")
    print(f"   Final citations:  {len(final_citations)} {final_citations}")
    if added_citations:
        print(f"   ✅ Added:         {', '.join(added_citations)}")
    if removed_citations:
        print(f"   ❌ Removed:       {', '.join(removed_citations)}")
    if not added_citations and not removed_citations:
        print(f"   ℹ️  Citations unchanged")
    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETED")
    print("=" * 80 + "\n")

    return {
        "messages": [AIMessage(content=answer, name="answer")],
        "answer": answer,
    }