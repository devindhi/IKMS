"""Utilities for serializing retrieved document chunks."""

from typing import List

from langchain_core.documents import Document


def serialize_chunks(docs: List[Document]) -> tuple[str, dict]:
    """Serialize a list of Document objects into a formatted CONTEXT string.

    Formats chunks with indices and page numbers as specified in the PRD:
    - Chunks are numbered (Chunk 1, Chunk 2, etc.)
    - Page numbers are included in the format "page=X"
    - Produces a clean CONTEXT section for agent consumption

    Args:
        docs: List of Document objects with metadata.

    Returns:
        Formatted string with all chunks serialized.
    """
    context_parts = []
    citation_map = {}
    
    for i, doc in enumerate(docs):
        chunk_id = f"C{i+1}"
        page = doc.metadata.get("page", "unknown")
        
        context_parts.append(f"[{chunk_id}] Chunk from page {page}:\n{doc.page_content}")
        
        citation_map[chunk_id] = {
            "page": page,
            "snippet": doc.page_content[:100] + "...",
            "source": doc.metadata.get("source", "unknown")
        }
    
    return "\n\n".join(context_parts), citation_map
