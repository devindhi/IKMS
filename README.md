# Building a Knowledge-Based Q&A Application with

# LangChain and Pinecone

In this session, we will develop a **document question-answering application** step by step. The application
will load a knowledge document (a PDF), index its content in a vector database, and use a GPT-based
language model to answer questions by retrieving information from the document. We’ll use **LangChain
1.0** (with the new LangGraph framework) for building our pipeline, **Pinecone** as the vector database, and an
OpenAI GPT-3.5 model (a "mini" GPT) for answering questions. Each part below introduces a component of
the system with background and code snippets.

Currently attention is all you need research paper is added to the vector database

frontend - https://github.com/devindhi/IKMS-frontend
backend - https://github.com/devindhi/IKMS
