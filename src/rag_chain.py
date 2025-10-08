from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from .embed_store import get_retriever
from .llm_providers import load_llm

SYS = """You are an insurance assistant. Answer using ONLY the provided context.
If the answer is not in the context, say you don't know. Cite sources by filename."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYS),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer succinctly with citations:")
])

def _format_docs(docs):
    out = []
    for d in docs:
        out.append(f"[{d.metadata.get('source')}] {d.page_content.strip()}")
    return "\n\n".join(out)

def build_chain(k=4):
    retriever = get_retriever(k=k)
    llm = load_llm()
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )
    return chain
