from src.embed_store import build_index
from src.rag_chain import build_chain

print("Building index...")
n = build_index()
print(f"Indexed {n} chunks.\n")

chain = build_chain(k=4)
q = "How do I file an auto claim?"
print("Q:", q)
print("\nA:", chain.invoke(q))
