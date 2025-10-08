# chat_cli.py

from src.rag_chain import build_chain
  
def chat():
  chain = build_chain()
  print("\n🤖 Insurance RAG Chatbot (type 'exit' to quit)\n")
  while True:
    q = input("You: ").strip()
    if q.lower() in ["exit", "quit"]:
      print("Goodbye!")
      break
    if not q.strip():
	    continue
    resp = chain.invoke(q)
      # resp might be string or an object; handle both
    if isinstance(resp, str):
      print("Bot:", resp)
    else:
      try:
        print("Bot:", resp.content)
      except:
        # fallback
        print("Bot:", resp)

if __name__ == "__main__":
  chat()
