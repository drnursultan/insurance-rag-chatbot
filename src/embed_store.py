from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import datetime

def build_index():
    print(f"\n🕒 Embedding process started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    loader = DirectoryLoader("data", glob="*.txt")
    docs = loader.load()
    print(f"📁 Loaded {len(docs)} text files for embedding:")
    for d in docs:
        print("   •", d.metadata.get("source", "unknown file"))
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create new Chroma DB (force fresh)
    db = Chroma(
        collection_name="insurance_docs",
        embedding_function=embeddings,
        persist_directory="chroma"
    )
    db.delete_collection()  # Reset if collection exists
    db = Chroma(
        collection_name="insurance_docs",
        embedding_function=embeddings,
        persist_directory="chroma"
    )

    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    db.add_texts(texts, metadatas=metas)
    db.persist()
    
    print(f"✅ Finished embedding {len(docs)} document chunks.")
    return db


def get_retriever(k=4):
    """Loads the existing Chroma DB and returns a retriever."""
    from src import config
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        collection_name="insurance_docs",
        embedding_function=embeddings,
        persist_directory=config.VECTOR_DB_DIR
    )
    retriever = db.as_retriever(search_kwargs={"k": k})
    print(f"✅ Retriever loaded with top-{k} context chunks.")
    return retriever
  
  
if __name__ == "__main__":
    print("🚀 Running embedding builder...\n")
    build_index()
