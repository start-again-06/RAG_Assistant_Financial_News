import os
from src.ingestion import build_vector_store
from src.generation import get_rag_chain

JSON_DATA = "stock_news.json"
DB_DIR = "./chroma_db"

def main():
   
    if not os.path.exists(DB_DIR):
        print("Initializing Vector Database...")
        build_vector_store(JSON_DATA, DB_DIR)
    
    
    rag_system = get_rag_chain(DB_DIR)

    
    query = "What are the latest updates on Apple's AI and product launches?"
    print(f"\nQuerying: {query}")
    
    response = rag_system.invoke({"query": query})

    print("\n=== AI RESPONSE ===")
    print(response['result'])

    print("\n=== SOURCES USED ===")
    for doc in response['source_documents']:
        print(f"- {doc.metadata['title']} ({doc.metadata['ticker']})")

if __name__ == "__main__":
    main()
