from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_classic.chains import RetrievalQA 

from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv
load_dotenv()

def get_rag_chain(persist_dir):
   
    vector_db = Chroma(
        persist_directory=persist_dir, 
        embedding_function=OpenAIEmbeddings()
    )

    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    
    template = """
    You are a professional financial assistant. Answer the question based ONLY on the following context.
    If the answer is not contained within the context, say "I don't have enough information from the news to answer this."
    Do not add any outside information or personal opinions.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
    )
    
    print("RAG Chain initialized with restricted prompt.")
    return chain
