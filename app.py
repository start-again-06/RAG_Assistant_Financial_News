import time
import structlog
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.generation import get_rag_chain
from dotenv import load_dotenv

load_dotenv()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(), # <--- Key for production/Datadog/ELK
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20), # 20 = INFO
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

app = FastAPI(title="RAG Financial API")
chain = get_rag_chain("chroma_db")

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
   
    logger.info(
        "request_processed",
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration=f"{process_time:.4f}s",
    )
    return response

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_rag(request: QueryRequest):
   
    logger.info("rag_chain_started", query=request.question)
    
    response = chain.invoke({"query": request.question})
   
    return {
        "answer": response["result"],
        "sources": [doc.page_content for doc in response["source_documents"]]
    }

# To run: uvicorn app:app --reload
