"""
FastAPI backend for Medical RAG System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core import config
from backend.core.exceptions import (
    SessionNotFound,
    DocumentProcessingError,
    RetrievalError,
    LLMError,
)
from backend.api.routes import session_routes, query_routes, document_routes, evaluation_routes


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Production-ready Medical RAG API with session management",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers must return a Response, not raise/return HTTPException.
@app.exception_handler(SessionNotFound)
async def session_not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(DocumentProcessingError)
async def document_processing_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def llm_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(session_routes.router)
app.include_router(query_routes.router)
app.include_router(document_routes.router)
app.include_router(evaluation_routes.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": config.API_VERSION}


@app.get("/")
async def root():
    return {
        "message": "Medical RAG API",
        "docs": "/docs",
        "health": "/health",
    }
