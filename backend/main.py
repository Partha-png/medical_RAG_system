"""
FastAPI backend for Medical RAG System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.core import config
from backend.core.exceptions import SessionNotFound, DocumentProcessingError, RetrievalError, LLMError
from backend.api.routes import session_routes, query_routes, document_routes, evaluation_routes

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Production-ready Medical RAG API with session management"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(SessionNotFound)
async def session_not_found_handler(request, exc):
    return HTTPException(status_code=404, detail=str(exc))

@app.exception_handler(DocumentProcessingError)
async def document_processing_error_handler(request, exc):
    return HTTPException(status_code=422, detail=str(exc))

@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request, exc):
    return HTTPException(status_code=500, detail=str(exc))

@app.exception_handler(LLMError)
async def llm_error_handler(request, exc):
    return HTTPException(status_code=500, detail=str(exc))

# Include routers
app.include_router(session_routes.router)
app.include_router(query_routes.router)
app.include_router(document_routes.router)
app.include_router(evaluation_routes.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "version": config.API_VERSION
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical RAG API",
        "docs": "/docs",
        "health": "/health"
    }
