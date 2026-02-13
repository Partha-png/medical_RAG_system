"""
API routes for evaluation metrics
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from backend.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])
evaluation_service = EvaluationService()


class RetrievalEvaluationRequest(BaseModel):
    """Request model for retrieval evaluation"""
    retrieved_docs: List[str]
    relevant_docs: List[str]


class RAGEvaluationRequest(BaseModel):
    """Request model for RAG evaluation"""
    generated_answer: str
    reference_answer: str
    retrieved_chunks: Optional[List[str]] = None


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation"""
    queries: List[str]
    retrieved_docs_list: List[List[str]]
    generated_answers: List[str]
    relevant_docs_list: Optional[List[List[str]]] = None
    reference_answers: Optional[List[str]] = None


@router.post("/retrieval")
async def evaluate_retrieval(request: RetrievalEvaluationRequest):
    """
    Evaluate retrieval quality metrics
    """
    try:
        metrics = evaluation_service.evaluate_retrieval(
            retrieved_docs=request.retrieved_docs,
            relevant_docs=request.relevant_docs
        )
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval evaluation failed: {str(e)}"
        )


@router.post("/rag")
async def evaluate_rag(request: RAGEvaluationRequest):
    """
    Evaluate RAG answer quality metrics
    """
    try:
        metrics = evaluation_service.evaluate_rag(
            generated_answer=request.generated_answer,
            reference_answer=request.reference_answer,
            retrieved_chunks=request.retrieved_chunks
        )
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {str(e)}"
        )


@router.post("/batch")
async def batch_evaluate(request: BatchEvaluationRequest):
    """
    Batch evaluation for multiple queries
    """
    try:
        results = evaluation_service.batch_evaluate(
            queries=request.queries,
            retrieved_docs_list=request.retrieved_docs_list,
            generated_answers=request.generated_answers,
            relevant_docs_list=request.relevant_docs_list,
            reference_answers=request.reference_answers
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evaluation failed: {str(e)}"
        )
