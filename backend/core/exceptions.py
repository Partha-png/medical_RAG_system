"""
Simple custom exceptions for Medical RAG System
"""

class SessionNotFound(Exception):
    """Session doesn't exist"""
    pass

class DocumentProcessingError(Exception):
    """Document processing failed"""
    pass

class RetrievalError(Exception):
    """Retrieval failed"""
    pass

class LLMError(Exception):
    """LLM call failed"""
    pass
