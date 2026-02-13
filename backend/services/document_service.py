"""
Document processing service
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from backend.core import config
from backend.core.exceptions import DocumentProcessingError
from information_retrieval.document_encoding.encoder import encode_documents


class DocumentService:
    """Business logic for document processing"""
    
    def __init__(self):
        self.temp_uploads_dir = config.TEMP_UPLOADS_DIR
        self.faiss_sessions_dir = config.FAISS_SESSIONS_DIR
    
    def process_document(
        self, 
        file_path: str, 
        session_id: str, 
        encoder_type: str,
        batch_size: int = 8
    ) -> dict:
        """
        Process a document: chunk, encode, and create FAISS index
        
        Args:
            file_path: Path to the uploaded document
            session_id: Session ID to associate with this document
            encoder_type: Type of encoder ('biobert' or 'medcpt')
            batch_size: Batch size for encoding
            
        Returns:
            dict with processing stats (num_chunks, encoder_type, etc.)
        """
        try:
            # Create session-specific FAISS directory
            session_faiss_dir = self.faiss_sessions_dir / session_id
            session_faiss_dir.mkdir(parents=True, exist_ok=True)
            
            # Encode the document and create indices
            # Always create BM25 index for hybrid retrieval support
            embeddings = encode_documents(
                model_type=encoder_type,
                output_folder=str(session_faiss_dir),
                input_file=file_path,
                batch_size=batch_size,
                create_bm25=True  # Enable BM25 for hybrid retrieval
            )
            
            # Verify index was created based on encoder type
            if encoder_type.lower() == 'bm25':
                # BM25 creates a different index file
                bm25_index_path = session_faiss_dir / "bm25_index.pkl"
                if not bm25_index_path.exists():
                    raise DocumentProcessingError(
                        f"BM25 index creation failed for session {session_id}"
                    )
                return {
                    "session_id": session_id,
                    "encoder_type": encoder_type,
                    "num_chunks": 0,
                    "index_path": str(bm25_index_path)
                }
            elif encoder_type.lower() == 'elasticsearch':
                # Elasticsearch creates a documents pickle file
                es_docs_path = session_faiss_dir / "elasticsearch_documents.pkl"
                if not es_docs_path.exists():
                    raise DocumentProcessingError(
                        f"Elasticsearch document save failed for session {session_id}"
                    )
                return {
                    "session_id": session_id,
                    "encoder_type": encoder_type,
                    "num_chunks": 0,
                    "index_path": str(es_docs_path)
                }
            elif encoder_type.lower() == 'hybrid':
                # Hybrid uses BioBERT FAISS + BM25
                biobert_index_path = session_faiss_dir / "biobertindex.faiss"
                biobert_metadata_path = session_faiss_dir / "biobertmetadata.pkl"
                bm25_index_path = session_faiss_dir / "bm25_index.pkl"
                
                if not biobert_index_path.exists() or not biobert_metadata_path.exists():
                    raise DocumentProcessingError(
                        f"Hybrid FAISS index creation failed for session {session_id}"
                    )
                if not bm25_index_path.exists():
                    raise DocumentProcessingError(
                        f"Hybrid BM25 index creation failed for session {session_id}"
                    )
                    
                return {
                    "session_id": session_id,
                    "encoder_type": encoder_type,
                    "num_embeddings": len(embeddings) if embeddings else 0,
                    "index_path": str(biobert_index_path),
                    "metadata_path": str(biobert_metadata_path),
                    "bm25_path": str(bm25_index_path)
                }
            else:
                # Dense models (BioBERT, MedCPT) create FAISS indices
                index_path = session_faiss_dir / f"{encoder_type}index.faiss"
                metadata_path = session_faiss_dir / f"{encoder_type}metadata.pkl"
                
                if not index_path.exists() or not metadata_path.exists():
                    raise DocumentProcessingError(
                        f"FAISS index creation failed for session {session_id}"
                    )
                
                return {
                    "session_id": session_id,
                    "encoder_type": encoder_type,
                    "num_embeddings": len(embeddings) if embeddings else 0,
                    "index_path": str(index_path),
                    "metadata_path": str(metadata_path)
                }
            
        except Exception as e:
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    def get_session_faiss_dir(self, session_id: str) -> Path:
        """Get the FAISS directory for a session"""
        return self.faiss_sessions_dir / session_id
    
    def delete_session_data(self, session_id: str) -> bool:
        """Delete all data associated with a session"""
        try:
            session_faiss_dir = self.faiss_sessions_dir / session_id
            
            if session_faiss_dir.exists():
                shutil.rmtree(session_faiss_dir)
            
            return True
        except Exception as e:
            raise DocumentProcessingError(f"Failed to delete session data: {str(e)}")
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to temp directory"""
        try:
            file_path = self.temp_uploads_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            return str(file_path)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to save uploaded file: {str(e)}")
