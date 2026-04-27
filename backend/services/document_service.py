"""
Document processing service
"""
import os
import re
import shutil
import uuid
from pathlib import Path
from backend.core import config
from backend.core.exceptions import DocumentProcessingError
from information_retrieval.document_encoding.encoder import encode_documents


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(filename: str) -> str:
    """Return a filesystem-safe basename, with no path components."""
    base = os.path.basename(filename or "")
    base = _SAFE_NAME_RE.sub("_", base).lstrip(".") or "upload"
    # Always prefix with a uuid so two users with the same name don't collide.
    return f"{uuid.uuid4().hex}_{base}"


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
        batch_size: int = 8,
    ) -> dict:
        """Process a document: chunk, encode, and create FAISS index."""
        try:
            session_faiss_dir = self.faiss_sessions_dir / session_id
            session_faiss_dir.mkdir(parents=True, exist_ok=True)

            embeddings = encode_documents(
                model_type=encoder_type,
                output_folder=str(session_faiss_dir),
                input_file=file_path,
                batch_size=batch_size,
                create_bm25=True,
            )

            if encoder_type.lower() == "bm25":
                bm25_index_path = session_faiss_dir / "bm25_index.pkl"
                if not bm25_index_path.exists():
                    raise DocumentProcessingError(
                        f"BM25 index creation failed for session {session_id}"
                    )
                return {
                    "session_id": session_id,
                    "encoder_type": encoder_type,
                    "num_chunks": 0,
                    "index_path": str(bm25_index_path),
                }

            if encoder_type.lower() == "hybrid":
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
                    "bm25_path": str(bm25_index_path),
                }

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
                "metadata_path": str(metadata_path),
            }

        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Document processing failed: {str(e)}") from e

    def get_session_faiss_dir(self, session_id: str) -> Path:
        return self.faiss_sessions_dir / session_id

    def delete_session_data(self, session_id: str) -> bool:
        """Delete all data associated with a session"""
        try:
            session_faiss_dir = self.faiss_sessions_dir / session_id
            if session_faiss_dir.exists():
                shutil.rmtree(session_faiss_dir)
            return True
        except Exception as e:
            raise DocumentProcessingError(f"Failed to delete session data: {str(e)}") from e

    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to temp directory.

        The supplied filename is sanitized to prevent path traversal and
        prefixed with a uuid so concurrent uploads don't collide.
        """
        try:
            self.temp_uploads_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_filename(filename)
            file_path = (self.temp_uploads_dir / safe_name).resolve()

            # Defense in depth: verify the resolved path is still under temp_uploads.
            uploads_root = self.temp_uploads_dir.resolve()
            if uploads_root not in file_path.parents:
                raise DocumentProcessingError("Invalid upload path")

            with open(file_path, "wb") as f:
                f.write(file_content)

            return str(file_path)
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Failed to save uploaded file: {str(e)}") from e

    def cleanup_temp_file(self, file_path: str) -> None:
        """Remove a previously-saved temp upload, ignoring errors."""
        try:
            p = Path(file_path).resolve()
            uploads_root = self.temp_uploads_dir.resolve()
            if uploads_root in p.parents and p.exists():
                p.unlink()
        except Exception:
            pass
