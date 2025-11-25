import os
import faiss
import pickle
import numpy as np


def get_or_create_faiss_index(faiss_dir: str, dim: int):
    """
    Load an existing FAISS index and metadata if available.
    If loading fails or files do not exist, create a new index.

    Args:
        faiss_dir (str): Directory where FAISS index and metadata are stored.
        dim (int): Dimensionality of the embedding vectors.

    Returns:
        tuple: (faiss.Index, metadata_list)
    """
    os.makedirs(faiss_dir, exist_ok=True)
    index_path = os.path.join(faiss_dir, 'index.faiss')
    metadata_path = os.path.join(faiss_dir, 'metadata.pkl')

    try:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            index = faiss.read_index(index_path)

            if index.d != dim:
                raise ValueError("Embedding dimension mismatch.")

            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            return index, metadata

    except Exception:
        pass

    index = faiss.IndexFlatL2(dim)
    metadata = []
    return index, metadata


def add_embeddings_to_faiss(embeddings: np.ndarray, new_texts: list, faiss_dir: str):
    """
    Add new embeddings and corresponding text metadata to the FAISS index.
    Automatically loads or creates the FAISS index and saves updated files.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (N, dim).
        new_texts (list): List of text chunks corresponding to embeddings.
        faiss_dir (str): Directory where FAISS index and metadata should be stored.

    Raises:
        ValueError: If embeddings cannot be added to the FAISS index.
        IOError: If saving FAISS index or metadata fails.
    """
    dim = embeddings.shape[1]
    index, metadata = get_or_create_faiss_index(faiss_dir, dim)

    try:
        index.add(embeddings.astype("float32"))
    except Exception as e:
        raise ValueError(f"Failed to add embeddings: {e}")

    metadata.extend(new_texts)

    try:
        faiss.write_index(index, os.path.join(faiss_dir, "index.faiss"))
        with open(os.path.join(faiss_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
    except Exception as e:
        raise IOError(f"Failed to save FAISS index or metadata: {e}")