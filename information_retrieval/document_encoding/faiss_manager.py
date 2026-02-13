import os
import faiss
import pickle
import numpy as np
from .bioBERT_encoder import BioBERTEncoder
from .medcpt_encoder import MEDCPTEncoder
from .chunker import token_chunk
from PyPDF2 import PdfReader

def get_or_create_faiss_index(faiss_dir: str, dim: int,model_name,file_path):
    metadata = []
    if model_name.lower()=='biobert':
        encoder=BioBERTEncoder()
    elif model_name.lower()=='medcpt':
        encoder=MEDCPTEncoder()

    os.makedirs(faiss_dir, exist_ok=True)
    index_path = os.path.join(faiss_dir, f"{model_name}index.faiss")
    metadata_path = os.path.join(faiss_dir, f"{model_name}metadata.pkl")

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
    text=[]
    ext=os.path.splitext(file_path)[1].lower()
    if ext=='.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    text.append(content)
            

    elif ext=='.pdf':
            
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                   text.append(page_text)
    full_text = " ".join(text)
            
    num_docs=len(token_chunk(full_text,encoder.tokenizer,max_tokens=400,overlap=100))
    
    if num_docs<1000:
        return faiss.IndexFlatL2(dim), metadata
    else:
        nlist=min(int(4*np.sqrt(num_docs)),num_docs//39)
        quantizer=faiss.IndexFlatL2(dim)
        index=faiss.IndexIVFFlat(quantizer,dim,nlist)
        return index, metadata


def add_embeddings_to_faiss(embeddings: np.ndarray, new_texts: list, faiss_dir: str,model_name,file_path):
    
    dim = embeddings.shape[1]
    index, metadata = get_or_create_faiss_index(faiss_dir, dim,model_name,file_path)
    if index.__class__.__name__.startswith("IndexIVF") and not index.is_trained:

        index.train(embeddings.astype("float32"))
    try:
        index.add(embeddings.astype("float32"))
    except Exception as e:
        raise ValueError(f"Failed to add embeddings: {e}")

    metadata.extend(new_texts)

    try:
        faiss.write_index(index, os.path.join(faiss_dir, f"{model_name}index.faiss"))
        with open(os.path.join(faiss_dir, f"{model_name}metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
    except Exception as e:
        raise IOError(f"Failed to save FAISS index or metadata: {e}")