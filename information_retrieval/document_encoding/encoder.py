"""
Document encoder with BM25 index creation
"""
import os
import numpy as np
import pickle
from .bioBERT_encoder import BioBERTEncoder
from .medcpt_encoder import MEDCPTEncoder
from PyPDF2 import PdfReader
from .chunker import token_chunk
from .faiss_manager import add_embeddings_to_faiss


def read_file(file_path):
    """read different types of files"""
    text = []
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    text.append(content)
            return text

        elif ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return text

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def encode_documents(model_type, output_folder, input_file, batch_size=8, create_bm25=True):
    """
    Encode documents and optionally create BM25 index
    
    Args:
        model_type: 'biobert', 'medcpt', or 'bm25'
        output_folder: Where to save embeddings and indices
        input_file: Input document file
        batch_size: Batch size for encoding
        create_bm25: Whether to also create BM25 index for hybrid retrieval
    """
    documents = []
    file_text = read_file(input_file)
    
    if not file_text:
        raise ValueError(f"No text extracted from {input_file}")
    
    # For BM25-only, skip dense encoding
    if model_type.lower() == 'bm25':
        for page in file_text:
            # Simple chunking for BM25 (no tokenizer needed)
            chunks = page.split('\n\n')  # Split by paragraphs
            documents.extend([c.strip() for c in chunks if c.strip()])
        
        # Create BM25 index
        from information_retrieval.retrievers.bm25_retriever import create_bm25_index
        create_bm25_index(documents, output_folder)
        print(f"Created BM25 index with {len(documents)} documents")
        return []
    
    # For Elasticsearch, create a simple index file (actual indexing happens in ES)
    if model_type.lower() == 'elasticsearch':
        for page in file_text:
            chunks = page.split('\n\n')
            documents.extend([c.strip() for c in chunks if c.strip()])
        
        # Save documents for Elasticsearch indexing
        import pickle
        es_index_path = os.path.join(output_folder, "elasticsearch_documents.pkl")
        with open(es_index_path, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Saved {len(documents)} documents for Elasticsearch indexing")
        return []
    
    # For Hybrid, use BioBERT as the dense component
    if model_type.lower() == 'hybrid':
        encoder = BioBERTEncoder()
        actual_model_type = 'biobert'  # Use BioBERT for dense embeddings
    elif model_type.lower() == 'biobert':
        encoder = BioBERTEncoder()
        actual_model_type = 'biobert'
    elif model_type.lower() == 'medcpt':
        encoder = MEDCPTEncoder()
        actual_model_type = 'medcpt'
    else:
        raise ValueError("Unsupported model type. Choose 'biobert', 'medcpt', 'bm25', 'hybrid', or 'elasticsearch'.")
    
    # Chunk documents
    for page in file_text:
        page_chunks = token_chunk(page, encoder.tokenizer)
        documents.extend(page_chunks)
    
    print(f"Created {len(documents)} chunks from document")
    
    # Encode documents in batches
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeds = encoder.encode(batch)
        all_embeddings.extend(batch_embeds)

    os.makedirs(output_folder, exist_ok=True)
    embeddings_path = os.path.join(output_folder, f"{model_type}_embeddings.npy")
    np.save(embeddings_path, np.array(all_embeddings))

    print(f"Embeddings saved to {embeddings_path}")
    faiss_dir = output_folder
    print(f"Creating FAISS index in: {faiss_dir}")
    
    add_embeddings_to_faiss(
        np.array(all_embeddings), 
        documents, 
        faiss_dir, 
        actual_model_type,  # Use actual model type (biobert for hybrid)
        input_file
    )
    
    # Optionally create BM25 index for hybrid retrieval
    if create_bm25:
        try:
            from information_retrieval.retrievers.bm25_retriever import create_bm25_index
            create_bm25_index(documents, faiss_dir)
            print(f"Also created BM25 index for hybrid retrieval")
        except Exception as e:
            print(f"Warning: Could not create BM25 index: {e}")
    
    return all_embeddings


if __name__ == "__main__":
    model_name = "biobert"
    input_folder = r"C:\Users\PARTHA SARATHI\Downloads\Untitled document (1).pdf"
    output_folder = r"C:\Users\PARTHA SARATHI\Downloads"
    encode_documents(model_name, output_folder, input_folder, batch_size=8)