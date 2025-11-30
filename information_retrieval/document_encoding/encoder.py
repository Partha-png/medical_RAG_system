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

def encode_documents(model_type, output_folder, input_file, batch_size=8):
    if model_type.lower() == 'biobert':
        encoder = BioBERTEncoder()
    elif model_type.lower() == 'medcpt':
        encoder = MEDCPTEncoder()
    else:
        raise ValueError("Unsupported model type. Choose 'biobert' or 'medcpt'.")
    documents = []
    file_text = read_file(input_file)
    
    if not file_text:
        raise ValueError(f"No text extracted from {input_file}")
    
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
    faiss_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_container")
    faiss_dir = os.path.abspath(faiss_dir)
    print(f"Creating FAISS index in: {faiss_dir}")
    
    add_embeddings_to_faiss(
        np.array(all_embeddings), 
        documents, 
        faiss_dir, 
        model_type, 
        input_file
    )
    return all_embeddings


if __name__ == "__main__":
    model_name = "biobert"
    input_folder = r"C:\Users\PARTHA SARATHI\Downloads\Untitled document (1).pdf"
    output_folder = r"C:\Users\PARTHA SARATHI\Downloads"
    encode_documents(model_name, output_folder, input_folder, batch_size=8)