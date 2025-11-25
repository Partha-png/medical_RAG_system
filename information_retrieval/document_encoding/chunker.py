def token_chunk(text, tokenizer, max_tokens=150, overlap=30):
    """Token based chunking with overlap for RAG."""
    
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0

    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens).strip()
        chunks.append(chunk_text)

        i += max_tokens - overlap  # sliding window

    return chunks

if __name__ == "__main__":
    sample_text = "This is a sample text to demonstrate token-based chunking. " * 50
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    chunks = token_chunk(sample_text, tokenizer)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}:\n{chunk}\n")