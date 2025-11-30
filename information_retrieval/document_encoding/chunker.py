def token_chunk(text, tokenizer, max_tokens=150, overlap=100):
    sentences=text.split('.')
    chunks=[]
    current_chunk=[]
    current_length=0
    for sen in sentences:
        sent_tokens=tokenizer.tokenize(sen)
        sent_len=len(sent_tokens)
        if current_length+sent_len>max_tokens and current_chunk:
            chunk_text='.'.join(current_chunk)+'.'
            chunks.append(chunk_text)
            overlap_sentences=current_chunk[-2:] if len(current_chunk)>=2 else current_chunk
            current_chunk=overlap_sentences+[sen]
            current_length=sum(len(tokenizer.tokenize(s)) for s in current_chunk)
        else:
            current_chunk.append(sen)
            current_length+=sent_len
    if current_chunk:
        chunks.append('.'.join(current_chunk)+'.')
    return chunks
if __name__ == "__main__":
    sample_text = "This is a sample text to demonstrate token-based chunking. " * 50
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    chunks = token_chunk(sample_text, tokenizer)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}:\n{chunk}\n")