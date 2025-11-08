import os
import numpy as np
import pickle
from .bioBERT_encoder import BioBERTEncoder
from .medcpt_encoder import MEDCPTEncoder
from PyPDF2 import PdfReader
def read_file(file_path):
    """read different types of files"""
    text=[]
    ext=os.path.splitext(file_path)[1].lower()
    try:
        if ext=='.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                if f.read().strip():
                    text.append(f.read())
            return text

        elif ext=='.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                if page.extract_text():
                   text.append(page.extract_text())
            return text

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def encode_documents(model_type,output_folder,input_file,batch_size=8):
        if model_type.lower()=='biobert':
            encoder=BioBERTEncoder()
        elif model_type.lower()=='medcpt':
            encoder=MEDCPTEncoder()
        else:
            raise ValueError("Unsupported model type. Choose 'biobert' or 'medcpt'.")
        
        file_text=[]
        documents=[]

        if not os.path.isfile(input_file):
            raise ValueError(f"{input_file} is not a valid file path.")
        file_text = read_file(input_file)
        documents.extend(file_text)
        all_embeddings=[]
        for i in range(0,len(file_text),batch_size):
            batch=documents[i:i+batch_size]
            batch_embeds=encoder.encode(batch)
            all_embeddings.extend(batch_embeds)

        os.makedirs(output_folder,exist_ok=True)
        np.save(os.path.join(output_folder,f"{model_name}_embeddings.npy"),np.array(all_embeddings))
        print(f"Embeddings saved to {output_folder}")
        return all_embeddings

if __name__ == "__main__":
       model_name="biobert"
       input_folder=r"C:\Users\PARTHA SARATHI\Downloads\asish_resume.pdf"
       output_folder=r"C:\Users\PARTHA SARATHI\Downloads"
       encode_documents (model_name,output_folder,input_folder,batch_size=8)