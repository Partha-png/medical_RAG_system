from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import torch.nn.functional as F


class BioBERTQueryEncoder:
    """BioBERT encoder for queries"""
    
    def __init__(self, model_name="dmis-lab/biobert-v1.1", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, query: str):
        """Encode a single query into embedding vector"""
        tokens = self.tokenizer(
            [query],  # Wrap in list for batch processing
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens['input_ids'], 
                attention_mask=tokens['attention_mask']
            )
            last_hidden_state = outputs.last_hidden_state

        attention_mask = tokens["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        token_counts = mask_expanded.sum(dim=1)
        sum_mask = torch.maximum(token_counts, torch.tensor(1e-9, device=self.device))
        embeddings = sum_embeddings / sum_mask
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
