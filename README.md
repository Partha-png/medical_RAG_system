# Medical RAG System

Medical RAG System is an intelligent document retrieval and question-answering platform designed specifically for medical literature. It leverages domain-specific language models and vector similarity search to provide accurate, context-aware answers to medical queries by retrieving relevant information from uploaded medical documents.

The project was inspired by a personal goal: to bridge the gap between vast medical knowledge repositories and healthcare professionals who need quick, reliable access to specific information, while exploring the capabilities of Retrieval-Augmented Generation (RAG) with biomedical language models.

## ✨ Features

🏥 **Medical Document Processing**: Supports PDF and TXT medical literature with intelligent chunking  
🧠 **Domain-Specific Encoders**: BioBERT and MedCPT models optimized for biomedical text  
🔍 **Efficient Vector Search**: FAISS-powered similarity search with adaptive indexing  
💬 **Interactive Chat Interface**: Streamlit-based UI with conversation history  
⚡ **Configurable Retrieval**: Adjustable chunk retrieval (k=1-10) for precision control  
🔗 **LLM Integration**: Groq API integration for natural language generation

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical_RAG_system.git
cd medical_RAG_system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add your API key:
```
GROQ_API_KEY=your_groq_api_key
```

5. Run the application:
```bash
streamlit run app.py
```

## 🏗️ Project Structure
```
medical_RAG_system/
├── information_retrieval/           # Document processing and retrieval
│   ├── document_encoding/          # Encoding components
│   │   ├── bioBERT_encoder.py     # BioBERT embedding model
│   │   ├── medcpt_encoder.py      # MedCPT embedding model
│   │   ├── chunker.py             # Token-based text chunking
│   │   ├── encoder.py             # Main encoding orchestrator
│   │   └── faiss_manager.py       # FAISS index management
│   └── faiss_container/            # Stored vector indices
├── rag_system/                     # RAG pipeline components
│   ├── bioBERTqueryencoder.py     # BioBERT query encoding
│   ├── biobertretriever.py        # BioBERT retrieval engine
│   ├── medcptqueryencoder.py      # MedCPT query encoding
│   ├── medcptretriever.py         # MedCPT retrieval engine
│   └── rag_pipeline.py            # Main RAG orchestrator
├── temp_uploads/                   # Temporary file storage
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── .gitignore                      # Git ignore file
├── LICENSE                         # Apache 2.0 License
└── README.md                       # This file
```

## 🤖 How It Works

### 1. Document Processing:
- Uploads medical documents (PDF/TXT)
- Extracts text with page-level granularity
- Applies intelligent chunking with token overlap (150 tokens, 100 overlap)
- Preserves context across chunk boundaries

### 2. Embedding Generation:
- Encodes chunks using BioBERT or MedCPT models
- Generates normalized 768-dimensional embeddings
- Processes in configurable batches for memory efficiency
- Stores embeddings in FAISS index

### 3. Retrieval System:
- Converts user queries to embeddings using matched encoder
- Performs efficient similarity search in FAISS
- Returns top-k most relevant chunks
- Supports both Flat and IVF index types based on dataset size

### 4. Answer Generation:
- Formats retrieved chunks as context
- Sends query + context to Groq LLM (GPT-OSS-120B)
- Generates medically accurate responses
- Maintains conversation history for follow-up questions

## 🎯 Usage

### Processing a Document

1. Navigate to the **Upload & Process** tab
2. Upload a medical document (PDF or TXT format)
3. Select your preferred encoder model (BioBERT or MedCPT)
4. Click **Process Document** and wait for indexing to complete

### Asking Questions

1. Switch to the **Chat** tab
2. Adjust the number of chunks to retrieve (k) using the sidebar slider
3. Type your medical question in the chat input
4. View the AI-generated answer based on your document
5. Expand "View Retrieved Chunks" to see source passages

### Configuration Options

- **Encoder Model**: Choose between BioBERT (general biomedical) or MedCPT (clinical text optimized)
- **Chunks to Retrieve**: Set k=1-10 to balance between precision and context
- **Reset Session**: Clear all data and start fresh

## 📊 Technical Details

### Encoder Models

**BioBERT** (`dmis-lab/biobert-v1.1`)
- Finetuned Bert Model on PubMed abstracts and PMC full-text articles
- Optimized for biomedical named entity recognition and relation extraction
- Best for: General medical literature, research papers, drug information

**MedCPT** (`ncbi/MedCPT-Article-Encoder`)
- Contrastive Pre-trained Transformer for medical information retrieval
- Trained on PubMed articles with query-document pairs
- Best for: Clinical notes, patient information, diagnostic queries

### Chunking Strategy

The system uses token-based chunking with overlap:
- **Max tokens per chunk**: 150
- **Overlap**: 100 tokens
- **Benefits**: Preserves context, prevents information loss at boundaries
- **Implementation**: Sentence-aware splitting to maintain semantic coherence

### FAISS Indexing

Adaptive index selection based on document size:
- **Small datasets (<1000 chunks)**: `IndexFlatL2` for exact search
- **Large datasets (≥1000 chunks)**: `IndexIVFFlat` for faster approximate search
- **Distance metric**: L2 (Euclidean) on normalized embeddings

## 📖 Research Inspiration

Medical RAG System's architecture is influenced by recent advances in biomedical NLP and retrieval systems:

[1] "Efficient and Reproducible Biomedical Question Answering using RAG" - Stuhlmann et al., 2025  
[2] "Context-Aware RAG Using Similarity Validation to Handle Context Inconsistencies in LLMs" - Collini et al., IEEE Access, 2025  
[3] "Enhancing the Precision and Interpretability of RAG in Legal Technology: A Survey" - Hindi et al., IEEE Access, 2025

The system applies insights from these studies by:

- Using domain-specific pre-trained models (BioBERT, MedCPT) for specialized text understanding
- Implementing dense retrieval with FAISS for efficient similarity search
- Planning hybrid retrieval integration (BM25 + dense embeddings) for improved precision
- Incorporating conversation history for contextual question answering
- Maintaining source attribution through chunk retrieval for verifiable answers

## 🚀 Future Enhancements

### Planned Features
- [ ] **Hybrid Retrieval System**: Integration of BM25 (sparse retrieval) with dense embeddings for improved recall and precision
- [ ] **Elasticsearch Integration**: Scalable full-text search capabilities for handling larger medical corpora
- [ ] **Conversation Memory Enhancement**: Advanced chat history utilization with context-aware follow-up handling
- [ ] **Multi-document Knowledge Base**: Support for querying across multiple medical documents simultaneously
- [ ] **Citation Tracking**: Document-level and paragraph-level attribution for verifiable medical claims
- [ ] **Cross-encoder Reranking**: MedCPT cross-encoder for refined document ranking post-retrieval

### Research-Backed Improvements
Based on recent advances in medical RAG systems (Stuhlmann et al., 2025; Collini et al., 2025):
- [ ] **Adaptive Retrieval Depth**: Dynamic adjustment of top-k chunks based on query complexity
- [ ] **Query Rewriting**: Automatic reformulation of medical queries for better retrieval accuracy
- [ ] **Context-Aware Post-processing**: Similarity validation between generated answers and source chunks to reduce hallucinations
- [ ] **Semantic Chunking**: Advanced chunking strategies that preserve medical context boundaries
- [ ] **Retrieval Evaluation Metrics**: Implementation of context precision, recall, and faithfulness scoring

### Advanced Features
- [ ] Support for medical imaging integration (multimodal RAG)
- [ ] Export functionality for consultation sessions
- [ ] Fine-tuning on specialized medical Q&A datasets (e.g., MedQA, PubMedQA)
- [ ] IVF-based FAISS indexing for corpora exceeding 100K documents
- [ ] Real-time document updates with incremental indexing

## 🛡️ License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with **Streamlit** for the web interface
- Powered by **Groq API** for LLM inference
- **BioBERT** from DMIS Lab, Korea University
- **MedCPT** from NCBI/NLM
- Vector search via **FAISS** (Facebook AI Similarity Search)
- Document processing with **PyPDF2**
- Transformers library by **Hugging Face**
