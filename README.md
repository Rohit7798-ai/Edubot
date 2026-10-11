# MediBot - Medical RAG Assistant

MediBot is an AI-powered medical assistant application built with Retrieval-Augmented Generation (RAG). It provides accurate, context-grounded answers to medical queries based on reference literature (The Gale Encyclopedia of Medicine) using LangChain, Hugging Face models, FAISS vector search, and Streamlit.

## Features

- Retrieval-Augmented Generation: Retrieves relevant reference passages before generating answers to minimize hallucination.
- Document Ingestion Pipeline: Automatically parses PDF medical texts, splits them into semantic chunks, and creates vector embeddings.
- Vector Store Search: Uses FAISS (Facebook AI Similarity Search) with Sentence Transformers for fast similarity lookup.
- LLM Integration: Uses Meta Llama 3.1 8B Instruct through Hugging Face endpoints for natural language responses.
- Streamlit Web Interface: Interactive chat UI with conversational history and reference source citations with page numbers.
- Command-Line Testing: Standalone terminal interface to test retrieval and prompt chains directly.

## Architecture

1. Ingestion: Raw medical reference documents (PDFs) are loaded using LangChain's DirectoryLoader and PyPDFLoader.
2. Chunking: Text is split into overlapping chunks (500 characters chunk size, 50 characters overlap) using RecursiveCharacterTextSplitter.
3. Embedding: Text chunks are converted to numerical vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4. Indexing: Vector representations are stored locally in a FAISS index.
5. Retrieval and Generation: When a query is received, the top-k relevant context chunks are fetched and passed with the query into Llama 3.1 8B Instruct via a custom prompt template.

## Project Structure

```text
data agent/
├── data/
│   ├── .env.example                                  # Template for environment configuration
│   ├── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf  # Source medical reference document
│   ├── connect_memory_with_llm.py                    # CLI script for querying the RAG pipeline
│   ├── create_memory_for_llm.py                      # Pipeline script to generate vector index
│   ├── medibot.py                                    # Streamlit web application
│   ├── Pipfile                                       # Pipenv dependency specifications
│   ├── Pipfile.lock                                  # Locked dependency versions
│   └── vectorstore/
│       └── db_faiss/                                 # Serialized FAISS vector index
├── .gitignore                                        # Git ignore rules
├── requirements.txt                                  # Python dependencies
└── README.md                                         # Project documentation
```

## Prerequisites

- Python 3.10 or higher (Python 3.11 / 3.12 / 3.13 supported)
- Hugging Face Account and Access Token with read permissions
- Granted access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face (or another compatible model endpoint)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Rohit7798-ai/medibot.git
cd medibot
```

### 2. Create and Activate a Virtual Environment

Using venv:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

Using Pipenv:

```bash
cd data
pipenv install
pipenv shell
```

### 3. Install Dependencies

If using standard pip:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the `data/` directory by copying the example template:

```bash
cp data/.env.example data/.env
```

Open `data/.env` and add your Hugging Face API token:

```env
HF_TOKEN=your_huggingface_access_token_here
```

## Usage

### 1. Build or Rebuild the Vector Store (Optional if pre-built index exists)

To process the PDF documents and create the FAISS vector index:

```bash
cd data
python create_memory_for_llm.py
```

This will read the PDF files in the directory, compute embeddings, and save the index to `vectorstore/db_faiss/`.

### 2. Run the Streamlit Web Application

To launch the web interface:

```bash
cd data
streamlit run medibot.py
```

The application will start locally at `http://localhost:8501`. Enter your medical questions in the input field to receive grounded answers and view references.

### 3. Run via Command-Line Interface

To query the assistant directly from the terminal:

```bash
cd data
python connect_memory_with_llm.py
```

Enter your query when prompted to view the generated answer and associated source documents.

## Configuration Details

- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM Model: `meta-llama/Llama-3.1-8B-Instruct`
- Text Chunker: RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
- Retriever: FAISS similarity search (top-k = 3)

## Disclaimer

This software is for educational, research, and informational purposes only. It is not intended as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified physician or healthcare provider with any medical questions.
