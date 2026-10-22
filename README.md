# EduBot - Educational RAG Assistant

EduBot is an AI-powered educational assistant application built with Retrieval-Augmented Generation (RAG). It provides accurate, context-grounded answers to curriculum questions based on course notes and PDFs placed in `data_sources/` using LangChain, NVIDIA NIM / AI Foundation models, FAISS vector search, and Streamlit.

## Features

- Retrieval-Augmented Generation: Retrieves relevant reference passages from your notes before generating answers to minimize hallucination.
- Document Ingestion Pipeline: Automatically parses PDF course materials and text notes, splits them into semantic chunks, and creates vector embeddings.
- Vector Store Search: Uses FAISS (Facebook AI Similarity Search) with Sentence Transformers for fast similarity lookup.
- LLM Integration: Uses Meta Llama 3.2 11B Instruct through NVIDIA NIM endpoints (`meta/llama-3.2-11b-vision-instruct`) for natural language responses.
- Streamlit Web Interface: Interactive chat UI with conversational history, indexed course materials list, and reference source citations with page numbers.
- Command-Line Testing: Standalone terminal interface to test retrieval and prompt chains directly.

## Architecture

1. Ingestion: Course notes and reference documents (PDFs, TXT, MD) from `data_sources/` are loaded using LangChain's DirectoryLoader, PyPDFLoader, and TextLoader.
2. Chunking: Text is split into overlapping chunks (500 characters chunk size, 50 characters overlap) using RecursiveCharacterTextSplitter.
3. Embedding: Text chunks are converted to numerical vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4. Indexing: Vector representations are stored locally in a FAISS index.
5. Retrieval and Generation: When a query is received, the top-k relevant context chunks are fetched and passed with the query into Llama 3.1 8B Instruct via a custom educational prompt template.

## Project Structure

```text
data agent/
├── data/
│   ├── .env.example                                  # Template for environment configuration
│   ├── data_sources/                                 # Folder containing your course notes and PDFs
│   │   ├── unit 1.pdf
│   │   ├── unit 2 new.pdf
│   │   ├── unit 3.pdf
│   │   └── unit 4.pdf
│   ├── connect_memory_with_llm.py                    # CLI script for querying the RAG pipeline
│   ├── create_memory_for_llm.py                      # Pipeline script to generate vector index
│   ├── edubot.py                                     # Streamlit educational web application
│   ├── medibot.py                                    # Backwards-compatible launcher for edubot.py
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
- NVIDIA API Key (obtain free credits at [build.nvidia.com](https://build.nvidia.com))

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

Open `data/.env` and add your NVIDIA API key:

```env
NVIDIA_API_KEY=nvapi-your_nvidia_api_key_here
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
streamlit run edubot.py
```

(Note: `streamlit run medibot.py` will also launch the updated EduBot application for backward compatibility).

The application will start locally at `http://localhost:8501`. Enter your questions in the input field to receive grounded answers and view references with page numbers.

### 3. Run via Command-Line Interface

To query the assistant directly from the terminal:

```bash
cd data
python connect_memory_with_llm.py
```

Enter your query when prompted to view the generated answer and associated source documents.

## Configuration Details

- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM Model: `meta/llama-3.2-11b-vision-instruct` (via NVIDIA NIM)
- Text Chunker: RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
- Retriever: FAISS similarity search (top-k = 3)

