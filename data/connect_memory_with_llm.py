import os
from dotenv import load_dotenv
load_dotenv() 

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1 : Setup LLM (NVIDIA NIM)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct") 

def load_llm(model_name=NVIDIA_MODEL):
    return ChatNVIDIA(
        model=model_name,
        api_key=NVIDIA_API_KEY,
        temperature=0.5,
        max_completion_tokens=512,
    )

 
from pathlib import Path

# Step 2 : connect LLM with FAISS and create chain
BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = str(BASE_DIR / "vectorstore" / "db_faiss")
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up answer.
Provide a clear, educational, and accurate explanation based strictly on the provided study notes and course PDFs.
Dont provide anything outside the given context.

Context : {context}
Question : {question}

Start the answer directly. Explain key concepts clearly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])
    return prompt

# Load Database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(NVIDIA_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

# Now Invoke with single Query
user_query = input("Ask a question about your study notes / PDFs: ")
response = qa_chain.invoke({"query": user_query})
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(response.get("source_documents", []), 1):
    source = os.path.basename(doc.metadata.get("source", "Document"))
    page = doc.metadata.get("page", "N/A")
    display_page = page + 1 if isinstance(page, int) else page
    print(f"[{i}] {source} (Page {display_page})")