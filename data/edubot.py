import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = str(BASE_DIR / "vectorstore" / "db_faiss")
DATA_SOURCES_DIR = BASE_DIR / "data_sources"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")

st.set_page_config(
    page_title="EduBot - Educational Study Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 EduBot: Educational Study Assistant")
st.markdown("Ask conceptual and curriculum questions based on your provided study notes and course PDFs.")

# Sidebar for course materials information
with st.sidebar:
    st.header("📚 Course Materials")
    if DATA_SOURCES_DIR.is_dir():
        materials = [f.name for f in DATA_SOURCES_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
        if materials:
            st.markdown("Indexed study files:")
            for item in sorted(materials):
                st.markdown(f"- 📄 `{item}`")
        else:
            st.info("No documents found in `data_sources/`.")
    else:
        st.warning("`data_sources/` folder not found.")

    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("- Ask for definitions, explanations, formulas, or summaries.")
    st.markdown("- Specify unit or topic names if you want focused answers.")

CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer the student's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a clear, educational, and accurate explanation based strictly on the provided study notes and course materials.
Do not provide anything outside the given context.

Context : {context}
Question : {question}

Start the answer directly. Explain key concepts clearly. No small talk please.
"""

@st.cache_resource(show_spinner="Loading educational knowledge base and models...")
def get_qa_chain():
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_api_key:
        st.error("NVIDIA_API_KEY not found in environment variables or .env file. Please add your NVIDIA API key.")
        st.stop()

    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Vector store not found at `{DB_FAISS_PATH}`. Please run `python create_memory_for_llm.py` first.")
        st.stop()

    llm = ChatNVIDIA(
        model=NVIDIA_MODEL,
        api_key=nvidia_api_key,
        temperature=0.3,
        max_completion_tokens=512,
        timeout=120,
        max_retries=3,
    )

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

qa_chain = get_qa_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if user_query := st.chat_input("Ask a question about your study notes or PDFs..."):
    # Display user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching study notes and course materials..."):
            try:
                response = qa_chain.invoke({"query": user_query})
                answer = response.get("result", "No answer found.")
                st.write(answer)

                # Show sources in an expander
                sources = response.get("source_documents", [])
                if sources:
                    with st.expander("📖 View Reference Sources"):
                        for i, doc in enumerate(sources, 1):
                            source_file = os.path.basename(doc.metadata.get("source", "Document"))
                            page = doc.metadata.get("page", "N/A")
                            page_str = f"Page {page + 1}" if isinstance(page, int) else f"Page {page}"
                            st.markdown(f"**Source {i} — `{source_file}` ({page_str}):**")
                            st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error generating answer: {e}"
                st.error(error_msg)
