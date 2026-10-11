import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="MediBot - Medical Assistant", page_icon="🩺", layout="centered")

st.title("🩺 MediBot: Medical Assistant")
st.markdown("Ask medical questions based on the Gale Encyclopedia of Medicine.")

DB_FAISS_PATH = "vectorstore/db_faiss"
HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"

CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer , just say that you dont know , dont try to make up answer.
Dont provide anythin out the given context

Context : {context}
Question : {question}

Start the answer directly. No small talk please.
"""

@st.cache_resource(show_spinner="Loading medical database and models...")
def get_qa_chain():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN not found in environment variables or .env file.")
        st.stop()

    endpoint = HuggingFaceEndpoint(
        repo_id=HF_REPO_ID,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=endpoint)

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
if user_query := st.chat_input("Ask a medical question..."):
    # Display user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching medical database..."):
            try:
                response = qa_chain.invoke({"query": user_query})
                answer = response.get("result", "No answer found.")
                st.write(answer)

                # Show sources in an expander
                sources = response.get("source_documents", [])
                if sources:
                    with st.expander("View Reference Sources"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Source {i} (Page {page}):**")
                            st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error generating answer: {e}"
                st.error(error_msg)