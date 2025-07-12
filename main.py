# main.py

import streamlit as st
import PyPDF2
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline as hf_pipeline
import random

# Configuring the page
st.set_page_config(page_title="DocuMind – AI Research Assistant", page_icon="🧠", layout="wide")

# Styling (Custom CSS)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1596865249308-2472dc5807d7?q=80&w=1806&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    position: relative;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: -1;
}
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: white;
}
input[type="text"], textarea {
    background: rgba(255,255,255,0.1);
    border: 1px solid #ccc;
    padding: 10px;
    color: white;
    border-radius: 8px;
}
.stButton > button {
    background-color: #28a745;
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
}
.stButton > button:hover {
    background-color: #218838;
}
.summary-box, .justification-box {
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 12px;
    margin: 15px 0;
    border-left: 6px solid #28a745;
    backdrop-filter: blur(10px);
}
.justification-box {
    border-left-color: #007BFF;
    font-style: italic;
}
h2, .stSubheader {
    color: #f1f1f1;
    text-shadow: 1px 1px 2px black;
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
<div style="background-color: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 8px; font-size: 24px; font-weight: bold;">
    🧠 DocuMind – <span style="color:#28a745;">AI Research Assistant</span>
</div>
""", unsafe_allow_html=True)

st.markdown("Upload any PDF or text file to get started!")

# Initialize session variables
for key in ["raw_text", "qa_chain", "vectorstore", "questions", "answers", "user_answers"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "answers" in key else None

# File readers
def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    result = ""
    for pg in reader.pages:
        content = pg.extract_text()
        if content:
            content = content.replace('\n', ' ')
            result += ' '.join(content.split()) + "\n\n"
    return result

def load_txt(file):
    return file.read().decode('utf-8')

def parse_file(uploaded):
    if uploaded.type == "application/pdf":
        return load_pdf(uploaded)
    elif uploaded.type == "text/plain":
        return load_txt(uploaded)
    else:
        raise Exception("Only PDF and TXT files are allowed.")

# Summary generator
def create_summary(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    try:
        summarizer = hf_pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=summarizer)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        return summary[:150] + "..." if len(summary) > 150 else summary
    except Exception as e:
        return f"Error while summarizing: {e}"

# QA setup
def initialize_qa(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(chunks, embeddings)
    st.session_state.vectorstore = vs

    try:
        qa_pipe = hf_pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=qa_pipe)

        qa_system = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vs.as_retriever(),
            return_source_documents=True
        )
        st.session_state.qa_chain = qa_system
    except Exception as e:
        st.error(f"Couldn't initialize Q&A: {e}")

# Question creator
def make_questions(text, count=3):
    sent = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
    chosen = random.sample(sent, min(count, len(sent)))
    generated = [f"What does this part mean: '{s[:100]}...'" for s in chosen]
    return generated, chosen

# File uploader UI
uploaded_file = st.file_uploader("📄 Upload PDF or TXT", type=["pdf", "txt"], key="fileUploader")

if uploaded_file:
    with st.spinner("Reading the file..."):
        content = parse_file(uploaded_file)
        st.session_state.raw_text = content
        summary = create_summary(content)
        st.subheader("📌 Summary:")
        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
        initialize_qa(content)

    # Ask Anything Section
    st.subheader("❓ Ask Anything")
    query = st.text_input("Ask something:")

    if query and st.session_state.qa_chain:
        result = st.session_state.qa_chain({"query": query}, return_only_outputs=False)
        ans = result["result"]
        src = result.get("source_documents", [])

        st.markdown("🤖 **Answer:**")
        st.markdown(f"> {ans}")

        if src:
            st.markdown("📌 **Based on:**")
            for doc in src[:2]:
                st.markdown(f"<div class='justification-box'>{doc.page_content}</div>", unsafe_allow_html=True)

    # Challenge Me Section
    st.subheader("💥 Challenge Me")
    if st.button("Generate Questions", key="btn_gen_qs"):
        qs, ans_list = make_questions(st.session_state.raw_text, 3)
        st.session_state.questions = qs
        st.session_state.answers = ans_list
        st.session_state.user_answers = []

    if st.session_state.questions:
        st.markdown("📝 Try to answer these:")
        for idx, q in enumerate(st.session_state.questions):
            response = st.text_input(f"Q{idx+1}: {q}", key=f"ua_{idx}")
            st.session_state.user_answers.append(response)

        if st.button("Submit Answers", key="btn_submit"):
            st.markdown("✅ **Check your answers:**")
            if not st.session_state.answers:
                st.warning("No answers found. Try regenerating.")
            else:
                for idx, correct in enumerate(st.session_state.answers):
                    st.markdown(f"✔️ Q{idx+1} Expected Answer: {correct[:200]}...")
