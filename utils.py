# main.py

import streamlit as st
import PyPDF2
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import random

# Page config
st.set_page_config(page_title="DocuMind – AI Research Assistant", page_icon="🧠")
st.title("🧠 DocuMind – AI Research Assistant")
st.write("Upload a PDF or TXT document and start exploring!")

# Session state initialization
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Helper functions
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_txt(file):
    return str(file.read(), 'utf-8')

def get_document_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return read_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file format")

def generate_summary(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary[:150] + "..." if len(summary) > 150 else summary

def setup_qa_system(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    st.session_state.vectorstore = vectorstore

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    st.session_state.qa_chain = qa_chain

def generate_questions(text, num=3):
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
    selected = random.sample(sentences, min(num, len(sentences)))
    questions = [f"What is the significance of: '{s[:100]}...'" for s in selected]
    return questions, selected

# File upload section
uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Processing your document..."):
        raw_text = get_document_text(uploaded_file)
        st.session_state.raw_text = raw_text
        summary = generate_summary(raw_text)
        st.subheader("📄 Summary:")
        st.write(summary)

        setup_qa_system(raw_text)

    # Ask Anything Mode
    st.subheader("❓ Ask Anything")
    user_question = st.text_input("Ask a question about the document:")

    if user_question and st.session_state.qa_chain:
        response = st.session_state.qa_chain({"query": user_question}, return_only_outputs=False)
        st.write("🤖 Answer:", response["result"])
        source = response.get("source_documents", [])
        if source:
            st.markdown("🔍 **Justification from Document:**")
            for doc in source[:2]:
                st.text(doc.page_content)

    # Challenge Me Mode
    st.subheader("💥 Challenge Me")
    if st.button("Generate Questions"):
        questions, answers = generate_questions(st.session_state.raw_text, num=3)
        st.session_state.questions = questions
        st.session_state.answers = answers
        st.session_state.user_answers = []
        st.session_state.correct_count = 0

    if "questions" in st.session_state:
        for i, q in enumerate(st.session_state.questions):
            user_ans = st.text_input(f"Q{i+1}: {q}", key=f"ans_{i}")
            st.session_state.user_answers.append(user_ans)

        if st.button("Submit Answers"):
            st.write("✅ Evaluation Results:")
            for i, ans in enumerate(st.session_state.user_answers):
                st.markdown(f"**Q{i+1} Correct Answer:** {st.session_state.answers[i][:200]}...")