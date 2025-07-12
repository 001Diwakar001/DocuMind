## 🧠 DocuMind – AI Research Assistant

DocuMind is a locally hosted AI-powered assistant that reads research documents and performs deep reasoning to answer questions, generate logic-based challenges, and provide concise summaries — all grounded in actual document content.

---

### 📌 Features

- **📄 Upload Support**: Accepts PDF and TXT documents.
- **🧠 Ask Anything Mode**: Free-form Q&A with contextual answers from the uploaded document.
- **🧹 Challenge Me Mode**: Automatically generates 3 reasoning-based questions, evaluates user responses, and justifies feedback.
- **📝 Auto Summary**: Instantly generates a 150-word summary of the uploaded document.
- **📍 Justified Responses**: Every answer includes a citation from the document (e.g., "As stated in section 2...").

> ✅ Bonus Features Included:
>
> - Memory support for follow-up questions.
> - Highlighted snippets from source documents to justify answers.

---

### 🛠️ Tech Stack

- **Frontend**: Streamlit (clean, responsive UI)
- **Backend**: Python with LangChain, FAISS, PyPDF2
- **LLM Integration**: HuggingFace Pipeline or OpenAI (pluggable)
- **Vector Store**: FAISS for semantic retrieval

---

### 🧱 Architecture

```mermaid
graph TD
    A[User Uploads PDF/TXT] --> B[Text Extraction (PyPDF2)]
    B --> C[Text Chunking (LangChain Splitter)]
    C --> D[Vector Embedding (FAISS)]
    D --> E[Interaction Mode Selection]

    E --> F1[Ask Anything Mode]
    E --> F2[Challenge Me Mode]

    F1 --> G1[Question Answering via RetrievalQA Chain]
    F2 --> G2[Question Generation + Evaluation Logic]

    G1 --> H1[Answer + Justification + Highlight]
    G2 --> H2[User Response Evaluation + Feedback]
```

---

### 🧪 How to Run Locally

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/documind-genai
cd documind-genai
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the App

```bash
streamlit run main.py
```

---

### 📷 Demo Walkthrough

> *(Optional)* Link to [YouTube/Loom video](https://your-demo-link.com) showcasing how the app works in under 2–3 mins.

---

### ✅ Submission Checklist

-

---

### 📊 Evaluation Criteria Mapping

| Criteria                          | Implementation Status           |
| --------------------------------- | ------------------------------- |
| Response Accuracy & Justification | ✅ Implemented                   |
| Reasoning Mode                    | ✅ Logic questions + feedback    |
| UI/UX                             | ✅ Intuitive Streamlit interface |
| Code Structure & Docs             | ✅ Modular + This README         |
| Bonus Features                    | ✅ Memory & Highlighting         |
| Contextual Use, No Hallucination  | ✅ Verified via doc-grounding    |

---

### 📬 Contact

For any queries, feel free to reach out:\
📧 **[**[**your.email@example.com**](mailto\:your.email@example.com)**]**

