# DocuMind – AI Research Assistant

DocuMind is a local AI-powered tool that allows users to upload PDF or TXT files and:

- Ask free-form questions based on the document
- Generate logic-based questions and evaluate user answers
- Get a summary (within 150 words)

## Features

- Document upload (PDF/TXT)
- Ask Anything mode
- Challenge Me mode
- Justified answers with references from the document

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- PyPDF2

## How to Run

```bash
git clone https://github.com/001Diwakar001/DocuMind.git
cd DocuMind
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
