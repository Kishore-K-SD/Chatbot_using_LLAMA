# Running Llama 2 and other Open-Source LLMs on CPU Inference Locally for Video Q&A

## Problem Statement
Interacting with unstructured data, such as videos, and deriving meaningful insights often involves manually sifting through vast amounts of information. This process is not only time-consuming but also inefficient, especially for users who need quick and accurate responses based on the provided documents. Existing tools either lack flexibility in handling different file formats or fail to offer conversational engagement powered by advanced language models.

This repository addresses these challenges by leveraging a conversational chatbot built with Streamlit and LangChain components. It integrates Llama2 for natural language understanding and retrieval-based answering, offering a seamless way to query uploaded data interactively.

## Features
- **File Upload:** Upload videos, PDFs, CSVs, or text files via a user-friendly Streamlit interface.
- **Video to Text Conversion:** Automatically transcribes uploaded videos into CSV format using Whisper.
- **Conversational Chat:** Ask natural language questions and get context-aware responses.
- **Context Management:** Maintains chat history and trims it intelligently to stay within token limits.
- **Multiple File Formats:** Supports videos (converted to CSV), CSV, plain text, and PDF formats.
- **Custom LLM Integration:** Utilizes Llama2 with retrieval-augmented generation for accurate answers.
- **Dynamic Summarization:** Summarizes chat history to optimize token usage while retaining context.

## Prerequisites
To run this application, you need the following:

- Python 3.9 or later
- Streamlit: For the interactive user interface.
- LangChain: For document loading, embeddings, and retrieval.
- HuggingFace Transformers: For tokenization and embeddings.
- FAISS: For efficient vector storage and retrieval.
- Llama2 Model: Pre-trained Llama2 model files.
- Whisper: For video-to-text transcription.

## Installing Dependencies
Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

Ensure you have the pre-trained Llama2 model file (e.g., `llama-2-7b-chat.ggmlv3.q8_0.bin`) and place it in the `models/` directory.

## How to Run the App
1. Clone or download this repository to your local machine.
2. Install the required dependencies (see above).
3. Launch the Streamlit app by running:

```bash
streamlit run main.py
```

4. Open the app in your browser and upload a file to start chatting.

## Example Workflow
1. **Upload a File:** Upload a video, PDF, CSV, or text file using the Streamlit sidebar.
2. **Video Processing:** If a video is uploaded, it is transcribed into a CSV file using Whisper.
3. **Ask Questions:** Type your query in the chat box to interact with the uploaded data.
4. **Retrieve Answers:** Get accurate and context-aware responses based on the content of the uploaded file.
5. **Iterative Queries:** Continue asking questions, with the chatbot maintaining context for a coherent conversation.

## Project Structure
```
├── /config                 # Configuration files for LLM application
├── /models                 # Binary file of GGML quantized LLM model (i.e., Llama-2-7B-Chat)
├── /src                    # Python codes of key components of LLM application, namely `llm.py`, `utils.py`, and `prompts.py`
├── /vectorstore            # FAISS vector store for documents
├── db_build.py             # Python script to ingest dataset and generate FAISS vector store
├── main.py                 # Main Python script to launch the application and to pass user query via command line
├── pyproject.toml          # TOML file to specify which versions of the dependencies used (Poetry)
├── requirements.txt        # List of Python dependencies (and version)
```

## Key Functions
### Chatbot Core Functions
- **load_llm():** Loads the Llama2 model with specified configurations.
- **calculate_token_count(text):** Calculates token count for a given text using HuggingFace's tokenizer.
- **truncate_history(history, max_tokens):** Keeps chat history within the token limit by truncating older entries.

### File Handling Functions
- **CSVLoader, TextLoader, PyPDFLoader:** Load data from CSV, plain text, or PDF files.
- **VideoToCSV:** Transcribes videos to CSV format using Whisper.

### Retrieval Chain
- **RetrievalQA:** Combines Llama2 with FAISS-based document retrieval for accurate, context-aware answers.
___
