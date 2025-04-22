# PDF File Analyzer

## Overview
This project is a Streamlit-based web application that allows users to upload PDF files, store their content in a vector database, and answer questions about the uploaded documents using a Retrieval-Augmented Generation (RAG) approach. Users can upload multiple PDFs, and the system will incrementally build a knowledge base to provide accurate answers based on the content of the uploaded files.

## Features
- **PDF Upload**: Upload PDF files to extract and process their content.
- **Vector Database**: Store document embeddings in an in-memory vector store for efficient retrieval.
- **Question Answering**: Ask questions about the uploaded PDFs and receive answers powered by OpenAI's GPT-4o-mini model.
- **Chat Interface**: Interactive chat interface to communicate with the AI and view conversation history.
- **Scalable Knowledge Base**: Add multiple PDFs to the database, enabling cumulative knowledge storage.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Framework for building the web application.
- **LangChain**: Tools for document processing, embeddings, and RAG-based question answering.
- **OpenAI**: GPT-4o-mini model for generating answers and text-embedding-3-large for embeddings.
- **PyPDFLoader**: For extracting text from PDF files.
- **InMemoryVectorStore**: For storing document embeddings.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-file-analyzer.git
   cd pdf-file-analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `requirements.txt` file with the following dependencies:
   ```
   streamlit
   langchain
   langchain-openai
   langchain-community
   pypdf
   ```

4. Obtain an OpenAI API key from [OpenAI](https://platform.openai.com/).

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. In the sidebar:
   - Enter your OpenAI API key.
   - Upload a PDF file.

4. Once the PDF is processed, use the chat input box to ask questions about the document's content.

5. Upload additional PDFs to expand the knowledge base and ask questions across all uploaded documents.

## How It Works
1. **PDF Processing**:
   - Uploaded PDFs are read and converted into text using `PyPDFLoader`.
   - The text is split into chunks using `RecursiveCharacterTextSplitter` for efficient processing.

2. **Vector Storage**:
   - Text chunks are converted into embeddings using OpenAI's `text-embedding-3-large` model.
   - Embeddings are stored in an `InMemoryVectorStore` for similarity search.

3. **Question Answering**:
   - User queries are processed using a similarity search to retrieve relevant document chunks.
   - The retrieved chunks are combined with the query and passed to the `gpt-4o-mini` model via a RAG prompt to generate accurate answers.

## Limitations
- Only PDF files are supported for upload.
- The vector store is in-memory, so the knowledge base resets when the application is restarted.
- Requires a valid OpenAI API key for embeddings and question answering.

## Future Improvements
- Support for additional file formats (e.g., DOCX, TXT).
- Persistent vector storage using a database like FAISS or Chroma.
- Enhanced error handling for invalid or corrupted PDFs.
- Multi-language support for document processing and question answering.