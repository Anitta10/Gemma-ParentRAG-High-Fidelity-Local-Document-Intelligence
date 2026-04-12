

# Gemma-ParentRAG: High-Fidelity Local Document Intelligence

**Gemma-ParentRAG** is a private, local-first RAG (Retrieval-Augmented Generation) system. Unlike standard RAG implementations that suffer from context fragmentation, this project utilizes a **Parent Document Retrieval** strategy. It ensures the LLM receives broad, meaningful context while maintaining high-precision search capabilities.

Built with **LangChain**, **Chainlit**, and powered by **Gemma 3** via **Ollama**.

## 🚀 Key Features

* **Parent-Document Retrieval:** Decouples retrieval (small child chunks) from generation (large parent chunks) for superior accuracy.
* **Fully Local & Private:** Powered by **Ollama** (Gemma 3 + Qwen-Embeddings). No data leaves your machine.
* **Async Ingestion Pipeline:** Uses Python's `asyncio` and thread executors to process PDFs in the background without freezing the UI.
* **Persistent Memory:** State-aware tracking of processed files and a persistent vector store (ChromaDB) to avoid redundant indexing.
* **History-Aware Conversations:** A specialized sub-chain reformulates user queries based on chat history for seamless multi-turn dialogue.

## 🛠️ Technical Architecture



1.  **Ingestion:** PDFs are loaded and split into "Parents" (2000 chars) and "Children" (400 chars).
2.  **Indexing:** Child chunks are vectorized and stored in **ChromaDB**. The mapping to Parent chunks is managed via a persistent **InMemoryStore**.
3.  **Retrieval:** The system searches for the most relevant child chunks but feeds the associated *Parent* chunks to the LLM.
4.  **UI:** **Chainlit** provides an interactive chat interface with real-time task tracking for document processing.

## 📦 Installation

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running.
* Required models:
    ```bash
    ollama pull gemma3:1b
    ollama pull qwen3-embedding:0.6b
    ```

### 2. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/Gemma-ParentRAG.git
cd Gemma-ParentRAG
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory:
```env
DATA_PATH="./data"
```
Place your PDF files inside the `./data` folder.

## 🖥️ Usage

Start the application using Chainlit:
```bash
chainlit run app.py
```

* The system will automatically detect new PDFs in the `/data` folder.
* The **Task List** in the UI will show the ingestion progress.
* Once indexed, you can ask complex questions about your documents.

## 📂 Project Structure

```text
├── app.py              # Main application logic & Chainlit hooks
├── data/               # Source PDF directory
├── .env                # Environment variables
├── Data/
│   ├── docs/           # Persistent docstore (pickle) & file tracker
│   └── vectors/        # ChromaDB vector storage
└── requirements.txt    # Python dependencies
```

## 🧠 Why Parent-Document Retrieval?
Standard RAG often retrieves tiny snippets of text that lack the necessary surrounding context to answer complex questions. By using a **ParentDocumentRetriever**, this system:
1.  Searches small chunks for better "keyword/concept" matching.
2.  Provides the LLM with the full paragraph or section (the Parent) to ensure the answer is grounded and complete.
