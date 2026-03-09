README.md

# Gemma-ParentRAG: High-Fidelity Local Document Intelligence

**Gemma-ParentRAG** is an advanced Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Gemma 3**, and **Qwen 3 Embeddings**. By utilizing a **Parent Document Retrieval** strategy, this system provides superior contextual accuracy while maintaining the speed of localized vector searches.

---

## 🌟 Key Features

* **Parent-Child Splitting Strategy**:
* **Child Chunks (400 chars)**: Optimized for high-speed, granular vector similarity search.
* **Parent Documents (2000 chars)**: Reconstructs the broader context to provide the LLM (Gemma 3) with a complete understanding of the topic.


* **Smart Ingestion Engine**: Automatically detects when a new PDF is added to the directory. It wipes the old vector store and re-indexes the new source to prevent data cross-contamination.
* **Intelligent Source Labeling**: Uses Euclidean distance scoring to determine if an answer is grounded in your document.
* **Match Score $\le$ 0.65**: Attributed to the local PDF source.
* **Match Score > 0.65**: Labeled as "AI Generated" to ensure transparency.


* **Stateful Conversations**: Full chat history support using `InMemoryChatMessageHistory`, allowing for follow-up questions.

---

## 🛠️ Tech Stack

* **LLM**: `gemma3:1b` (via Ollama)
* **Embeddings**: `qwen3-embedding:0.6b` (via Ollama)
* **Orchestration**: LangChain
* **Vector Database**: ChromaDB
* **Storage**: InMemoryStore with Pickle persistence

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure [Ollama](https://ollama.ai/) is installed and running. Download the required models:

```bash
ollama pull gemma3:1b
ollama pull qwen3-embedding:0.6b

```

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt

```

### 3. Configuration

Create a `.env` file in the root directory:

```env
DATA_PATH=./data

```

### 4. Usage

Place your PDF files in the `/data` folder and launch the application:

```bash
python main.py


## 📂 Project Structure

* `main.py`: The core logic for ingestion, retrieval, and the chat loop.
* `/vectorstores`: Persistent storage for ChromaDB and the Parent Docstore.
* `/data`: The input directory for your PDF documents.
* `requirements.txt`: Necessary Python libraries.



