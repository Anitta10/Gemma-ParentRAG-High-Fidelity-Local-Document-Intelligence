import os
import pickle
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from typing import cast 

# LangChain Imports
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.stores import InMemoryStore

load_dotenv()

# --- 1. Configuration & Paths ---
BASE_DIR = r"C:/Users/SSLTP11229/Desktop/Projects/Chainlit/Data"
CHROMA_PATH = os.path.join(BASE_DIR, "vectors")
DOCSTORE_PATH = os.path.join(BASE_DIR, "docs")
PICKLE_PATH = os.path.join(DOCSTORE_PATH, "docstore.pkl")
TRACKER_FILE = os.path.join(DOCSTORE_PATH, "processed_files.txt")
DATA_PATH = os.getenv("DATA_PATH", "./data")

os.makedirs(DOCSTORE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# --- 2. Initialize Models & Vector Store ---
llm = ChatOllama(model="gemma3:1b", temperature=0)
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
store = InMemoryStore()

if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, "rb") as f:
        full_dict = pickle.load(f)
        store.mset(list(full_dict.items()))

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)

# --- 3. Async Ingestion Logic ---
async def ingest_new_documents():
    pdfs = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    processed_files = set()
    
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            processed_files = set(line.strip() for line in f if line.strip())

    new_pdfs = [f for f in pdfs if f not in processed_files]
    
    if not new_pdfs:
        return 0, len(processed_files)

    task_list = cl.TaskList()
    await task_list.send()
    
    loop = asyncio.get_running_loop()
    
    for pdf_name in new_pdfs:
        # Corrected Task Update Logic
        task = cl.Task(title=f"Processing {pdf_name}", status=cl.TaskStatus.RUNNING)
        await task_list.add_task(task)
        await task_list.send() # Push update to UI
        
        file_path = os.path.join(DATA_PATH, pdf_name)
        loader = PyPDFLoader(file_path)
        
        # Load and Index (Threaded to prevent UI freeze)
        docs = await loop.run_in_executor(None, loader.load)
        await loop.run_in_executor(None, retriever.add_documents, docs)
        
        processed_files.add(pdf_name)
        
        # Mark as Done
        task.status = cl.TaskStatus.DONE
        await task_list.send() # Push final update to UI

    # Final Persistence
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(store.store, f)
    with open(TRACKER_FILE, "w") as f:
        for name in sorted(processed_files):
            f.write(f"{name}\n")
            
    return len(new_pdfs), len(processed_files)

# --- 4. Chainlit Hooks ---

@cl.on_chat_start
async def start():
    new_count, total_count = await ingest_new_documents()
    
    ready_msg = f"✅ System ready. {total_count} documents in database."
    if new_count > 0:
        ready_msg = f"✅ Ingested {new_count} new PDF(s). Total: {total_count} documents."
    await cl.Message(content=ready_msg).send()

    # RAG Chain Setup
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the latest user question to be a standalone question based on history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context to answer. If it's not there, use your own knowledge accurately. Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = InMemoryChatMessageHistory()
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda s_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    cl.user_session.set("chain", conversational_rag_chain)

@cl.on_message
async def main(message: cl.Message):
    # Use cast to tell Pylance: "I guarantee this is a RunnableWithMessageHistory object"
    chain = cast(RunnableWithMessageHistory, cl.user_session.get("chain"))
    
    if not chain:
        await cl.Message(content="Session expired or chain not initialized.").send()
        return

    res = await chain.ainvoke(
        {"input": message.content},
        config={"configurable": {"session_id": cl.user_session.get("id")}}
    )
    
    answer = res["answer"]
    context = res.get("context", [])
    
    # Simple Source String
    sources = {os.path.basename(d.metadata.get('source', 'Unknown')) for d in context}
    source_text = f"\n\n**Sources:** {', '.join(sources)}" if sources else "\n\n**Source:** General Knowledge"

    await cl.Message(content=f"{answer}{source_text}").send()