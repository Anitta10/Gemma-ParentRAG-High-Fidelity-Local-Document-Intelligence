import os
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.stores import InMemoryStore

load_dotenv()

# --- 1. Paths Configuration ---
BASE_DIR = r"C:/Users/SSLTP11486/Desktop/RAG project/ollama-with-rag/vectorstores"
CHROMA_PATH = os.path.join(BASE_DIR, "vectors")
DOCSTORE_PATH = os.path.join(BASE_DIR, "docs")
PICKLE_PATH = os.path.join(DOCSTORE_PATH, "docstore.pkl")
TRACKER_FILE = os.path.join(DOCSTORE_PATH, "last_processed.txt")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# --- 2. Initialize Models ---
llm = ChatOllama(model="gemma3:1b", temperature=0, num_predict=256)
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# --- 3. Initialize Storage & Retriever ---
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

store = InMemoryStore()

# Load existing parent docs if they exist
if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, "rb") as f:
        full_dict = pickle.load(f)
        store.mset(list(full_dict.items()))

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)

# --- 4. Logic for Clearing and Ingesting ---

def clear_data():
    """Wipes the Vectorstore, Docstore, and trackers for a fresh start."""
    print("🧹 Different PDF detected. Clearing old data stores...")
    
    # Clear Chroma
    all_ids = vectorstore.get()['ids']
    if all_ids:
        vectorstore.delete(ids=all_ids)
    
    # Clear Parent Store
    keys = list(store.yield_keys())
    if keys:
        store.mdelete(keys)

    # Delete physical files
    if os.path.exists(PICKLE_PATH):
        os.remove(PICKLE_PATH)
    print("✨ Reset complete.")

def ingest_documents():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    pdfs = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdfs:
        print(f"❌ No PDFs found in {DATA_PATH}. Please add a file.")
        return
    
    current_pdf_name = pdfs[0]

    # Check tracker file to see what was last indexed
    last_pdf = ""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            last_pdf = f.read().strip()

    # THE GATEKEEPER LOGIC
    if current_pdf_name != last_pdf:
        # This ONLY runs if a new/different PDF is detected
        clear_data()
        
        loader = PyPDFDirectoryLoader(DATA_PATH)
        docs = loader.load()
        
        print(f"🚀 Indexing NEW source: {current_pdf_name}...")
        for doc in tqdm(docs, desc="Progress", unit="page"):
            retriever.add_documents([doc])
        
        # Save Parent Docstore to disk
        if not os.path.exists(DOCSTORE_PATH): os.makedirs(DOCSTORE_PATH)
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(store.store, f)
            
        # Update Tracker with the current PDF name
        with open(TRACKER_FILE, "w") as f:
            f.write(current_pdf_name)
        print(f"✅ Success. {current_pdf_name} is now your active source.")
    else:
        # If the PDF is the same as last time, it does nothing and proceeds to chat
        print(f"✅ Source '{current_pdf_name}' is already indexed. Skipping ingestion.")

# Run the smart ingestion check
ingest_documents()

# --- 5. Chain Construction ---
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the latest user question to be a standalone question based on history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer. If the answer is not in the context, answer using your own knowledge accurately."),
    MessagesPlaceholder("chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history_store = {}
def get_session_history(session_id: str):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- 6. Chat Loop with Commands and Source Labeling ---

print("\n🟢 --- RAG Chatbot Online (Gemma 3 + Qwen 3) ---")
print("Commands: 'exit' to quit, 'clear' to reset chat history.\n")
config = {"configurable": {"session_id": "1"}}

# THRESHOLD ADJUSTMENT:
# For Qwen 3 embeddings, a distance of 0.4 - 0.6 is a strong match.
# If it says 'AI Generated' too often, INCREASE this to 0.7.
# If it shows PDF sources for unrelated questions, DECREASE this to 0.4.
DISTANCE_THRESHOLD = 0.65 

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if user_input.lower() == 'clear':
        chat_history_store["1"] = InMemoryChatMessageHistory()
        print("🧹 Chat history has been cleared.")
        continue
    
    print("Thinking...")
    try:
        # 1. Get response from Gemma 3
        result = conversational_rag_chain.invoke({"input": user_input}, config=config)
        print(f"\nAI: {result['answer']}")
        
        # 2. Fix the error by using similarity_search_with_score
        # This bypasses the 'relevance_score' normalization error.
        search_results = vectorstore.similarity_search_with_score(user_input, k=1)
        
        is_related_to_pdf = False
        current_distance = 999.0 # Default high distance
        
        if search_results:
            doc, score = search_results[0]
            current_distance = score
            # Lower distance = More similar
            if current_distance <= DISTANCE_THRESHOLD:
                is_related_to_pdf = True

        # 3. Display the Source Label
        if is_related_to_pdf:
            retrieved_docs = result.get('context', [])
            sources = set(os.path.basename(d.metadata.get('source', 'Unknown')) for d in retrieved_docs)
            print(f"Source: {', '.join(sources)} (Match Score: {current_distance:.3f})")
        else:
            print(f"Source: AI Generated Content (Match Score: {current_distance:.3f} - Outside Threshold)")

    except Exception as e:
        print(f"Error: {e}")