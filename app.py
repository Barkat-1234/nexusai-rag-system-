import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import shutil
from datetime import datetime

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat area */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][aria-label="user"] {
        background-color: #2c2c2c;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"][aria-label="assistant"] {
        background-color: #1e1e1e;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #2c2c2c;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00b4ff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# ========== FUNCTIONS ==========
def process_documents(uploaded_files):
    with st.spinner("📄 Processing your documents..."):
        try:
            temp_dir = tempfile.mkdtemp()
            documents = []
            
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                if file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                documents.extend(loader.load())
                st.session_state.uploaded_files_info.append({
                    "name": file.name,
                    "size": f"{file.size/1024:.1f} KB",
                    "type": "PDF" if file.name.endswith('.pdf') else "Text",
                    "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
            vectordb.persist()
            
            st.session_state.vectordb = vectordb
            st.session_state.documents_processed = True
            
            return True, len(documents), len(chunks)
            
        except Exception as e:
            return False, str(e)

def create_new_conversation():
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.conversations[conv_id] = {
        "name": f"Conversation {len(st.session_state.conversations) + 1}",
        "messages": [],
        "created": datetime.now()
    }
    st.session_state.current_conversation = conv_id
    st.session_state.messages = []

def load_conversation(conv_id):
    st.session_state.current_conversation = conv_id
    st.session_state.messages = st.session_state.conversations[conv_id]["messages"]

def delete_conversation(conv_id):
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation == conv_id:
            st.session_state.current_conversation = None
            st.session_state.messages = []

def clear_all_conversations():
    st.session_state.conversations = {}
    st.session_state.current_conversation = None
    st.session_state.messages = []

# ========== SIDEBAR ==========
with st.sidebar:
    # Logo/Brand
    st.markdown("### 🤖 NexusAI")
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_conversation()
        st.rerun()
    
    st.markdown("---")
    
    # Conversations History
    st.markdown("### 📝 Conversations")
    
    if st.session_state.conversations:
        for conv_id, conv_data in st.session_state.conversations.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"💬 {conv_data['name']}", key=f"conv_{conv_id}", use_container_width=True):
                    load_conversation(conv_id)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    delete_conversation(conv_id)
                    st.rerun()
    else:
        st.info("No conversations yet. Start a new chat!")
    
    st.markdown("---")
    
    # Document Management
    st.markdown("### 📚 Document Library")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                success, *result = process_documents(uploaded_files)
                if success:
                    st.success(f"✅ Processed {result[0]} documents into {result[1]} chunks!")
                    st.rerun()
                else:
                    st.error(f"Error: {result[0]}")
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.vectordb = None
                st.session_state.documents_processed = False
                st.session_state.uploaded_files_info = []
                st.rerun()
    
    # Document List
    if st.session_state.uploaded_files_info:
        st.markdown("#### 📄 Active Documents:")
        for doc in st.session_state.uploaded_files_info:
            st.markdown(f"- **{doc['name']}** ({doc['size']})")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    # Model selection
    model_options = ["models/gemini-2.5-flash", "models/gemini-2.0-flash-lite", "models/gemini-pro-latest"]
    selected_model = st.selectbox("AI Model", model_options, index=0)
    
    # Temperature slider
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.3, 0.1)
    
    # Number of retrieved chunks
    k_chunks = st.slider("Retrieved Chunks", 1, 10, 3)
    
    st.markdown("---")
    
    # Clear all chats button
    if st.button("🗑️ Delete All Conversations", use_container_width=True):
        clear_all_conversations()
        st.rerun()
    
    # API Status
    st.markdown("---")
    if GOOGLE_API_KEY:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Key Missing")

# ========== MAIN CHAT AREA ==========
# Title Section
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown('<div class="main-title">NexusAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Document Assistant</div>', unsafe_allow_html=True)
with col2:
    if st.session_state.current_conversation:
        conv_name = st.text_input("", value=st.session_state.conversations[st.session_state.current_conversation]["name"], 
                                  key="conv_name", label_visibility="collapsed")
        if conv_name:
            st.session_state.conversations[st.session_state.current_conversation]["name"] = conv_name

st.markdown("---")

# Check if documents are processed
if not st.session_state.documents_processed:
    st.info("👋 **Welcome to NexusAI!** Please upload documents in the sidebar to get started.")
    st.markdown("""
    ### How to use:
    1. 📁 **Upload** your PDF or text files in the sidebar
    2. 🔄 **Process** the documents
    3. 💬 **Start asking** questions about your content!
    
    ### Features:
    - ✅ Answers based ONLY on your documents
    - ✅ No data leaves your computer
    - ✅ Save and manage multiple conversations
    - ✅ Adjust AI creativity with temperature control
    """)
else:
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        if st.session_state.vectordb:
                            docs = st.session_state.vectordb.similarity_search(prompt, k=k_chunks)
                            context = "\n".join([doc.page_content for doc in docs])
                            
                            full_prompt = f"""You are NexusAI, a helpful AI assistant. Answer based ONLY on the provided context.
If the answer is not in the context, say "I cannot find that information in your documents."

Context:
{context}

Question: {prompt}

Answer:"""
                            
                            llm = ChatGoogleGenerativeAI(
                                model=selected_model, 
                                temperature=temperature, 
                                google_api_key=GOOGLE_API_KEY
                            )
                            response = llm.invoke(full_prompt)
                            answer = response.content
                            
                            st.markdown(answer)
                            
                            # Save to conversation
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                            # Update conversation in history
                            if st.session_state.current_conversation:
                                st.session_state.conversations[st.session_state.current_conversation]["messages"] = st.session_state.messages
                        else:
                            st.error("Please process your documents first!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Try switching to a different model in Settings or wait a moment for quota reset.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888888; font-size: 0.8rem;'>"
    "🔒 Your data stays private | Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)