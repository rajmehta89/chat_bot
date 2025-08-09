import os
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
import cohere
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    st.error("COHERE_API_KEY environment variable not found.")
    st.stop()

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

st.set_page_config(
    page_title="PDF Q&A Chatbot with Memory",
    layout="wide",
    initial_sidebar_state="expanded" # This ensures the sidebar is expanded on every script rerun
)

# ------------- Helper classes and functions -------------
class CohereEmbeddings:
    """Wrap Cohere embedding calls for LangChain compatibility."""
    def embed_documents(self, texts):
        try:
            response = co.embed(texts=texts, model="embed-english-v2.0")
            return response.embeddings
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return [[] for _ in texts]

    def embed_query(self, text):
        res = co.embed(texts=[text], model="embed-english-v2.0")
        return res.embeddings[0]

def get_base_user_dir(email: str) -> Path:
    """Return base dir path for storing user data, safe folder name."""
    safe_email = email.replace("@", "_at_").replace(".", "_dot_")
    base_dir = Path("user_data") / safe_email
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def get_chats_metadata_path(user_dir: Path) -> Path:
    return user_dir / "chats.json"

def load_chats_metadata(user_dir: Path):
    """Load chats metadata JSON or return empty list if none."""
    meta_path = get_chats_metadata_path(user_dir)
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return []

def save_chats_metadata(user_dir: Path, chats: list):
    """Save chats metadata list as JSON."""
    meta_path = get_chats_metadata_path(user_dir)
    meta_path.write_text(json.dumps(chats, indent=2), encoding="utf-8")

def add_new_chat(user_dir: Path, chats: list):
    """Add new chat with unique id, default name, empty PDF info."""
    new_chat_id = str(uuid.uuid4())
    new_chat = {
        "chat_id": new_chat_id,
        "chat_name": f"Chat {len(chats) + 1}",
        "file_name": "",
        "vectorstore_path": "",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    chats.insert(0, new_chat)  # Insert newest at front
    # Limit to max 10 chats
    if len(chats) > 10:
        # Remove oldest chat folder
        oldest = chats.pop(-1)
        if oldest["vectorstore_path"]:
            shutil.rmtree(oldest["vectorstore_path"], ignore_errors=True)
        # Also remove chat history
        chat_history_path = get_chat_history_path(user_dir, oldest["chat_id"])
        if chat_history_path.exists():
            chat_history_path.unlink()
    save_chats_metadata(user_dir, chats)
    return chats, new_chat_id

def update_chat_name(user_dir: Path, chats: list, chat_id: str, new_name: str):
    for c in chats:
        if c["chat_id"] == chat_id:
            c["chat_name"] = new_name.strip()
            c["last_updated"] = datetime.now().isoformat()
            save_chats_metadata(user_dir, chats)
            return

def update_chat_file_info(user_dir: Path, chats: list, chat_id: str, file_name: str, vectorstore_path: str):
    for c in chats:
        if c["chat_id"] == chat_id:
            c["file_name"] = file_name
            c["vectorstore_path"] = vectorstore_path
            c["last_updated"] = datetime.now().isoformat()
            save_chats_metadata(user_dir, chats)
            return

def get_chat_dir(user_dir: Path, chat_id: str) -> Path:
    """Return folder path for a specific chat."""
    chat_dir = user_dir / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir

def get_chat_history_path(user_dir: Path, chat_id: str) -> Path:
    """Return path for chat history JSON file."""
    return user_dir / f"{chat_id}_history.json"

def load_chat_history(user_dir: Path, chat_id: str) -> List[Dict]:
    """Load chat history for a specific chat."""
    history_path = get_chat_history_path(user_dir, chat_id)
    if history_path.exists():
        try:
            return json.loads(history_path.read_text(encoding="utf-8"))
        except:
            return []
    return []

def save_chat_history(user_dir: Path, chat_id: str, history: List[Dict]):
    """Save chat history for a specific chat."""
    history_path = get_chat_history_path(user_dir, chat_id)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

def add_message_to_history(user_dir: Path, chat_id: str, role: str, content: str):
    """Add a message to chat history."""
    history = load_chat_history(user_dir, chat_id)
    message = {
        "role": role,  # "user" or "assistant"
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    history.append(message)
    # Keep only last 20 messages to avoid too much context
    if len(history) > 20:
        history = history[-20:]
    save_chat_history(user_dir, chat_id, history)
    return history

def cleanup_chat_dir(chat_dir: Path):
    """Delete all files and folders inside chat_dir with Windows compatibility."""
    if chat_dir.exists() and chat_dir.is_dir():
        try:
            # Force remove entire directory and recreate it (Windows-safe)
            shutil.rmtree(chat_dir, ignore_errors=True)
            # Small delay to ensure file handles are released
            import time
            time.sleep(0.1)
            chat_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If still fails, just ignore and continue
            pass

def load_pdf_chunks(pdf_path: str):
    """Load PDF and split into text chunks."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def create_vectorstore(chunks, persist_dir: str):
    embeddings = CohereEmbeddings()
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

def load_vectorstore(persist_dir: str):
    embeddings = CohereEmbeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def generate_chat_name_from_pdf(chunks):
    """Generate a descriptive chat name based on PDF content."""
    if not chunks:
        return "New Chat"
    # Get first few chunks to analyze content
    sample_text = " ".join([chunk.page_content[:200] for chunk in chunks[:3]])
    prompt = f"""Based on this PDF content, suggest a short, descriptive name (max 4 words) for this document:Content: {sample_text[:500]}Instructions:- If it's a resume, use: "Resume - [Name]" - If it's a technical document, use the main topic- If it's Lorem Ipsum or placeholder text, use: "Sample Document"- Keep it concise and professional- Only return the name, nothing elseName:"""
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=50,
            temperature=0.3
        )
        name = response.text.strip().replace('"', '').replace("'", "")
        # Limit to reasonable length
        if len(name) > 30:
            name = name[:30] + "..."
        return name if name else "New Chat"
    except Exception:
        return "New Chat"

def format_chat_history_for_context(history: List[Dict]) -> str:
    """Format chat history into a readable context string."""
    if not history:
        return ""
    # Only use last 10 messages to keep context manageable
    recent_history = history[-10:] if len(history) > 10 else history
    context_parts = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")
    return "\n".join(context_parts)

def answer_query_with_context(question: str, relevant_docs, chat_history: List[Dict]):
    """Generate an answer based on question, relevant PDF documents, and chat history."""
    if not relevant_docs:
        # Even without PDF context, we can still have a conversation using chat history
        if chat_history:
            history_context = format_chat_history_for_context(chat_history)
            prompt = f"""You are a helpful AI assistant. Continue the conversation based on our chat history.Previous conversation:{history_context}Current question: {question}Please provide a helpful response. If the question is about a PDF and no PDF content is available, let the user know they need to upload a PDF first."""
        else:
            return "Hello! I'm your AI assistant. I can help you with questions about PDFs you upload, or just have a general conversation. What would you like to talk about?"
    else:
        pdf_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        history_context = format_chat_history_for_context(chat_history)
        prompt = f"""You are a helpful PDF assistant. Answer the question based on the PDF content and our conversation history.PDF Content:{pdf_context}Previous conversation:{history_context}Current question: {question}Instructions:- Use both the PDF content and conversation history to provide a comprehensive answer- If the question refers to something we discussed before, acknowledge that context- If the PDF content is relevant, prioritize it in your answer- Be conversational and remember what we've talked about- If the PDF contains Lorem Ipsum or placeholder text, mention that and suggest uploading a real documentAnswer:"""
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=800,
            temperature=0.3
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# ---------------- Streamlit UI ------------------
def apply_custom_css():
    """Apply custom CSS for better UI."""
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Ensure header is visible to show the sidebar toggle button */
    /* header {visibility: hidden;} */ 
        
    /* Chat message styling */
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 70%;
        word-wrap: break-word;
        float: right;
        clear: both;
    }
        
    .assistant-message {
        background-color: #f1f3f4;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 70%;
        word-wrap: break-word;
        float: left;
        clear: both;
    }
        
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: 4px;
    }
        
    /* Chat container */
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
    }
        
    /* Sidebar styling */
    .sidebar-chat-item {
        margin-bottom: 5px;
    }
        
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
        
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def login_ui():
    """Enhanced login UI."""
    apply_custom_css()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #007bff;'>🤖 PDF Q&A Chatbot</h1>
            <p style='color: #666; font-size: 1.1rem;'>Your intelligent PDF assistant with memory</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🔐 Login to Continue")
        email = st.text_input("📧 Email Address", placeholder="Enter your email")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if email and "@" in email:
                    st.session_state["user_email"] = email.strip().lower()
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid email address.")
        # Features info
        st.markdown("---")
        st.markdown("""
        ### ✨ Features:
        - 🧠 **Smart Memory** - Remembers your conversations
        - 📄 **PDF Analysis** - Upload and ask questions about PDFs
        - 💬 **Multiple Chats** - Organize conversations by topics
        - 🔒 **Personal Space** - Your data stays private
        """)

def display_chat_history(history: List[Dict]):
    """Display chat history with improved styling."""
    if not history:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; color: #666;'>
            <h3>💬 Start a New Conversation</h3>
            <p>Ask me anything about your PDFs or just have a chat!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    # Display messages
    for message in history:
        timestamp = datetime.fromisoformat(message["timestamp"]).strftime("%H:%M")
        if message["role"] == "user":
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='background: linear-gradient(135deg, #007bff, #0056b3); color: white;
                            padding: 12px 16px; border-radius: 18px; max-width: 70%;
                            box-shadow: 0 2px 8px rgba(0,123,255,0.3);'>
                    <div><strong>You</strong></div>
                    <div style='margin-top: 4px;'>{message["content"]}</div>
                    <div style='font-size: 0.75em; opacity: 0.8; margin-top: 6px;'>{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <div style='background: #f8f9fa; border: 1px solid #e9ecef; color: #333;
                            padding: 12px 16px; border-radius: 18px; max-width: 70%;
                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div><strong>🤖 Assistant</strong></div>
                    <div style='margin-top: 4px; line-height: 1.5;'>{message["content"]}</div>
                    <div style='font-size: 0.75em; color: #666; margin-top: 6px;'>{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    apply_custom_css()
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    # Clear input field after message is sent
    if "clear_input" not in st.session_state:
        st.session_state["clear_input"] = False
    if st.session_state.get("clear_input", False):
        if "user_input" in st.session_state:
            del st.session_state["user_input"]
        st.session_state["clear_input"] = False

    if not st.session_state["logged_in"]:
        login_ui()
        return

    if "user_email" not in st.session_state:
        st.session_state["logged_in"] = False
        st.rerun()
        return

    user_email = st.session_state["user_email"]
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white; border-radius: 10px; margin-bottom: 1rem;'>
            <h3>👋 Welcome</h3>
            <p style='margin: 0; opacity: 0.9;'>{user_email}</p>
        </div>
        """, unsafe_allow_html=True)

        # New Chat button
        if st.button("➕ Create New Chat", type="primary", use_container_width=True):
            chats, new_chat_id = add_new_chat(user_dir, chats)
            st.session_state["selected_chat_id"] = new_chat_id
            st.rerun()

        # Select existing chat
        if "selected_chat_id" not in st.session_state or st.session_state["selected_chat_id"] not in [c["chat_id"] for c in chats]:
            if chats:
                st.session_state["selected_chat_id"] = chats[0]["chat_id"]
            else:
                # Create first chat automatically
                chats, new_chat_id = add_new_chat(user_dir, chats)
                st.session_state["selected_chat_id"] = new_chat_id

        selected_chat_id = st.session_state["selected_chat_id"]
        selected_chat = next(c for c in chats if c["chat_id"] == selected_chat_id)

        # Chat list
        if chats:
            st.markdown("### 📚 Your Chats")
            for i, chat in enumerate(chats):
                chat_label = chat["chat_name"]
                if len(chat_label) > 20:
                    chat_label = chat_label[:20] + "..."
                file_indicator = "📄" if chat["file_name"] else "💬"
                is_selected = chat["chat_id"] == selected_chat_id
                button_type = "primary" if is_selected else "secondary"
                if st.button(
                        f"{file_indicator} {chat_label}",
                        key=f"chat_btn_{i}",
                        disabled=is_selected,
                        use_container_width=True,
                        type=button_type if not is_selected else "secondary"
                ):
                    st.session_state["selected_chat_id"] = chat["chat_id"]
                    st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Only clear login-related session state, keep data intact
            st.session_state["logged_in"] = False
            if "user_email" in st.session_state:
                del st.session_state["user_email"]
            if "selected_chat_id" in st.session_state:
                del st.session_state["selected_chat_id"]
            st.rerun()

    # Main interface
    st.title(f"💬 {selected_chat['chat_name']}")

    # Chat rename in the same line
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("✏️ Rename", key="rename_btn"):
            st.session_state["show_rename"] = not st.session_state.get("show_rename", False)

    if st.session_state.get("show_rename", False):
        new_name = st.text_input("New chat name:", value=selected_chat["chat_name"], key="new_name_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Save", key="save_name"):
                if new_name.strip() and new_name.strip() != selected_chat["chat_name"]:
                    update_chat_name(user_dir, chats, selected_chat_id, new_name.strip())
                    st.session_state["show_rename"] = False
                    st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_name"):
                st.session_state["show_rename"] = False
                st.rerun()

    # PDF Management (Collapsible)
    with st.expander("📎 PDF Management", expanded=not selected_chat["file_name"]):
        if selected_chat["file_name"]:
            st.success(f"📄 **Current PDF:** {selected_chat['file_name']}")
            replace_pdf = st.checkbox("🔄 Replace with new PDF", value=False)
        else:
            st.info("📄 No PDF uploaded yet. Upload one to start asking questions!")
            replace_pdf = True

        if replace_pdf or not selected_chat["file_name"]:
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
            if uploaded_file:
                chat_dir = get_chat_dir(user_dir, selected_chat_id)
                pdf_path = chat_dir / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("🔄 Processing your PDF... This may take a moment."):
                    chunks = load_pdf_chunks(str(pdf_path))
                    if chunks:
                        # Cleanup if replacing
                        if selected_chat["vectorstore_path"] and Path(selected_chat["vectorstore_path"]).exists():
                            cleanup_chat_dir(chat_dir)
                        chat_dir.mkdir(parents=True, exist_ok=True)
                        # Generate smart name
                        smart_name = generate_chat_name_from_pdf(chunks)
                        update_chat_name(user_dir, chats, selected_chat_id, smart_name)
                        # Create vectorstore
                        vectorstore = create_vectorstore(chunks, persist_dir=str(chat_dir))
                        update_chat_file_info(user_dir, chats, selected_chat_id, uploaded_file.name, str(chat_dir))
                        st.success(f"✅ **PDF processed successfully!** Chat renamed to: **{smart_name}**")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Failed to process the PDF. Please try again.")

    # Load vectorstore if available
    vectorstore = None
    if selected_chat["vectorstore_path"] and Path(selected_chat["vectorstore_path"]).exists():
        vectorstore = load_vectorstore(selected_chat["vectorstore_path"])

    # Load and display chat history
    chat_history = load_chat_history(user_dir, selected_chat_id)

    # Chat history container with fixed height
    with st.container():
        st.markdown("### 💬 Conversation")
        chat_container = st.container()
        with chat_container:
            display_chat_history(chat_history)

    # Chat input section (fixed at bottom)
    st.markdown("---")
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_button = st.columns([4, 1])
        with col_input:
            user_question = st.text_input(
                "Message",
                placeholder="Ask about your PDF or just chat...",
                label_visibility="collapsed",
                key="message_input"
            )
        with col_button:
            send_clicked = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)

    # Process user input
    if send_clicked and user_question and user_question.strip():
        # Add user message to history
        chat_history = add_message_to_history(user_dir, selected_chat_id, "user", user_question.strip())
        with st.spinner("🤖 Generating response..."):
            # Get relevant documents if vectorstore exists
            relevant_docs = []
            if vectorstore:
                try:
                    relevant_docs = vectorstore.similarity_search(user_question.strip(), k=3)
                except Exception as e:
                    st.error(f"Error searching PDF: {e}")
                    relevant_docs = []
            # Generate response with context (exclude current user message from history)
            answer = answer_query_with_context(user_question.strip(), relevant_docs, chat_history[:-1])
            # Add assistant response to history
            add_message_to_history(user_dir, selected_chat_id, "assistant", answer)
        # Refresh to show new messages
        st.rerun()

    # Quick actions (if no chat history)
    if not chat_history and selected_chat["file_name"]:
        st.markdown("### 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        quick_questions = [
            "What is this document about?",
            "Give me a summary",
            "What are the key points?"
        ]
        for i, (col, question) in enumerate(zip([col1, col2, col3], quick_questions)):
            with col:
                if st.button(f"💡 {question}", key=f"quick_{i}", use_container_width=True):
                    # Add user message
                    chat_history = add_message_to_history(user_dir, selected_chat_id, "user", question)
                    with st.spinner("🤖 Generating response..."):
                        relevant_docs = []
                        if vectorstore:
                            try:
                                relevant_docs = vectorstore.similarity_search(question, k=3)
                            except:
                                relevant_docs = []
                        answer = answer_query_with_context(question, relevant_docs, chat_history[:-1])
                        add_message_to_history(user_dir, selected_chat_id, "assistant", answer)
                    st.rerun()

if __name__ == "__main__":
    main()
