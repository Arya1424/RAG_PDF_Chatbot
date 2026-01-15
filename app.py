import streamlit as st
import requests
from typing import List
import json

# API Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Document-Based RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_subject" not in st.session_state:
    st.session_state.current_subject = None

def create_subject(subject_name: str) -> dict:
    """Create a new subject via API"""
    response = requests.post(
        f"{API_URL}/subjects",
        json={"name": subject_name}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error creating subject: {response.text}")
        return None

def get_subjects() -> List[dict]:
    """Get all subjects from API"""
    try:
        response = requests.get(f"{API_URL}/subjects")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def upload_documents(subject_id: str, files) -> dict:
    """Upload documents to a subject"""
    files_data = [("files", (file.name, file, file.type)) for file in files]
    response = requests.post(
        f"{API_URL}/subjects/{subject_id}/documents",
        files=files_data
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error uploading documents: {response.text}")
        return None

def get_subject_details(subject_id: str) -> dict:
    """Get detailed subject information including documents"""
    try:
        response = requests.get(f"{API_URL}/subjects/{subject_id}/details")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def ask_question(subject_id: str, question: str) -> dict:
    """Ask a question to the chatbot"""
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "subject_id": subject_id,
            "question": question,
            "top_k": 3
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting answer: {response.text}")
        return None
    """Ask a question to the chatbot"""
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "subject_id": subject_id,
            "question": question,
            "top_k": 3
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting answer: {response.text}")
        return None

# Main UI
st.title("📚 Document-Based Subject Chatbot")
st.markdown("Upload documents under specific subjects and ask questions based on the content.")

# Sidebar
with st.sidebar:
    st.header("📂 Subject Management")
    
    # Create new subject
    st.subheader("Create New Subject")
    new_subject_name = st.text_input("Subject Name", placeholder="e.g., HR, Finance, Product")
    if st.button("Create Subject", use_container_width=True):
        if new_subject_name:
            result = create_subject(new_subject_name)
            if result:
                st.success(f"✅ Subject '{new_subject_name}' created!")
                st.rerun()
        else:
            st.warning("Please enter a subject name")
    
    st.divider()
    
    # List subjects
    st.subheader("Available Subjects")
    subjects = get_subjects()
    
    if subjects:
        subject_names = [f"{s['name']} ({s['document_count']} docs)" for s in subjects]
        selected_subject_idx = st.selectbox(
            "Select Subject",
            range(len(subjects)),
            format_func=lambda i: subject_names[i]
        )
        st.session_state.current_subject = subjects[selected_subject_idx]
    else:
        st.info("No subjects available. Create one to get started!")
        st.session_state.current_subject = None

# Main content area
if st.session_state.current_subject:
    subject = st.session_state.current_subject
    
    # Create navigation using radio buttons
    selected_tab = st.radio(
        "Navigation",
        ["💬 Chat", "📤 Upload Documents", "ℹ️ Subject Info"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Chat Section
    if selected_tab == "💬 Chat":
        st.header(f"Chat with {subject['name']}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["subject_id"] == subject["id"]:
                with st.chat_message("user"):
                    st.write(message["question"])
                with st.chat_message("assistant"):
                    st.write(message["answer"])
                    if message.get("sources"):
                        with st.expander("📄 Sources"):
                            for source in message["sources"]:
                                st.write(f"- {source}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = [
                msg for msg in st.session_state.chat_history 
                if msg["subject_id"] != subject["id"]
            ]
            st.rerun()
    
    # Upload Documents Section
    elif selected_tab == "📤 Upload Documents":
        st.header(f"Upload Documents to {subject['name']}")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "txt", "ppt", "pptx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if st.button("Upload Files", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    result = upload_documents(subject["id"], uploaded_files)
                    
                    if result:
                        st.success(f"✅ Successfully uploaded {len(result['uploaded_files'])} files!")
                        st.json(result)
                        st.rerun()
            else:
                st.warning("Please select files to upload")
    
    # Subject Info Section
    elif selected_tab == "ℹ️ Subject Info":
        st.header(f"Subject Information: {subject['name']}")
        
        # Get detailed info
        details = get_subject_details(subject["id"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Subject ID", subject["id"][:8] + "...")
        with col2:
            st.metric("Documents", subject["document_count"])
        with col3:
            st.metric("Created", subject["created_at"][:10])
        
        st.divider()
        
        # Show folder structure
        if details and "folder_path" in details:
            st.subheader("📁 Storage Location")
            st.code(details["folder_path"], language="text")
            
            st.info(f"""
            **Folder Structure:**
            - 📄 Documents: `{details['folder_path']}/documents/`
            - 🔍 Vector DB: `{details['folder_path']}/vector_db/`
            - 📊 Metadata: `{details['folder_path']}/metadata/`
            """)
        
        # Show documents on disk
        if details and "documents_on_disk" in details:
            st.subheader("📚 Uploaded Documents")
            
            if details["documents_on_disk"]:
                for doc in details["documents_on_disk"]:
                    with st.expander(f"📄 {doc['filename']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Size:** {doc['size_mb']} MB")
                        with col2:
                            st.write(f"**Modified:** {doc['modified'][:19]}")
            else:
                st.info("No documents uploaded yet")
        
        st.divider()
        
        st.subheader("🔧 Subject Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Info", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Subject", use_container_width=True, type="secondary"):
                st.warning("⚠️ This will delete all documents and data for this subject!")
                if st.button("⚠️ Confirm Delete", type="primary"):
                    try:
                        response = requests.delete(f"{API_URL}/subjects/{subject['id']}")
                        if response.status_code == 200:
                            st.success("Subject deleted successfully!")
                            st.session_state.current_subject = None
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting subject: {e}")
    
    # Chat input - ALWAYS at the bottom, outside all sections
    st.divider()
    question = st.chat_input("Ask a question about this subject...")
    
    if question:
        # Add question to history immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = ask_question(subject["id"], question)
                
                if result:
                    st.write(result["answer"])
                    
                    if result.get("sources"):
                        with st.expander("📄 Sources"):
                            for source in result["sources"]:
                                st.write(f"- {source}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "subject_id": subject["id"],
                        "question": question,
                        "answer": result["answer"],
                        "sources": result.get("sources", [])
                    })
                    st.rerun()

else:
    st.info("👈 Please create or select a subject from the sidebar to get started.")

# Footer
st.divider()
