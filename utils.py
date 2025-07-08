# utils.py

import os
import logging
from dotenv import load_dotenv
import streamlit as st # For session_state management

# --- Logging Configuration ---
# Configure logging to show information and errors in the terminal where Streamlit is run
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Key Utility ---
def get_gemini_api_key():
    """
    Loads the Gemini API key from environment variables (first .env, then system).
    """
    load_dotenv() # Load environment variables from .env file
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.warning("GEMINI_API_KEY not found in .env or system environment variables.")
    return api_key

# --- Session State Management ---
def initialize_session_state():
    """
    Initializes Streamlit session state variables if they don't already exist.
    This prevents re-running expensive operations on every rerun.
    """
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = [] # List of uploaded file objects
    if 'indexed_datasets' not in st.session_state:
        st.session_state['indexed_datasets'] = {} # Stores parsed schema and metadata for each dataset
                                                  # Format: {file_id: {name, type, columns, rows, size, ...}}
    if 'vector_db_initialized' not in st.session_state:
        st.session_state['vector_db_initialized'] = False # Flag to check if vector DB is ready
    if 'search_query' not in st.session_state:
        st.session_state['search_query'] = "" # Current user search query
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = [] # List of matched datasets from search
    if 'ai_explanation' not in st.session_state:
        st.session_state['ai_explanation'] = "" # Gemini's explanation for search results
    if 'search_history' not in st.session_state:
        st.session_state['search_history'] = [] # Stores past queries and results for timeline
    if 'gemini_api_key_loaded' not in st.session_state:
        st.session_state['gemini_api_key_loaded'] = False # Status of API key loading
    if 'chroma_client' not in st.session_state:
        st.session_state['chroma_client'] = None # Store ChromaDB client
    if 'chroma_collection' not in st.session_state:
        st.session_state['chroma_collection'] = None # Store ChromaDB collection

def log_message(level, message, **kwargs):
    """
    Custom logging function that can also push messages to Streamlit for UI display.
    """
    if level == 'info':
        logging.info(message, **kwargs)
    elif level == 'warning':
        logging.warning(message, **kwargs)
    elif level == 'error':
        logging.error(message, **kwargs)
    elif level == 'debug':
        logging.debug(message, **kwargs)