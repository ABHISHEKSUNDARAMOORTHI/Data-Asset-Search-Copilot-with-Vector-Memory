# vector_store.py

import chromadb
from chromadb.utils import embedding_functions
import os
import json
import logging
import uuid
import time
from google.api_core.exceptions import ResourceExhausted, InternalServerError, GoogleAPIError # For API errors like 429

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ChromaDB Configuration ---
# Persist the ChromaDB client to a local directory
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "dataset_schemas"

class VectorStoreManager:
    def __init__(self, ai_logic_module):
        """
        Initializes the VectorStoreManager with a ChromaDB client.
        Args:
            ai_logic_module: The loaded ai_logic module, so we can call its embedding function.
        """
        self.client = None
        self.collection = None
        self.ai_logic = ai_logic_module # Store the ai_logic module reference

        logging.info(f"Attempting to initialize ChromaDB client at: {CHROMA_DB_PATH}")
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            logging.info("ChromaDB client initialized successfully.")
            # Immediately try to get/create the collection to ensure it's ready
            self.collection = self.get_collection()
            if self.collection:
                logging.info("ChromaDB collection also initialized/retrieved successfully during manager init.")
            else:
                logging.error("ChromaDB collection failed to initialize during manager init.")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB PersistentClient: {e}", exc_info=True)
            self.client = None # Ensure client is None on failure
            self.collection = None

    def get_collection(self):
        """
        Gets or creates the ChromaDB collection.
        """
        if not self.client:
            logging.error("ChromaDB client is not initialized. Cannot get or create collection.")
            return None
        
        if self.collection: # Return existing collection if already fetched
            return self.collection

        logging.info(f"Attempting to get or create ChromaDB collection '{COLLECTION_NAME}'.")
        try:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
            )
            logging.info(f"ChromaDB collection '{COLLECTION_NAME}' ready.")
            return self.collection
        except Exception as e:
            logging.error(f"Failed to get or create ChromaDB collection '{COLLECTION_NAME}': {e}", exc_info=True)
            self.collection = None
            return None

    def index_dataset(self, dataset_info: dict):
        """
        Indexes a single dataset's schema and metadata into the vector store.
        Generates an embedding for the dataset's summary text using Gemini.

        Args:
            dataset_info (dict): Dictionary containing dataset schema and metadata,
                                 including a 'summary_text' field.
                                 Expected keys: 'id', 'name', 'summary_text', 'columns', 'file_type', etc.
        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        collection = self.get_collection()
        if not collection:
            logging.error("ChromaDB collection is not initialized. Cannot index dataset.")
            return False

        doc_id = dataset_info.get('id')
        summary_text = dataset_info.get('summary_text')
        
        if not doc_id or not summary_text:
            logging.warning(f"Missing ID or summary_text for dataset: {dataset_info.get('name', 'Unknown')}. Skipping indexing.")
            return False

        try:
            # Generate embedding using Gemini via ai_logic module
            embedding_result = self.ai_logic.get_embedding_from_text(summary_text)
            
            if embedding_result.get('error'):
                logging.error(f"Failed to get embedding for {dataset_info['name']}: {embedding_result['error']}")
                return False
            
            embedding = embedding_result['embedding']
            if not embedding:
                logging.error(f"Received empty embedding for {dataset_info['name']}. Skipping indexing.")
                return False

            # Prepare metadata for ChromaDB
            metadata = {
                "name": dataset_info.get('name'),
                "file_type": dataset_info.get('file_type'),
                "num_rows": dataset_info.get('num_rows'),
                "num_columns": dataset_info.get('num_columns'),
                "columns_info": json.dumps(dataset_info.get('columns', [])), # Store as JSON string
                "data_preview": json.dumps(dataset_info.get('data_preview', [])), # Store as JSON string
                "summary_text": summary_text # Keep summary text in metadata too
            }

            collection.add(
                embeddings=[embedding],
                documents=[summary_text], # The document content itself
                metadatas=[metadata],
                ids=[doc_id]
            )
            logging.info(f"Indexed dataset: {dataset_info['name']} (ID: {doc_id})")
            return True
        except Exception as e:
            logging.error(f"Error indexing dataset {dataset_info.get('name', 'Unknown')}: {e}", exc_info=True)
            return False

    def search_datasets(self, query_text: str, n_results: int = 5) -> list:
        """
        Performs a semantic search for datasets matching the query.
        Generates an embedding for the query text using Gemini.

        Args:
            query_text (str): The natural language query.
            n_results (int): Number of top results to return.

        Returns:
            list: A list of dictionaries, each representing a matched dataset with its
                  metadata and a 'score' (distance).
                  Example: [{"id": "...", "name": "...", "score": 0.8, "metadata": {...}}]
        """
        collection = self.get_collection()
        if not collection:
            logging.error("ChromaDB collection is not initialized. Cannot perform search.")
            return []
        
        if not query_text.strip():
            logging.warning("Empty query text provided for search.")
            return []

        try:
            # Generate embedding for the query using Gemini via ai_logic module
            query_embedding_result = self.ai_logic.get_embedding_from_text(query_text)
            
            if query_embedding_result.get('error'):
                logging.error(f"Failed to get embedding for query '{query_text}': {query_embedding_result['error']}")
                return []
            
            query_embedding = query_embedding_result['embedding']
            if not query_embedding:
                logging.error(f"Received empty query embedding for '{query_text}'. Cannot perform search.")
                return []

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            matched_datasets = []
            if results and results['ids'] and results['ids'][0]: # Ensure results are not empty
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    document_content = results['documents'][0][i]

                    # Ensure columns_info and data_preview are parsed back from JSON strings
                    if 'columns_info' in metadata and isinstance(metadata['columns_info'], str):
                        try:
                            metadata['columns_info'] = json.loads(metadata['columns_info'])
                        except json.JSONDecodeError:
                            logging.warning(f"Failed to parse columns_info for {doc_id}")
                            metadata['columns_info'] = []
                    
                    if 'data_preview' in metadata and isinstance(metadata['data_preview'], str):
                        try:
                            metadata['data_preview'] = json.loads(metadata['data_preview'])
                        except json.JSONDecodeError:
                            logging.warning(f"Failed to parse data_preview for {doc_id}")
                            metadata['data_preview'] = []

                    matched_datasets.append({
                        "id": doc_id,
                        "name": metadata.get('name', 'N/A'),
                        "score": float(distance), # Keep as distance for now, lower is better
                        "metadata": metadata,
                        "document_content": document_content # The summary text used for embedding
                    })
            
            matched_datasets.sort(key=lambda x: x['score'])
            logging.info(f"Search for '{query_text}' returned {len(matched_datasets)} results.")
            return matched_datasets

        except Exception as e:
            logging.error(f"Error searching datasets for query '{query_text}': {e}", exc_info=True)
            return []

    def get_all_indexed_datasets(self) -> list:
        """
        Retrieves all documents currently in the collection.
        Useful for displaying the "Dataset Overview Table".
        """
        collection = self.get_collection()
        if not collection:
            logging.error("ChromaDB collection is not initialized. Cannot retrieve all indexed datasets.")
            return []
        
        try:
            # Fetch all documents (ids, documents, metadatas)
            # This might be slow for very large collections
            # Check if there are any IDs before trying to get them
            all_ids = collection.get()['ids']
            if not all_ids:
                logging.info("No documents found in ChromaDB collection.")
                return []

            all_data = collection.get(
                ids=all_ids,
                include=['documents', 'metadatas']
            )

            all_datasets = []
            for i in range(len(all_data['ids'])):
                doc_id = all_data['ids'][i]
                metadata = all_data['metadatas'][i]
                document_content = all_data['documents'][i]

                # Parse JSON strings back to objects
                if 'columns_info' in metadata and isinstance(metadata['columns_info'], str):
                    try:
                        metadata['columns_info'] = json.loads(metadata['columns_info'])
                    except json.JSONDecodeError:
                        metadata['columns_info'] = []
                
                if 'data_preview' in metadata and isinstance(metadata['data_preview'], str):
                    try:
                        metadata['data_preview'] = json.loads(metadata['data_preview'])
                    except json.JSONDecodeError:
                        metadata['data_preview'] = []

                all_datasets.append({
                    "id": doc_id,
                    "name": metadata.get('name', 'N/A'),
                    "metadata": metadata,
                    "document_content": document_content
                })
            return all_datasets
        except Exception as e:
            logging.error(f"Error retrieving all indexed datasets: {e}", exc_info=True)
            return []

    def reset_collection(self):
        """
        Deletes and recreates the collection, effectively clearing all indexed data.
        """
        if not self.client:
            logging.error("ChromaDB client is not initialized. Cannot reset collection.")
            return False
        try:
            logging.info(f"Attempting to delete ChromaDB collection '{COLLECTION_NAME}'.")
            self.client.delete_collection(name=COLLECTION_NAME)
            logging.info(f"ChromaDB collection '{COLLECTION_NAME}' deleted successfully.")
            self.collection = None # Clear the in-memory reference
            self.collection = self.get_collection() # Recreate it
            if self.collection:
                logging.info(f"ChromaDB collection '{COLLECTION_NAME}' recreated successfully.")
                return True
            else:
                logging.error(f"ChromaDB collection '{COLLECTION_NAME}' failed to recreate after deletion.")
                return False
        except Exception as e:
            logging.error(f"Error resetting ChromaDB collection '{COLLECTION_NAME}': {e}", exc_info=True)
            return False