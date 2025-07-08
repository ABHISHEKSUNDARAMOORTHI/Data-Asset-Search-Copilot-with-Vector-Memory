# ai_logic.py

import google.generativeai as genai
import os
import json
import logging
import time # For time.sleep
import random # For random jitter
from typing import Union, List, Dict
from google.api_core.exceptions import ResourceExhausted, InternalServerError, GoogleAPIError # For API errors like 429

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Retry Configuration ---
MAX_RETRIES = 5             # Maximum number of times to retry an API call
INITIAL_RETRY_DELAY = 2     # Initial delay in seconds before first retry (can be adjusted)
RETRY_JITTER_MAX = 0.5      # Max random additional delay to prevent thundering herd

# --- Gemini Model Configuration ---
# Initialize text generation model
text_gen_model = None
# Initialize embedding model
embedding_model = None

def _configure_gemini_models():
    """Configures and initializes both text generation and embedding models."""
    global text_gen_model, embedding_model # Declare global to modify them

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file or Streamlit sidebar.")
        raise ValueError("GEMINI_API_KEY is not set. Please provide it to enable AI features.")
    
    genai.configure(api_key=api_key)
    
    try:
        # Model for text generation (e.g., rephrasing, explanations)
        # Using gemini-1.5-flash-latest as it's more free-tier friendly and fast
        logging.info("Attempting to initialize text generation model: models/gemini-1.5-flash-latest")
        text_gen_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

        # Model for embeddings
        logging.info("Attempting to initialize embedding model: models/embedding-001")
        embedding_model = genai.GenerativeModel('models/embedding-001')
        
        logging.info("Gemini models initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini models: {e}", exc_info=True)
        text_gen_model = None
        embedding_model = None
        raise # Re-raise to indicate failure to the caller


# Call configuration on module load
try:
    _configure_gemini_models()
except ValueError as e:
    logging.error(f"Initial model configuration failed due to missing API key: {e}")
except Exception as e:
    logging.error(f"Initial model configuration failed unexpectedly: {e}")


# --- Helper for parsing JSON from Gemini responses ---
def _parse_gemini_json_response(response_text: str) -> Union[dict, list]:
    """
    Attempts to parse JSON from Gemini's text response, handling common markdown formatting.
    Returns the parsed JSON object (dict or list). Raises ValueError on parsing failure.
    """
    cleaned_text = response_text.strip()
    
    # Remove markdown code block wrappers if present
    if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[7:-3].strip()
    elif cleaned_text.startswith("```") and cleaned_text.endswith("```"): # Generic code block
        cleaned_text = cleaned_text[3:-3].strip()
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}. Raw response: {cleaned_text[:500]}...")
        raise ValueError(f"Failed to parse JSON response from AI: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        raise ValueError(f"Unexpected error parsing AI response: {e}")


# --- Internal Gemini API Call Helper with Retry Logic ---
def _call_gemini_with_retry(model_instance, prompt: str, is_embedding_call: bool = False) -> str:
    """
    Internal helper to call Gemini's generate_content or embed_content with retry logic.
    Raises an exception if all retries fail.
    """
    if not model_instance:
        raise ValueError("Gemini model is not initialized. Cannot make API call.")

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            logging.info(f"API Call Attempt {retry_count + 1}/{MAX_RETRIES} for {'embedding' if is_embedding_call else 'generation'}.")
            if is_embedding_call:
                # For embedding, we call embed_content and extract the embedding vector
                response = model_instance.embed_content(model=model_instance.model_name, content=prompt)
                # The embedding is usually in response.embedding.values or similar structure
                # For `models/embedding-001`, it's directly `response.embedding`
                if hasattr(response, 'embedding') and response.embedding:
                    return response.embedding # Return the embedding list directly
                else:
                    raise ValueError("Embedding response did not contain an embedding.")
            else:
                # For text generation, we call generate_content
                response = model_instance.generate_content(prompt)
                return response.text # Return text directly on success
        except ResourceExhausted as e: # This is the 429 quota error
            retry_count += 1
            if retry_count == MAX_RETRIES:
                logging.error(f"Max retries reached for ResourceExhausted. Last error: {e}")
                raise e # Re-raise the quota error if max retries reached

            wait_time = INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, RETRY_JITTER_MAX)
            logging.warning(f"Quota exceeded (429) on attempt {retry_count}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        except (InternalServerError, GoogleAPIError) as e: # Catch other API-related errors
            retry_count += 1
            logging.error(f"API error (non-quota) on attempt {retry_count}/{MAX_RETRIES}: {e}", exc_info=True)
            if retry_count == MAX_RETRIES:
                raise e
            wait_time = INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, RETRY_JITTER_MAX)
            logging.warning(f"API error. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        except Exception as e: # Catch any other general exceptions
            logging.error(f"An unexpected error occurred during Gemini API call on attempt {retry_count + 1}: {e}", exc_info=True)
            raise e # Re-raise immediately for other error types
    
    raise Exception("API call failed after multiple retries for an unknown reason.")


# --- Public AI Functions for the App ---

def get_embedding_from_text(text: str) -> Dict[str, Union[List[float], str]]:
    """
    Generates an embedding vector for the given text using Gemini's embedding model.
    """
    if not embedding_model:
        return {"embedding": [], "error": "Embedding model not initialized. Check API key."}
    if not text or not text.strip():
        return {"embedding": [], "error": "No text provided for embedding."}

    try:
        embedding = _call_gemini_with_retry(embedding_model, text, is_embedding_call=True)
        return {"embedding": embedding, "error": None}
    except Exception as e:
        logging.error(f"Error generating embedding for text: '{text[:50]}...': {e}", exc_info=True)
        return {"embedding": [], "error": f"Failed to generate embedding: {e}"}

def rephrase_query_for_semantic_search(user_query: str, indexed_dataset_summaries: List[str]) -> Dict[str, Union[str, List[str]]]:
    """
    Uses Gemini to rephrase a natural language user query into terms more suitable
    for semantic search across dataset schemas, and suggests relevant column names.

    Args:
        user_query (str): The original natural language query from the user.
        indexed_dataset_summaries (List[str]): A list of summary texts for all indexed datasets.
                                                Used to provide context on available data types.

    Returns:
        Dict[str, Union[str, List[str]]]: A dictionary containing:
            'rephrased_query': A more explicit query string for vector search.
            'suggested_keywords': A list of relevant keywords or column names.
            'error': An error message if the operation fails.
    """
    if not text_gen_model:
        return {"rephrased_query": user_query, "suggested_keywords": [], "error": "AI model not initialized."}
    if not user_query.strip():
        return {"rephrased_query": "", "suggested_keywords": [], "error": "Empty query provided."}

    # Limit the number of dataset summaries sent to the LLM to save tokens
    # For very large numbers of datasets, consider sampling or summarizing summaries.
    context_summaries = "\n".join(indexed_dataset_summaries[:10]) # Send up to 10 summaries for context

    prompt = f"""
    The user is searching for datasets. Their query is: "{user_query}"
    
    Based on this query and considering the following examples of dataset summaries that might be available:
    ---
    {context_summaries if context_summaries else "No specific dataset summaries provided for context."}
    ---

    Please rephrase the user's query into a concise, clear statement optimized for semantic search across dataset schemas and descriptions.
    Additionally, suggest 3-5 relevant keywords or potential column names that might appear in such datasets, even if not explicitly mentioned in the query.
    Return the output as a JSON object with two keys:
    "rephrased_query": (string) A rephrased query.
    "suggested_keywords": (array of strings) A list of relevant keywords/column names.
    """
    
    try:
        response_text = _call_gemini_with_retry(text_gen_model, prompt)
        parsed_response = _parse_gemini_json_response(response_text)
        
        if not isinstance(parsed_response, dict) or "rephrased_query" not in parsed_response or "suggested_keywords" not in parsed_response:
            raise ValueError("AI returned an unexpected JSON structure for query rephrasing.")
            
        return {
            "rephrased_query": parsed_response.get("rephrased_query", user_query),
            "suggested_keywords": parsed_response.get("suggested_keywords", []),
            "error": None
        }
    except Exception as e:
        logging.error(f"Error rephrasing query '{user_query}': {e}", exc_info=True)
        return {"rephrased_query": user_query, "suggested_keywords": [], "error": f"Failed to rephrase query: {e}"}


def explain_search_results(user_query: str, search_results: List[Dict]) -> Dict[str, str]:
    """
    Uses Gemini to provide a natural language explanation for why certain datasets
    were matched to the user's query, highlighting key matching columns/terms.

    Args:
        user_query (str): The original natural language query.
        search_results (List[Dict]): A list of dictionaries, each representing a matched dataset.
                                     Expected to contain 'name', 'score', 'metadata' (with 'columns_info', 'summary_text').

    Returns:
        Dict[str, str]: A dictionary where keys are dataset IDs and values are AI explanations.
                        Includes an 'error' key if the operation fails.
    """
    if not text_gen_model:
        return {"error": "AI model not initialized. Cannot generate explanations."}
    if not search_results:
        return {"explanation": "No search results to explain.", "error": None}

    # Prepare a concise summary of search results for the LLM
    results_summary = []
    for result in search_results[:3]: # Limit to top 3 results for explanation to save tokens
        dataset_name = result.get('name', 'Unknown Dataset')
        score = result.get('score', 'N/A')
        summary_text = result.get('metadata', {}).get('summary_text', 'No summary available.')
        columns_info = result.get('metadata', {}).get('columns_info', [])
        column_names = ", ".join([col.get('name') for col in columns_info if col.get('name')])
        
        results_summary.append(f"Dataset Name: {dataset_name}\n"
                               f"Similarity Score (lower is better): {score:.4f}\n"
                               f"Summary: {summary_text}\n"
                               f"Key Columns: {column_names}\n"
                               "---")

    prompt = f"""
    The user searched for datasets with the query: "{user_query}"

    Here are the top search results:
    ```
    {'\n'.join(results_summary)}
    ```

    For each dataset in the search results, explain concisely why it was matched to the user's query.
    Highlight the specific column names or terms from the dataset's summary/schema that are most relevant to the query.
    Use HTML <span> tags with class="highlight-match" for highlighting matched terms/columns.
    Also, mention the similarity score and whether it indicates a strong, moderate, or weak match (lower score is better for distance metrics).

    Return the output as a JSON object where keys are dataset IDs and values are the explanation strings.
    Example:
    {{
        "dataset_id_1": "This dataset was a strong match (score X.XX) because it contains <span class='highlight-match'>user_id</span> and <span class='highlight-match'>transaction_amount</span> columns, directly addressing the query.",
        "dataset_id_2": "This dataset was a moderate match (score Y.YY) due to its <span class='highlight-match'>customer_feedback</span> and <span class='highlight-match'>satisfaction_score</span> fields, which are related to customer satisfaction metrics."
    }}
    """
    
    try:
        response_text = _call_gemini_with_retry(text_gen_model, prompt)
        parsed_response = _parse_gemini_json_response(response_text)
        
        if not isinstance(parsed_response, dict):
            raise ValueError("AI returned an unexpected JSON structure for explanations (not a dictionary).")
            
        return parsed_response
    except Exception as e:
        logging.error(f"Error explaining search results for query '{user_query}': {e}", exc_info=True)
        return {"error": f"Failed to generate explanations: {e}"}