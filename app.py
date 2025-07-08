# app.py

import streamlit as st
import pandas as pd
import json
import os
from io import StringIO, BytesIO
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt # For wordcloud
from wordcloud import WordCloud # For wordcloud generation

# --- Project Module Imports ---
from utils import initialize_session_state, get_gemini_api_key, log_message
from styling import inject_custom_css
from schema_parser import parse_dataset_schema
from vector_store import VectorStoreManager
import ai_logic # Import as module to access models and functions

# --- App Configuration ---
st.set_page_config(page_title="Data Asset Search Copilot", page_icon="🧭", layout="wide")
inject_custom_css() # Apply custom CSS
initialize_session_state() # Initialize all session state variables

# --- Sidebar: Gemini API Configuration ---
st.sidebar.header("🔐 Gemini API Configuration")
api_key_from_env = get_gemini_api_key()

if api_key_from_env:
    st.sidebar.success("API Key loaded from .env")
    st.session_state['gemini_api_key_loaded'] = True
else:
    user_api_key_input = st.sidebar.text_input("Enter your Gemini API key", type="password", key="user_api_key_input")
    if user_api_key_input:
        os.environ["GEMINI_API_KEY"] = user_api_key_input
        st.session_state['gemini_api_key_loaded'] = True
        st.sidebar.success("API Key set for this session.")
        try:
            ai_logic._configure_gemini_models()
            log_message('info', "Gemini models re-configured with new API key.")
        except Exception as e:
            log_message('error', f"Failed to re-configure Gemini models: {e}")
            st.session_state['gemini_api_key_loaded'] = False
        st.rerun()
    else:
        st.sidebar.info("Please provide an API key to enable AI features.")
        st.session_state['gemini_api_key_loaded'] = False

# --- Initialize Vector Store Manager ---
if 'vector_store_manager' not in st.session_state or st.session_state['vector_store_manager'] is None:
    try:
        st.session_state['vector_store_manager'] = VectorStoreManager(ai_logic)
        st.session_state['vector_db_initialized'] = True
        log_message('info', "VectorStoreManager initialized.")
    except Exception as e:
        log_message('error', f"Failed to initialize VectorStoreManager: {e}")
        st.session_state['vector_db_initialized'] = False
        st.error("Failed to initialize vector database. Check logs for details.")

# --- Main App Title ---
st.title("🧭 Data Asset Search Copilot")
st.markdown("Semantic search assistant to explore and discover hidden insights across multiple datasets.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📂 Ingest Data", "🔍 Search & Explore", "📈 Overview & Export"])

# --- Tab 1: Ingest Data ---
with tab1:
    st.header("Upload Datasets for Indexing")
    st.markdown("Drag & drop your `.csv` or `.json` files here. Their schemas will be vectorized for semantic search.")
    st.warning("Note: Indexing involves AI calls for embeddings. Large numbers of files or frequent re-indexing can consume your free-tier quota quickly.")


    uploaded_files = st.file_uploader(
        "Upload multiple CSV/JSON files",
        type=["csv", "json"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if uploaded_files != st.session_state['uploaded_files']:
            st.session_state['uploaded_files'] = uploaded_files
            st.session_state['indexing_needed'] = True

        if st.session_state.get('indexing_needed', False):
            st.info(f"Processing {len(uploaded_files)} files for indexing...")
            indexed_count = 0
            st.session_state['indexed_datasets'] = {}

            if not st.session_state['vector_db_initialized']:
                st.error("Vector database is not initialized. Cannot index files.")
                st.session_state['indexing_needed'] = False
            elif not st.session_state['gemini_api_key_loaded']:
                st.warning("Gemini API key not loaded. Cannot generate embeddings for indexing.")
                st.session_state['indexing_needed'] = False
            else:
                if st.session_state['vector_store_manager'].reset_collection():
                    log_message('info', "Vector database collection reset for re-indexing.")
                else:
                    st.error("Failed to reset vector database. Please try again.")
                    st.session_state['indexing_needed'] = False
                    st.stop()

                for uploaded_file in uploaded_files:
                    with st.spinner(f"Parsing and indexing {uploaded_file.name}..."):
                        dataset_info = parse_dataset_schema(uploaded_file)
                        if dataset_info:
                            st.session_state['indexed_datasets'][dataset_info['id']] = dataset_info
                            if st.session_state['vector_store_manager'].index_dataset(dataset_info):
                                indexed_count += 1
                            else:
                                st.warning(f"Failed to index {uploaded_file.name}. Check logs for API quota errors.")
                        else:
                            st.warning(f"Failed to parse schema for {uploaded_file.name}.")
                
                st.success(f"Successfully indexed {indexed_count} of {len(uploaded_files)} datasets.")
                st.session_state['indexing_needed'] = False
                st.rerun()
    else:
        st.session_state['indexing_needed'] = False
        if st.session_state.get('indexed_datasets'):
            st.session_state['indexed_datasets'] = {}
            if st.session_state['vector_db_initialized']:
                st.session_state['vector_store_manager'].reset_collection()
                log_message('info', "No files uploaded, vector database cleared.")


# --- Tab 2: Search & Explore ---
with tab2:
    st.header("Natural Language Semantic Search")

    if not st.session_state['vector_db_initialized'] or not st.session_state['indexed_datasets']:
        st.info("Please upload and index datasets in the 'Ingest Data' tab first.")
    elif not st.session_state['gemini_api_key_loaded']:
        st.warning("Gemini API key not loaded. Please provide it in the sidebar to use search features.")
    else:
        st.session_state['search_query'] = st.text_input(
            "Type your query (e.g., 'datasets about customer transactions', 'files with user IDs and order amounts')",
            value=st.session_state['search_query'],
            key="semantic_search_input"
        )

        st.markdown("---")
        st.subheader("AI Feature Controls (to manage quota)")
        col_ai_controls_1, col_ai_controls_2 = st.columns(2)
        with col_ai_controls_1:
            # New checkbox to control AI Query Rephrasing
            use_ai_rephrase = st.checkbox("Enable AI Query Rephrasing", value=True, key="enable_ai_rephrase")
        with col_ai_controls_2:
            # New checkbox to control AI Explanation
            use_ai_explanation = st.checkbox("Enable AI Explanation of Results", value=True, key="enable_ai_explanation")
        
        st.info("Disabling AI features can help conserve your daily free-tier quota.")
        st.markdown("---")


        if st.button("🔍 Search Datasets", key="search_button"):
            if st.session_state['search_query'].strip():
                with st.spinner("Processing search... This may involve AI calls."):
                    rephrased_query = st.session_state['search_query']
                    suggested_keywords = []
                    st.session_state['ai_explanation'] = "Explanation not generated (AI feature disabled or no results)."

                    if use_ai_rephrase:
                        # Get summaries of all indexed datasets for AI context
                        all_indexed_summaries = [ds['summary_text'] for ds in st.session_state['indexed_datasets'].values()]
                        
                        # 1. Rephrase query using AI
                        rephrase_result = ai_logic.rephrase_query_for_semantic_search(
                            st.session_state['search_query'],
                            all_indexed_summaries
                        )
                        
                        if rephrase_result.get('error'):
                            st.error(f"AI Query Rephrasing Error: {rephrase_result['error']}")
                            st.session_state['search_results'] = []
                            st.session_state['ai_explanation'] = "Failed to rephrase query due to AI error."
                            log_message('error', f"Query rephrasing failed: {rephrase_result['error']}")
                            st.stop() # Stop execution on critical AI error

                        rephrased_query = rephrase_result['rephrased_query']
                        suggested_keywords = rephrase_result['suggested_keywords']
                        
                        st.info(f"AI Rephrased Query: **'{rephrased_query}'**")
                        if suggested_keywords:
                            st.info(f"AI Suggested Keywords: {', '.join(suggested_keywords)}")
                    else:
                        st.info("AI Query Rephrasing is disabled. Searching with original query.")


                    # 2. Perform semantic search (always happens)
                    search_results = st.session_state['vector_store_manager'].search_datasets(rephrased_query)
                    st.session_state['search_results'] = search_results

                    # 3. Get AI Explanation for results (optional)
                    if use_ai_explanation and search_results:
                        explanation_result = ai_logic.explain_search_results(
                            st.session_state['search_query'],
                            search_results
                        )
                        if explanation_result.get('error'):
                            st.error(f"AI Explanation Error: {explanation_result['error']}")
                            st.session_state['ai_explanation'] = "Failed to generate explanation due to AI error."
                            log_message('error', f"Explanation failed: {explanation_result['error']}")
                        else:
                            st.session_state['ai_explanation'] = explanation_result
                    elif not use_ai_explanation:
                        st.info("AI Explanation of Results is disabled.")
                    elif not search_results:
                        st.info("No search results to explain.")
                
                # Add to search history
                st.session_state['search_history'].append({
                    "query": st.session_state['search_query'],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_matches": len(st.session_state['search_results'])
                })
                st.rerun() # Rerun to display results

        # Display Search Results
        if st.session_state['search_results']:
            st.subheader("Top Matching Datasets")

            # 🎯 Top Match Score Bar Chart
            st.markdown("#### Match Score Overview")
            plot_data = []
            for res in st.session_state['search_results']:
                plot_data.append({"Dataset": res['name'], "Score": res['score']})
            df_plot = pd.DataFrame(plot_data)

            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X('Dataset:N', sort='-y', title='Dataset Name'),
                y=alt.Y('Score:Q', title='Similarity Distance (Lower is Better)'),
                color=alt.Color('Score:Q',
                                scale=alt.Scale(domain=[0, 0.7, 1.5], range=['#4CAF50', '#FFEB3B', '#F44336']), # Green, Yellow, Red
                                legend=alt.Legend(title="Match Quality (Distance)")),
                tooltip=['Dataset', alt.Tooltip('Score', format='.4f')]
            ).properties(
                title='Dataset Match Scores'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Display individual results with explanations
            st.markdown("#### Detailed Match Results")
            for result in st.session_state['search_results']:
                dataset_id = result['id']
                dataset_name = result['name']
                score = result['score']
                metadata = result['metadata']
                
                full_dataset_info = st.session_state['indexed_datasets'].get(dataset_id, {})

                with st.expander(f"📁 **{dataset_name}** (Score: {score:.4f})"):
                    # 🧠 AI Explanation Box with Highlighted Matches
                    # Check if ai_explanation is a dict before trying to .get()
                    if isinstance(st.session_state['ai_explanation'], dict):
                        explanation_text = st.session_state['ai_explanation'].get(dataset_id, "No specific AI explanation available for this dataset (AI feature disabled or no explanation generated).")
                    else: # It's a string (e.g., "Explanation not generated...")
                        explanation_text = st.session_state['ai_explanation']
                    
                    st.markdown(f"<div class='ai-explanation-box'>{explanation_text}</div>", unsafe_allow_html=True)
                    st.markdown("---")

                    st.subheader("Dataset Metadata & Schema")
                    st.write(f"**File Type:** {metadata.get('file_type', 'N/A')}")
                    st.write(f"**Number of Rows:** {metadata.get('num_rows', 'N/A')}")
                    st.write(f"**Number of Columns:** {metadata.get('num_columns', 'N/A')}")
                    st.write(f"**Size (bytes):** {metadata.get('size_bytes', 'N/A')}")

                    if full_dataset_info.get('columns'):
                        st.markdown("##### Columns:")
                        cols_df = pd.DataFrame(full_dataset_info['columns'])
                        st.dataframe(cols_df, use_container_width=True)
                    
                    if full_dataset_info.get('data_preview'):
                        st.markdown("##### Data Preview:")
                        try:
                            preview_df = pd.DataFrame(full_dataset_info['data_preview'])
                            st.dataframe(preview_df, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display data preview: {e}")
                            st.json(full_dataset_info['data_preview'])

        else:
            st.info("No search results yet. Type a query and click 'Search Datasets'.")

# --- Tab 3: Overview & Export ---
with tab3:
    st.header("Indexed Datasets Overview & Report Generation")

    # 📦 Dataset Overview Table
    st.subheader("All Indexed Datasets")
    if st.session_state['indexed_datasets']:
        overview_data = []
        for ds_id, ds_info in st.session_state['indexed_datasets'].items():
            overview_data.append({
                "ID": ds_id[:8] + "...",
                "Name": ds_info.get('name'),
                "Type": ds_info.get('file_type'),
                "Rows": ds_info.get('num_rows'),
                "Columns": ds_info.get('num_columns'),
                "Summary": ds_info.get('summary_text', 'No summary').split('.')[0] + "..."
            })
        df_overview = pd.DataFrame(overview_data)
        st.dataframe(df_overview, use_container_width=True)
    else:
        st.info("No datasets have been indexed yet. Go to 'Ingest Data' tab to upload files.")

    # 🧮 Schema Word Cloud
    st.subheader("Common Schema Keywords")
    if st.session_state['indexed_datasets']:
        all_column_names = []
        for ds_info in st.session_state['indexed_datasets'].values():
            for col in ds_info.get('columns', []):
                all_column_names.append(col.get('name', ''))
        
        if all_column_names:
            text = " ".join(all_column_names)
            if text.strip():
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Common Column Names Across Datasets')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating Schema Word Cloud: {e}")
            else:
                st.info("No column names found to generate word cloud.")
        else:
            st.info("No column names found to generate word cloud.")
    else:
        st.info("Upload and index datasets to see common schema keywords.")


    # 📥 Search History Timeline
    st.subheader("Search History")
    if st.session_state['search_history']:
        for entry in reversed(st.session_state['search_history']):
            with st.expander(f"Query: **{entry['query']}** ({entry['timestamp']})"):
                st.write(f"Matches found: **{entry['num_matches']}**")
    else:
        st.info("No search history yet. Perform a search in the 'Search & Explore' tab.")

    # --- Report Generation ---
    st.subheader("Generate Search Session Report")
    if st.session_state['search_results'] or st.session_state['search_history'] or st.session_state['indexed_datasets']:
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "search_query": st.session_state['search_query'],
            "search_results": [],
            "ai_explanations": st.session_state['ai_explanation'], # This can be a dict or string
            "search_history": st.session_state['search_history'],
            "indexed_datasets_summary": [
                {
                    "id": ds['id'],
                    "name": ds['name'],
                    "file_type": ds['file_type'],
                    "num_rows": ds['num_rows'],
                    "num_columns": ds['num_columns'],
                    "summary_text": ds['summary_text']
                } for ds in st.session_state['indexed_datasets'].values()
            ]
        }

        for result in st.session_state['search_results']:
            # Get explanation safely for report
            explanation_for_report = "Explanation not generated (AI feature disabled or error)."
            if isinstance(report_data['ai_explanations'], dict):
                explanation_for_report = report_data['ai_explanations'].get(result['id'], explanation_for_report)

            report_data['search_results'].append({
                "id": result['id'],
                "name": result['name'],
                "score": result['score'],
                "metadata": result['metadata'],
                "ai_explanation": explanation_for_report
            })

        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            "📥 Download JSON Report",
            report_json,
            file_name="data_asset_search_report.json",
            mime="application/json",
            key="download_json_report"
        )

        report_md = f"""
# Data Asset Search Session Report - {report_data['timestamp']}

## Search Query
**Original Query:** {report_data['search_query']}

## AI Explanations for Top Matches
"""
        if isinstance(report_data['ai_explanations'], dict):
            for ds_id, explanation in report_data['ai_explanations'].items():
                dataset_name = st.session_state['indexed_datasets'].get(ds_id, {}).get('name', f"Dataset ID: {ds_id}")
                report_md += f"""
### {dataset_name}
{explanation}
"""
        else:
            report_md += f"\n{report_data['ai_explanations']}\n"

        report_md += f"""
## Top Search Results
"""
        if report_data['search_results']:
            for result in report_data['search_results']:
                report_md += f"""
### {result['name']} (Score: {result['score']:.4f})
* **File Type:** {result['metadata'].get('file_type', 'N/A')}
* **Rows:** {result['metadata'].get('num_rows', 'N/A')}
* **Columns:** {result['metadata'].get('num_columns', 'N/A')}
* **Summary:** {result['metadata'].get('summary_text', 'N/A')}
* **Columns Info:**
    ```json
{json.dumps(result['metadata'].get('columns_info', []), indent=2)}
    ```
"""
        else:
            report_md += "No search results for this session.\n"

        report_md += f"""
## Indexed Datasets Overview
"""
        if report_data['indexed_datasets_summary']:
            for ds_summary in report_data['indexed_datasets_summary']:
                report_md += f"""
### {ds_summary['name']} (ID: {ds_summary['id'][:8]}...)
* **Type:** {ds_summary['file_type']}
* **Rows:** {ds_summary['num_rows']}
* **Columns:** {ds_summary['num_columns']}
* **Summary:** {ds_summary['summary_text']}
"""
        else:
            report_md += "No datasets indexed.\n"

        report_md += f"""
## Search History
"""
        if report_data['search_history']:
            for entry in report_data['search_history']:
                report_md += f"* **Query:** \"{entry['query']}\" - **Matches:** {entry['num_matches']} - **Time:** {entry['timestamp']}\n"
        else:
            report_md += "No search history recorded.\n"

        st.download_button(
            "📝 Download Markdown Report",
            report_md,
            file_name="data_asset_search_report.md",
            mime="text/markdown",
            key="download_md_report"
        )
    else:
        st.info("Perform a search or index datasets to generate a report.")

# --- Sidebar: Reset Button ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Application", key="reset_app_button"):
    st.session_state.clear()
    if 'vector_store_manager' in st.session_state and st.session_state['vector_store_manager'] is not None:
        try:
            st.session_state['vector_store_manager'].reset_collection()
            log_message('info', "ChromaDB collection cleared on app reset.")
        except Exception as e:
            log_message('error', f"Error clearing ChromaDB on reset: {e}")
    st.rerun()