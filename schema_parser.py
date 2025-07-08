# schema_parser.py

import pandas as pd
import json
from io import StringIO, BytesIO
import uuid # To generate unique IDs for datasets
from utils import log_message # Import custom logging utility

def parse_dataset_schema(uploaded_file) -> dict:
    """
    Parses the schema (column names, types, sample values) and metadata
    from an uploaded CSV or JSON file.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        dict: A dictionary containing dataset metadata and schema information.
              Example: {
                  "id": "unique_id_string",
                  "name": "my_data.csv",
                  "file_type": "csv",
                  "size_bytes": 1024,
                  "num_rows": 100,
                  "num_columns": 5,
                  "columns": [
                      {"name": "column_A", "type": "int64", "sample_values": [1, 2, 3]},
                      {"name": "column_B", "type": "object", "sample_values": ["text1", "text2"]},
                  ],
                  "summary_text": "A CSV file named my_data.csv with 100 rows and 5 columns. Columns include column_A (int64), column_B (object)."
              }
    """
    dataset_id = str(uuid.uuid4())
    file_name = uploaded_file.name
    file_type = file_name.split('.')[-1].lower()
    size_bytes = uploaded_file.size
    
    df = None
    data_preview = None
    num_rows = 0
    num_columns = 0
    columns_info = []
    summary_text = ""

    log_message('info', f"Parsing schema for file: {file_name} ({file_type})")

    try:
        if file_type == 'csv':
            # Read CSV as string to handle decoding
            string_data = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(string_data)
            
            num_rows = len(df)
            num_columns = len(df.columns)
            data_preview = df.head(5).to_dict('records') # Get first 5 rows as list of dicts

            for col in df.columns:
                col_type = str(df[col].dtype)
                # Get up to 5 unique sample values, convert to string to avoid serialization issues
                sample_values = df[col].dropna().unique().tolist()[:5]
                sample_values_str = [str(val) for val in sample_values]
                columns_info.append({
                    "name": col,
                    "type": col_type,
                    "sample_values": sample_values_str
                })
            
            summary_text = (
                f"A CSV file named {file_name} with {num_rows} rows and {num_columns} columns. "
                f"Columns and their types: {', '.join([f'{c["name"]} ({c["type"]})' for c in columns_info])}. "
                f"Sample data preview: {json.dumps(data_preview)}."
            )

        elif file_type == 'json':
            # Read JSON data
            json_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
            
            # Attempt to infer schema from the first few records if it's a list of objects
            if isinstance(json_data, list) and json_data:
                df = pd.DataFrame(json_data)
                num_rows = len(df)
                num_columns = len(df.columns)
                data_preview = df.head(5).to_dict('records')

                for col in df.columns:
                    col_type = str(df[col].dtype)
                    sample_values = df[col].dropna().unique().tolist()[:5]
                    sample_values_str = [str(val) for val in sample_values]
                    columns_info.append({
                        "name": col,
                        "type": col_type,
                        "sample_values": sample_values_str
                    })
                summary_text = (
                    f"A JSON file named {file_name} with {num_rows} records and {num_columns} inferred columns. "
                    f"Columns and their types: {', '.join([f'{c["name"]} ({c["type"]})' for c in columns_info])}. "
                    f"Sample data preview: {json.dumps(data_preview)}."
                )
            elif isinstance(json_data, dict):
                # If it's a single JSON object, treat its top-level keys as columns
                num_rows = 1 # Treat as a single record
                num_columns = len(json_data)
                data_preview = [json_data] # The object itself is the preview
                
                for key, value in json_data.items():
                    col_type = type(value).__name__
                    sample_values_str = [str(value)] if value is not None else []
                    columns_info.append({
                        "name": key,
                        "type": col_type,
                        "sample_values": sample_values_str
                    })
                summary_text = (
                    f"A JSON file named {file_name} with 1 record and {num_columns} top-level fields. "
                    f"Fields and their types: {', '.join([f'{c["name"]} ({c["type"]})' for c in columns_info])}. "
                    f"Sample data preview: {json.dumps(data_preview)}."
                )
            else:
                log_message('warning', f"Unsupported JSON structure for schema parsing: {file_name}. Not a list of objects or a single object.")
                summary_text = f"A JSON file named {file_name} with an unsupported structure for detailed schema parsing."

        else:
            log_message('warning', f"Unsupported file type for schema parsing: {file_type}")
            summary_text = f"Unsupported file type: {file_type} for file {file_name}."

    except Exception as e:
        log_message('error', f"Error parsing schema for {file_name}: {e}", exc_info=True)
        summary_text = f"Error parsing file {file_name}: {e}"
        columns_info = []
        data_preview = []
        num_rows = 0
        num_columns = 0

    return {
        "id": dataset_id,
        "name": file_name,
        "file_type": file_type,
        "size_bytes": size_bytes,
        "num_rows": num_rows,
        "num_columns": num_columns,
        "columns": columns_info,
        "data_preview": data_preview, # Store actual data preview for display
        "summary_text": summary_text # Textual summary for embedding
    }