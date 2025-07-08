# 🧭 Data Asset Search Copilot with Vector Memory

## 📚 Table of Contents

- [Project Overview](#project-overview)  
- [Key Features](#key-features)  
- [How It Works (Under the Hood)](#how-it-works-under-the-hood)  
- [Visual Insights](#visual-insights)  
- [Tech Stack](#tech-stack)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Google Gemini API Key Setup](#google-gemini-api-key-setup)  
  - [Running the Application](#running-the-application)  
- [Usage Guide](#usage-guide)  
- [Sample Files (for Testing)](#sample-files-for-testing)  
- [Project Structure](#project-structure)  
- [Future Enhancements](#future-enhancements)  
- [License](#license)  
- [Contributing](#contributing)

---

## 🌐 Project Overview

**Data Asset Search Copilot with Vector Memory** is a Streamlit-based semantic search assistant that helps data teams explore, discover, and understand the structure of multiple datasets — even without knowing exact column names or dataset locations.

It combines the power of **Google Gemini's LLM (natural language reasoning)** with **ChromaDB (vector memory)** to enable semantic search across CSV and JSON datasets.

---

## 🧩 Key Features

### 📂 Multi-File Dataset Ingestion

- Upload multiple `.csv` or `.json` files.
- Automatically parses schema: column names, types, and sample values.
- Tags datasets with metadata (filename, size, rows/columns, format).

### 🧠 Schema Vectorization & Indexing

- Uses Gemini’s `embedding-001` model to generate vector embeddings for schemas.
- Stores in ChromaDB for fast similarity-based search.
- No need to re-index unless new files are uploaded.

### 💬 Semantic Query Search

- Natural language queries like:
  - “Which datasets contain `user_id` and `revenue`?”
  - “Datasets on climate or emissions”
- Gemini rephrases queries and highlights matches.

### 🤖 Gemini Copilot Integration

- **Query Reformulation**: Better semantic match.
- **Explanation Mode**: Explains why datasets matched.
- **Quota-Aware**: AI features are optional and toggleable.

### 📊 Expert-Level Visual UX

- Dynamic match score bar charts.
- Schema word clouds.
- Query history timeline.
- AI explanation boxes with term highlighting.

### 🧾 Session Report Generation

- Export session results to:
  - JSON (machine-readable)
  - Markdown (for audit teams)
  - CSV (annotated KPIs and schema)

---

## 🛠 How It Works (Under the Hood)

### 🔍 Schema Parsing (`schema_parser.py`)
- Parses columns, types, and example values from each file.

### 📦 Vector Store (`vector_store.py`)
- Prepares summaries and embeds them using Gemini.
- Stores them with metadata in ChromaDB.

### 🧠 AI Query Pipeline (`ai_logic.py`)
- Rephrases vague queries (optional).
- Embeds queries and retrieves top matches via vector similarity.
- Explains why each dataset was selected.

---

## 📈 Visual Insights

| Feature                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| Match Score Bar Chart      | Visualize confidence of dataset match using color-coded bars.              |
| AI Reasoning Box           | Explanation of match relevance using LLM output.                           |
| Schema Word Cloud          | Common column terms across all datasets.                                   |
| Dataset Overview Table     | View filename, schema, type, sample rows.                                  |
| Search History Timeline    | Track previous queries and matches.                                        |

---

## ⚙️ Tech Stack

- **IDE**: VS Code  
- **UI**: Streamlit  
- **AI**: Google Gemini (`gemini-1.5-flash-latest`, `embedding-001`)  
- **Vector DB**: ChromaDB  
- **Charts**: Altair, Matplotlib (WordCloud)  
- **Helpers**: Pandas, JSON, python-dotenv  

---

## 🚀 Getting Started

### 🔧 Prerequisites
- Python 3.9+
- pip
- Git (optional)

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/vector-search-copilot.git
cd vector-search-copilot
python -m venv venv
````

### Activate Environment

```bash
# Windows (PowerShell)
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Install Dependencies

**requirements.txt**

```txt
streamlit==1.36.0
pandas==2.2.2
google-generativeai==0.6.0
python-dotenv==1.0.1
Pillow==10.4.0
numpy==1.26.4
chromadb==0.5.0
scikit-learn==1.5.0
altair==5.3.0
matplotlib==3.9.0
wordcloud==1.9.3
```

```bash
pip install -r requirements.txt
```

---

## 🔑 Google Gemini API Key Setup

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
2. Create a `.env` file:

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

---

## 🖥 Running the Application

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

> ℹ️ **Quota Warning**: Free tier may limit queries. Disable AI options in the UI to reduce token use.

---

## 📘 Usage Guide

### 📂 Tab 1: Ingest Data

* Upload CSV/JSON files.
* Automatically indexed into ChromaDB.

### 🔍 Tab 2: Search & Explore

* Enter semantic queries (e.g. “customer feedback and ratings”).
* Optional AI: Rephrasing and explanations.
* View:

  * Matched Datasets
  * Schema Highlights
  * AI Reasoning
  * Match Score Bar Chart

### 📤 Tab 3: Overview & Export

* Browse all indexed datasets.
* View search history timeline.
* Export session reports in:

  * JSON
  * Markdown
  * CSV

---

## 🧪 Sample Files (Testing)

### `sample_data/transactions.csv`

```csv
transaction_id,user_id,product_name,amount,currency,transaction_date,region
1001,U001,Laptop,1200.00,USD,2024-01-15,North
...
```

### `sample_data/customer_feedback.json`

```json
[
  {
    "feedback_id": "F001",
    "customer_id": "C001",
    "rating": 5,
    "comments": "Excellent service and fast delivery.",
    "feedback_date": "2024-02-01"
  }
]
```

---

## 🗂 Project Structure

```bash
vector-search-copilot/
│
├── .env                    # Gemini API Key
├── app.py                  # Main Streamlit UI
├── ai_logic.py             # Handles Gemini logic
├── vector_store.py         # Vector DB storage & retrieval
├── schema_parser.py        # Schema parsing from files
├── utils.py                # Utility helpers
├── styling.py              # CSS & Streamlit theme config
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 📈 Future Enhancements

* ✅ Chunk-level vectorization for large JSON files
* ✅ Feedback mechanism for AI explanations
* 🧠 Schema summarization via LLM
* 📤 Export to Data Catalogs (Amundsen/DataHub)
* 🌍 Add FAISS / Pinecone / Weaviate vector store options

---

## 📄 License

MIT License

---

## 🤝 Contributing

Contributions and feature requests are welcome! Please submit a pull request or open an issue.
