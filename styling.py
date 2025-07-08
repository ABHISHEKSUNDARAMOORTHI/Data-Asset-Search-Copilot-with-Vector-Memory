# styling.py
import streamlit as st

def inject_custom_css():
    """
    Injects custom CSS into the Streamlit app for a consistent dark theme
    and enhanced UI elements.
    """
    st.markdown("""
    <style>
        /* Overall App Container */
        .stApp {
            background-color: #0e1117; /* Dark background */
            color: #ffffff; /* White text */
            font-family: 'Inter', sans-serif; /* Use Inter font */
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-weight: 600; /* Semi-bold */
        }

        /* Sidebar Styling */
        .stSidebar {
            background-color: #1a1e24; /* Slightly lighter dark for sidebar */
            color: #ffffff;
            border-radius: 0.75rem; /* Rounded corners for sidebar */
            padding: 1rem;
            margin: 0.5rem; /* Margin around the sidebar */
        }
        .stSidebar .stButton > button {
            color: #ffffff;
            border-color: #2563eb;
            border-radius: 0.5rem;
        }

        /* Main Content Area - rounded corners */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            border-radius: 0.75rem; /* Rounded corners for main content blocks */
            background-color: #1c1f26; /* Darker grey for content areas */
            margin-bottom: 1rem; /* Space between content blocks */
        }

        /* Buttons */
        .stButton > button {
            background-color: #2563eb; /* Primary blue */
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.25rem;
            border: none;
            transition: background-color 0.2s, transform 0.1s;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        }
        .stButton > button:hover {
            background-color: #1a56cc;
            transform: translateY(-2px); /* Slight lift on hover */
        }
        .stButton > button:active {
            transform: translateY(0);
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #2a2e35; /* Slightly lighter for expander headers */
            color: #ffffff;
            border-radius: 0.5rem;
            border: 1px solid #33363e;
            padding: 0.75rem 1.25rem;
            margin-bottom: 0.5rem;
            transition: background-color 0.2s;
        }
        .streamlit-expanderHeader:hover {
            background-color: #333842;
        }
        .streamlit-expanderContent {
            background-color: #1c1f26; /* Same as main content for consistency */
            border-radius: 0.5rem;
            border: 1px solid #33363e;
            border-top: none; /* No top border to blend with header */
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* Input Widgets (text input, file uploader, selectbox, slider) */
        .stTextInput > div > div > input,
        .stFileUploader > div > button,
        .stSelectbox > div > div,
        .stSlider .st-cl { /* Slider track */
            background-color: #1c1f26;
            color: #ffffff;
            border: 1px solid #33363e;
            border-radius: 0.5rem;
        }
        .stTextInput > div > div > input:focus,
        .stFileUploader > div > button:focus,
        .stSelectbox > div > div:focus-within {
            border-color: #2563eb;
            box-shadow: 0 0 0 0.1rem #2563eb;
        }
        .stSlider .st-ce { /* Slider thumb */
            background-color: #2563eb;
            border: 2px solid #ffffff;
        }

        /* Info, Warning, Error boxes */
        .stAlert {
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Custom AI Explanation Box */
        .ai-explanation-box {
            background-color: #2a2e35; /* Slightly lighter than content blocks */
            border-left: 5px solid #9333ea; /* Purple accent */
            padding: 1rem 1.5rem;
            margin-top: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            color: #ffffff;
            line-height: 1.6;
        }
        .ai-explanation-box p {
            margin-bottom: 0.5rem;
        }
        .ai-explanation-box strong {
            color: #9333ea; /* Highlight strong text in purple */
        }

        /* Highlighted text within explanations */
        .highlight-match {
            background-color: rgba(76, 175, 80, 0.3); /* Green with transparency */
            border-radius: 0.25rem;
            padding: 0.1rem 0.3rem;
            font-weight: bold;
            color: #a7f3d0; /* Lighter green text */
        }
        .highlight-score {
            color: #2563eb; /* Blue for scores */
            font-weight: bold;
        }


        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] button {
            background-color: #1c1f26; /* Tab background */
            border-bottom: 2px solid transparent; /* default border */
            border-radius: 0.5rem 0.5rem 0 0; /* Rounded top corners */
            margin-right: 0.25rem;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            font-weight: bold;
            color: #ffffff; /* Tab text color */
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-bottom: 2px solid #2563eb; /* Active tab border */
            background-color: #2a2e35; /* Active tab background */
            color: #2563eb; /* Active tab text color */
        }
        .stTabs [data-baseweb="tab-list"] button:hover {
            background-color: #2a2e35; /* Hover effect */
        }

        /* Dataframe Styling */
        .stDataFrame {
            border-radius: 0.5rem;
            overflow: hidden; /* Ensures rounded corners apply to content */
        }
        /* Make dataframe headers and rows readable in dark mode */
        .stDataFrame table {
            color: #ffffff;
        }
        .stDataFrame th {
            background-color: #2a2e35 !important; /* Darker header background */
            color: #ffffff !important;
        }
        .stDataFrame tr:nth-child(even) {
            background-color: #1c1f26; /* Even row background */
        }
        .stDataFrame tr:nth-child(odd) {
            background-color: #1a1e24; /* Odd row background */
        }

        /* Plotly/Altair chart containers */
        .stPlotlyChart, .stAltairChart {
            border-radius: 0.5rem;
            overflow: hidden;
            background-color: #1c1f26; /* Match background for charts */
            padding: 1rem;
            margin-top: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

    </style>
    """, unsafe_allow_html=True)