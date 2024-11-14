import streamlit as st

from dataset_information import display_dataset_information
from eda import display_eda
from ml_implementation import display_ml_implementation

# Streamlit App Configuration
st.set_page_config(page_title="Mobile Device Usage Analysis", layout="wide")

# --- Sidebar Navigation with Dropdowns ---
st.sidebar.title("Navigation")

# Main sections as dropdowns
section = st.sidebar.selectbox("Choose a section", ["Dataset Information", "Exploratory Data Analysis", "Machine Learning Implementation"])

# Display content based on selected section
if section == "Dataset Information":
    display_dataset_information()
elif section == "Exploratory Data Analysis":
    display_eda()
elif section == "Machine Learning Implementation":
    display_ml_implementation()
