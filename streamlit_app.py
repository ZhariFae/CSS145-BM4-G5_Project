import streamlit as st

from dataset_information import display_dataset_information
from eda import display_eda
from ml_implementation import display_ml_implementation
from conclusion import display_conclusion
from prediction import display_prediction

# Streamlit App Configuration
st.set_page_config(page_title="Mobile Device Usage Analysis", layout="wide")

# --- Sidebar Navigation with Dropdowns ---
st.sidebar.title("Navigation")

# Main sections as dropdowns
section = st.sidebar.selectbox("Choose a page:", [
                               "Dataset Information", "Exploratory Data Analysis", "Machine Learning Implementation", "Prediction", "Conclusion"])

# Display content based on selected section
if section == "Dataset Information":
    display_dataset_information()
elif section == "Exploratory Data Analysis":
    display_eda()
elif section == "Machine Learning Implementation":
    display_ml_implementation()
elif section == "Prediction":
    display_prediction()
elif section == "Conclusion":
    display_conclusion()

st.sidebar.title("Group Members")
st.sidebar.markdown("""
- GATMAITAN, Gilbert Jan
- PALMA, Gian Carlo
- REYES, Jedidiah
- VILLAFRANCA, Johan Takkis
- VIOLENTA, Erielson Emmanuel
""")