import streamlit as st

from dataset_information import display_dataset_information
from eda import display_eda
from ml_implementation import display_ml_implementation
from conclusion import display_conclusion
from prediction import display_prediction

st.set_page_config(
    page_title="Mobile User Behavior Analysis",
    page_icon="🐈",
    layout="wide",
    initial_sidebar_state="expanded")

if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'Dataset Information'

# --- Sidebar Navigation with Dropdowns ---
st.sidebar.title("Mobile User Behavior Analysis")

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
1. GATMAITAN, Gilbert Jan
2. PALMA, Gian Carlo
3. REYES, Jedidiah
4. VILLAFRANCA, Johan Takkis
5. VIOLENTA, Erielson Emmanuel
""")