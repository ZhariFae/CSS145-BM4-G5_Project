import streamlit as st
import pandas as pd

def display_dataset_information():
    st.header("Dataset Overview")

    # Group Information Section
    st.subheader("CSS145-BM4 Group 5")
    st.markdown("""
    - **GATMAITAN, Gilbert Jan**
    - **PALMA, Gian Carlo**
    - **REYES, Jedidiah**
    - **VILLAFRANCA, Johan Takkis**
    - **VIOLENTA, Erielson Emmanuel**
    """)

    st.subheader("Dataset Information")
    st.markdown("""
    This project uses a dataset that analyzes **mobile device usage** and **user behavior**. 
    Access the dataset on Kaggle:
    [Mobile Device Usage and User Behavior Dataset](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset)
    """)

    # Dataset Description
    st.subheader("Dataset Description")
    st.markdown("""
    This dataset comprehensively analyzes trends in mobile device usage and userbehavior classification. It includes 700 user data samples encompassing metrics like data consumption, battery drain, screen-on time, and app usage duration. Each input is categorized into one of five user behavior groups, ranging from mild to excessive usage, to enable meaningful analysis and modeling.
    """)

    # Column Descriptions in a Table
    st.header("Column Descriptions")
    column_info = {
        "Column": [
            "User ID", "Device Model", "Operating System", "App Usage Time (min/day)",
            "Screen On Time (hours/day)", "Battery Drain (mAh/day)", "Number of Apps Installed",
            "Data Usage (MB/day)", "Age", "Gender", "User Behavior Class"
        ],
        "Description": [
            "A unique identifier for each user, ensuring user privacy.",
            "The specific model of the mobile device, affecting app performance and user behavior.",
            "The device's operating system (e.g., Android, iOS), influencing app availability and user experience.",
            "Average daily time spent on apps, reflecting engagement levels.",
            "Total daily screen-on time in hours, offering a broader measure of device usage.",
            "Daily battery consumption in milliampere-hours, indicating power usage.",
            "Total apps installed, hinting at user engagement variety.",
            "Daily data usage in MB, showing online activity extent.",
            "User's age, providing demographic insights affecting mobile usage.",
            "User's gender, enabling demographic-based analysis.",
            "Categorical classification of user behavior based on engagement."
        ]
    }

    column_df = pd.DataFrame(column_info)
    st.table(column_df)

    try:
        df = pd.read_csv('/workspaces/CSS145-BM4-G5_Project/user_behavior_dataset.csv')

        st.subheader("Preview of the Dataset")
        st.text("This contains the preview of the dataset used, you can also download the whole file.")
        st.dataframe(df.head(15)) 

    except FileNotFoundError:
        st.error(
            f"Dataset file not found at {'/workspaces/CSS145-BM4-G5_Project/user_behavior_dataset.csv'}. Please check the file path.")