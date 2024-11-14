import streamlit as st

def display_eda():
    st.header("Exploratory Data Analysis (EDA)")
    
    # Data Cleaning
    st.subheader("Data Cleaning")
    st.write("This step involves removing unnecessary columns, handling duplicates, and checking for missing data.")
    st.code("""
    # 'User Behavior Class' Removal
    df_cleaned = df.drop(columns=['User Behavior Class'])
    ...
    """, language="python")

    # Outlier Detection
    st.subheader("Outlier Detection")
    st.write("We check for outliers in numeric columns using box plots and the IQR method.")
    st.image("path/to/outlier_boxplots.png", caption="Boxplots of Numeric Columns", use_column_width=True)
    st.code("""
    numeric_columns = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', ...]
    sns.set_theme(style="whitegrid")
    ...
    """, language="python")

    # Additional EDA sections such as Summary Statistics, Distribution Analysis, etc.
    st.subheader("Distribution Analysis")
    st.write("Description and analysis of distribution patterns for each feature.")
    st.image("path/to/distribution_plots.png", caption="Distribution Plots", use_column_width=True)
