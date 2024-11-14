import streamlit as st
import pandas as pd
import io

def display_eda():
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Library and Dataset Imports")
    st.markdown(
        "We import necessary libraries, download the dataset from Kaggle, and perform an initial load.")

    with st.expander("🐈 Code for Library and Dataset Imports"):
        st.code("""
                # Library Imports
                import pandas as pd
                import numpy as np
                import seaborn as sns
                import matplotlib.pyplot as plt

                from sklearn.model_selection import train_test_split
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.svm import SVC
                from sklearn.cluster import KMeans
                from sklearn.metrics import classification_report, accuracy_score
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.metrics import ConfusionMatrixDisplay

                # Kaggle Download
                import kagglehub
                import os

                # Dataset Path and Load
                path = kagglehub.dataset_download("valakhorasani/mobile-device-usage-and-user-behavior-dataset")
                file_name = 'user_behavior_dataset.csv'
                file_path = os.path.join(path, file_name)
                df_initial = pd.read_csv(file_path)
                df = df_initial.copy()
                        """, language="python")
        
    st.subheader("🐈 Dataset Analysis")
    st.markdown(
        "we conducted an initial exploration of the dataset to understand its structure, key attributes, and any potential data quality issues.")

    with st.expander("📜 Code for Dataset Analysis"):
        st.code("""
print(df.info())
print(df.describe(include='all'))
        """, language="python")

    # # Data Cleaning

    # dataCleanCode = ```# 'User Behavior Class' Removal
    #     df_cleaned = df.drop(columns=['User Behavior Class'])

    #     # Check Duplicates
    #     duplicates_count = df_cleaned.duplicated().sum()
    #     print(f"\nNumber of duplicate rows found: {duplicates_count}")
    
    #     # Remove Duplicates
    #     df_cleaned = df_cleaned.drop_duplicates()

    #     # Display the cleaned dataset
    #     print("\nCleaned Dataset Information:")
    #     df_cleaned.info()
    #     print("\nCleaned Dataset Head:")
    #     print(df_cleaned.head())```

    # st.subheader("Data Cleaning")
    # st.write("This step involves removing unnecessary columns, handling duplicates, and checking for missing data.")
    # st.code(dataCleanCode, language="python")

    # # Outlier Detection

    # outlierDetection = ```numeric_columns = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
    #                'Battery Drain (mAh/day)', 'Number of Apps Installed',
    #                'Data Usage (MB/day)', 'Age']

    # sns.set_theme(style="whitegrid")

    # plt.figure(figsize=(18, 10))
    # for i, col in enumerate(numeric_columns, 1):
    #     plt.subplot(2, 3, i)
    #     sns.boxplot(
    #         data=df,
    #         x=col,
    #         color='lightblue',
    #         flierprops={'markerfacecolor': 'r', 'marker': 'o'}
    #     )
    #     plt.title(f'Boxplot of {col}', fontsize=14)
    #     plt.xlabel('')

    # plt.tight_layout()
    # plt.show()
    
    
    # ```
    # st.subheader("Outlier Detection")
    # st.write("We check for outliers in numeric columns using box plots and the IQR method.")
    # st.image("path/to/outlier_boxplots.png", caption="Boxplots of Numeric Columns", use_column_width=True)
    # st.code(outlierDetection, language="python")

    # # Additional EDA sections such as Summary Statistics, Distribution Analysis, etc.
    # st.subheader("Distribution Analysis")
    # st.write("Description and analysis of distribution patterns for each feature.")
    # st.image("path/to/distribution_plots.png", caption="Distribution Plots", use_column_width=True)
