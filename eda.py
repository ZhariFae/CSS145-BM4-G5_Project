import streamlit as st
import pandas as pd
import io

from PIL import Image

def display_eda():
    st.header("Exploratory Data Analysis (EDA)")

    # ------------------------------ LIBRARY AND DATASET IMPORTS ------------------------------

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

                # Kaggle Dataset Import
                path = kagglehub.dataset_download("valakhorasani/mobile-device-usage-and-user-behavior-dataset")
                print("Path to dataset files:", path)
                """, language="python")
        
        st.markdown("""Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.4)
                    Path to dataset files: /root/.cache/kagglehub/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset/versions/1""")
        
        st.code("""
                # Files in Downloaded Dataset Directory
                files = os.listdir(path)
                print("Files in the downloaded dataset directory:", files)
                """, language="python")
        
        st.markdown(
            """Files in the downloaded dataset directory: ['user_behavior_dataset.csv']""")
        
        st.code("""
                # Load the dataset
                file_name = 'user_behavior_dataset.csv'
                file_path = os.path.join(path, file_name)

                df_initial = pd.read_csv(file_path)
                """, language="python")
        
    # ------------------------------ DATASET ANALYSIS ------------------------------

    st.subheader("Dataset Analysis")
    st.markdown(
        "We conducted an initial exploration of the dataset to understand its structure, key attributes, and any potential data quality issues.")

    with st.expander("🐈 Code for Dataset Analysis"):
        st.code("""
                # Dataset Read
                df_initial.head()
                
                # Original Copy
                df = df_initial.copy()

                # Dataset Inspect
                print("Dataset Information:")
                print(df.info()) """, language="python")
        
        file_path = "/workspaces/CSS145-BM4-G5_Project/assets/user_behavior_dataset.csv"
        df = pd.read_csv(file_path)
        st.write(df.head())

        st.code("""
                print("Summary Statistics:")
                print(df.describe(include='all'))
                """, language="python")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text("Dataset Information:")
        st.text(info_str)

    # ------------------------------ DATASET CLEANING ------------------------------

    st.subheader("Dataset Cleaning")
    st.markdown(
        "In the dataset cleaning process, we removed the `User Behavior Class` feature due to its vague definition and lack of clarity, as it was not adequately explained in the dataset documentation. Excluding this column helps maintain a clear and interpretable dataset for analysis, focusing on well-defined, measurable features.")

    with st.expander("🐈 Code for Dataset Cleaning"):
        st.code("""
                # Dataset Read
                df_initial.head()
                
                # Original Copy
                df = df_initial.copy()

                # Dataset Inspect
                print("Dataset Information:")
                print(df.info()) """, language="python")

        df = pd.read_csv(
            "/workspaces/CSS145-BM4-G5_Project/assets/user_behavior_dataset.csv")

        # Remove 'User Behavior Class' column
        df_cleaned = df.drop(columns=['User Behavior Class'])

        # Check and remove duplicates
        duplicates_count = df_cleaned.duplicated().sum()
        df_cleaned = df_cleaned.drop_duplicates()

        # Show duplicate count
        st.write(f"**Number of duplicate rows found and removed:** {duplicates_count}")

        buffer = io.StringIO()
        df_cleaned.info(buf=buffer)
        info_str = buffer.getvalue()

        # Display cleaned dataset information
        st.text("Cleaned Dataset Information:")
        st.text(info_str)

        # Display cleaned dataset head
        st.write("**Cleaned Dataset Preview:**")
        st.dataframe(df_cleaned.head())

    # ------------------------------ OUTLIER DETECTION ------------------------------

    st.subheader("Outlier Detection")
    st.markdown(
        "We used statistical methods and visualizations (such as box plots and scatter plots) to detect potential outliers in our numerical features.")

    with st.expander("🐈 Code for Outlier Detection: Boxplots"):
        st.code("""
                numeric_columns = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                   'Battery Drain (mAh/day)', 'Number of Apps Installed',
                   'Data Usage (MB/day)', 'Age']

                sns.set_theme(style="whitegrid")

                plt.figure(figsize=(18, 10))
                for i, col in enumerate(numeric_columns, 1):
                    plt.subplot(2, 3, i)
                    sns.boxplot(
                        data=df,
                        x=col,
                        color='lightblue',
                        flierprops={'markerfacecolor': 'r', 'marker': 'o'}
                    )
                    plt.title(f'Boxplot of {col}', fontsize=14)
                    plt.xlabel('')

                plt.tight_layout()
                plt.show()
                """, language="python")
        
        image_path = "/workspaces/CSS145-BM4-G5_Project/assets/image01.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Boxplots of each numerical values.", use_container_width=True)
        except FileNotFoundError:
            st.write("Image file not found. Make sure 'image01.png' is in the correct path.")

    with st.expander("😸 Analysis: Boxplots"):
        st.markdown("""
                    The boxplots show potential outliers in several of the numeric columns. Specifically, you can observe:

                    1. `App Usage Time (min/day)` and `Screen On Time (hours/day)`
                        - Columns show some data points extending significantly above their median values, suggesting high usage outliers.
                    
                    2. `Battery Drain (mAh/day)` and `Data Usage (MB/day)`
                        - Columns have data points that could indicate unusually high device usage.

                    3. `Number of Apps Installed`
                        - Column displays a few cases with high app counts, possibly indicating heavy users.

                    4. `Age`
                        - Column has a few outlier points but appears relatively well-distributed within its range.
                    """)
    
    with st.expander("🐈 Code for Outlier Detection: IQR Method"):
        st.code("""
                outliers_detected = {}

                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outliers_detected[col] = {
                        "Lower Bound": lower_bound,
                        "Upper Bound": upper_bound,
                        "Outliers Count": len(outliers)
                    }

                outliers_detected
                """, language="python")
        
        outlier_info = {
            'App Usage Time (min/day)': {'Lower Bound': -368.25, 'Upper Bound': 915.75, 'Outliers Count': 0},
            'Screen On Time (hours/day)': {'Lower Bound': -4.85, 'Upper Bound': 14.75, 'Outliers Count': 0},
            'Battery Drain (mAh/day)': {'Lower Bound': -1538.625, 'Upper Bound': 4490.375, 'Outliers Count': 0},
            'Number of Apps Installed': {'Lower Bound': -46.0, 'Upper Bound': 146.0, 'Outliers Count': 0},
            'Data Usage (MB/day)': {'Lower Bound': -1079.0, 'Upper Bound': 2793.0, 'Outliers Count': 0},
            'Age': {'Lower Bound': -3.5, 'Upper Bound': 80.5, 'Outliers Count': 0}
        }

        outlier_df = pd.DataFrame(outlier_info).T
        st.table(outlier_df)
    
    with st.expander("😸 Analysis: IQR Method"):
        st.markdown("""
                    Using the **Interquartile Range (IQR) method**, no outliers were quantitatively detected for any of the continuous numeric columns. This suggests that the data distribution is relatively well-contained within 1.5 times the IQR from the lower and upper quartiles, or the potential outliers seen in the boxplots might not be extreme enough to surpass the threshold.
                    """)

    # ------------------------------ SUMMARY STATISTICS ------------------------------
        
    st.subheader("Summary Statistics")
    st.markdown(
        "We provided a quick overview of the dataset’s key numerical metrics, including measures such as the mean, median, standard deviation, minimum, and maximum values for each feature. These summary statistics offer insight into data distribution, central tendencies, and variability, helping to identify trends or anomalies before further analysis.")
    
    with st.expander("🐈 Code for Summary Statistics"):
        st.code("""
                summary_statistics = {}

                for col in numeric_columns:
                    summary_statistics[col] = {
                        "Mean": df[col].mean(),
                        "Median": df[col].median(),
                        "Variance": df[col].var(),
                        "Standard Deviation": df[col].std()
                    }

                summary_statistics
                """, language="python")


    
        # # Display Summary Statistics
        # st.subheader("Summary Statistics")
        # st.write(df.describe(include='all'))
        

        

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
