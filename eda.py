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
        
        file_path = "assets/user_behavior_dataset.csv"
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
            "assets/user_behavior_dataset.csv")

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
        "We used statistical methods and visualizations (box plots) to detect potential outliers in our numerical features.")

    with st.expander("🐈 Code for Outlier Detection: Boxplot"):
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
        
        image_path = "assets/image01.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Boxplots of each numerical values for outlier detection.", use_container_width=True)
        except FileNotFoundError:
            st.write("Image file not found. Make sure 'image01.png' is in the correct path.")

    with st.expander("😸 Analysis: Boxplot"):
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

    # ------------------------------ DISTRIBUTION ANALYSIS ------------------------------

    st.subheader("Distribution Analysis")
    st.markdown(
        "We examine the distribution of key numerical features within the dataset. By visualizing and analyzing the distribution, we gain insights into the central tendency, spread, and skewness of the data.")
    
    with st.expander("🐈 Code for Distribution Analysis: Histogram"):
        st.code("""
                plt.figure(figsize=(18, 12))

                for i, col in enumerate(numeric_columns, 1):
                    plt.subplot(3, 2, i)
                    sns.histplot(df[col], kde=True, color='royalblue', bins=30)
                    plt.title(f'Distribution of {col}', fontsize=14)
                    plt.xlabel(col)
                    plt.ylabel('Frequency')

                plt.tight_layout(pad=3)
                plt.show()
                """, language="python")
        
        image_path = "assets/image02.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Histogram of each numerical values to determine the distribution.",
                     use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image02.png' is in the correct path.")
        
    with st.expander("😸 Analysis: Histogram"):
        st.markdown("""
                    1. `App Usage Time (min/day)`
                        - The distribution is roughly right-skewed, indicating that while most users have moderate daily app usage, a smaller subset exhibits much higher usage times.
                        - The density plot highlights a peak in the lower usage range, suggesting that the majority of users engage in relatively short sessions.
                    
                    2. `Screen On Time (hours/day)`
                        - This feature shows a somewhat right-skewed distribution, where most users have moderate screen-on times.
                        - The tail towards higher values suggests that some users spend significantly more time on their devices, reflecting varying usage patterns across the dataset.

                    3. `Battery Drain (mAh/day)`
                        - The distribution of `battery drain` is relatively normal, with most users experiencing average battery consumption.
                        - Fewer users exhibit extreme values, suggesting that most individuals fall within a typical battery usage range, while a small minority have either very high or low drain rates.

                    4. `Number of Apps Installed`
                        - The distribution of the number of apps installed is heavily right-skewed, indicating that most users have a moderate number of apps.
                        - A smaller group of "power users" stands out with a significantly higher number of installed apps, contributing to the skew.
                    
                    5. `Data Usage (MB/day)`
                        - This feature is also right-skewed, indicating that while most users have average or moderate data consumption, some users exhibit much higher data usage.
                        - This pattern aligns with common data consumption habits, where a small group of users may engage in high-bandwidth activities.

                    6. `Age`
                        - The age distribution shows a peak in the younger demographic, with a noticeable decrease as age increases.
                        - This suggests that younger individuals are more likely to use mobile devices, which aligns with trends observed in mobile technology adoption among different age groups.
                    """)
    with st.expander("🐈 Code for Distribution Analysis: Bar Graph"):
        st.code("""
                categorical_columns = ['Device Model', 'Operating System', 'Gender']
                plt.figure(figsize=(18, 8))

                for i, col in enumerate(categorical_columns, 1):
                    plt.subplot(1, 3, i)
                    sns.countplot(data=df, x=col, palette='viridis')
                    plt.title(f'Count Plot of {col}', fontsize=14)
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=30)

                plt.tight_layout(pad=3)
                plt.show()
                """, language="python")

        image_path = "assets/image03.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Bar graphs of each categorical values to determine the distribution.",
                     use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image03.png' is in the correct path.")
    
    with st.expander("😸 Analysis: Bar Graph"):
        st.markdown("""
                    1. `Device Model`
                        - The bar plot shows the distribution of device models, with certain models (Xiami Mi 11 and iPhone 12) being more common among users. This indicates popularity, market dominance, or preferences for specific models.
                    
                    2. `Operating System`
                        - The operating system plot reveals the distribution of different OS platforms in use. A dominant OS (Android) have a larger share, highlighting potential user demographics or preferences.

                    3. `Gender`
                        - The distribution of genders shows the representation of male and female users. The male category is more dominant, it indicates a skew in user demographics, relevant for targeted analysis or marketing.
                    """)
        
    # ------------------------------ GRAPHICAL REPRESENTATIONS ------------------------------

    st.subheader("Graphical Representations")
    st.markdown(
        "We will present visual representations of the dataset to explore and understand key patterns, trends, and relationships between variables.")
    
    with st.expander("🐈 Code for Graphical Representations: Scatter Plot"):
        st.code("""
                plt.figure(figsize=(18, 6))

                # Scatter plot: Screen On Time and Battery Drain
                plt.subplot(1, 2, 1)
                sns.scatterplot(data=df, x='Screen On Time (hours/day)', y='Battery Drain (mAh/day)', hue='Gender', palette='Set2')
                plt.title('Screen On Time vs Battery Drain')
                plt.xlabel('Screen On Time (hours/day)')
                plt.ylabel('Battery Drain (mAh/day)')

                # Scatter plot: Screen On Time and App Usage Time
                plt.subplot(1, 2, 2)
                sns.scatterplot(data=df, x='Screen On Time (hours/day)', y='App Usage Time (min/day)', hue='Operating System', palette='Set1')
                plt.title('Screen On Time vs App Usage Time')
                plt.xlabel('Screen On Time (hours/day)')
                plt.ylabel('App Usage Time (min/day)')

                plt.tight_layout(pad=3)
                plt.show()
                """, language="python")
        
        image_path = "assets/image04.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Scatter plot to determine the representation.",
                     use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image04.png' is in the correct path.")

    with st.expander("😸 Analysis: Scatter Plot"):
        st.markdown("""
                    1. `Screen On Time vs. Battery Drain`
                        - Shows a positive trend between screen on time and battery drain. As screen-on time increases, battery drain generally increases, which aligns with expectations since longer screen usage typically requires more power. Different points are colored by gender, which helps visualize whether gender-specific patterns emerge.
                    
                    2. `Screen On Time vs. App Usage Time`
                        - Positive correlation between screen-on time and app usage can be seen. Users with longer screen-on times tend to use apps more frequently or for longer durations Different colors represent different operating systems.
                    """)
        
    with st.expander("🐈 Code for Graphical Representations: Box Plot"):
        st.code("""
                plt.figure(figsize=(15, 6))

                # Box plot segmented by Gender
                plt.subplot(1, 2, 1)
                sns.boxplot(data=df, x='Gender', y='Age', palette='pastel')
                plt.title('Age Distribution by Gender')
                plt.xlabel('Gender')
                plt.ylabel('Age')

                # Box plot segmented by Operating System
                plt.subplot(1, 2, 2)
                sns.boxplot(data=df, x='Operating System', y='Age', palette='muted')
                plt.title('Age Distribution by Operating System')
                plt.xlabel('Operating System')
                plt.ylabel('Age')

                plt.tight_layout(pad=3)
                plt.show()
                """, language="python")
        
        image_path = "assets/image05.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Boxplot of each Gender and OS to determine the representations.",
                     use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image05.png' is in the correct path.")

    with st.expander("😸 Analysis: Box Plot"):
        st.markdown("""
                    1. `Age Distribution by Gender`
                        - This plot compares the distribution of ages between different genders. Key metrics such as the median age, the range (interquartile range, or IQR), and potential outliers are shown for each gender.
                    
                    2. `Age Distribution by Operating System`
                        - This box plot visualizes how age distribution varies for different operating systems. It highlights any differences in the median, spread, and outlier presence of age among users of different OS platforms.
                    """)
    # ------------------------------ CORRELATION ANALYSIS ------------------------------

    st.subheader("Correlation Analysis")
    st.markdown(
        "We will examine the relationships between the variables in the dataset to identify potential correlations.")

    with st.expander("🐈 Code for Correlation Analysis: Heatmap"):
        st.code("""
                df_numeric = df_cleaned[numeric_columns]

                correlation_matrix = df_numeric.corr()

                # Correlation matrix using a heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
                plt.title("Correlation Matrix of Numerical Features")
                plt.show()
                """, language="python")

        image_path = "assets/image06.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Heatmap of each column to determine the correlation.",
                     use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image06.png' is in the correct path.")
            
    with st.expander("😸 Analysis: Heatmap"):
        st.markdown("""
                    1. `High Correlations`
                        - There is a strong positive correlation among `App Usage Time`, `Screen On Time`, `Battery Drain`, `Number of Apps Installed`, and `Data Usage`. For example, `App Usage Time` is highly correlated with `Screen On Time` (0.95) and `Battery Drain (0.96)`. This suggests these variables may measure overlapping aspects of user behavior.
                    
                    2. `Low Correlation`
                        - The `Age` feature shows very weak correlations with other variables, indicating it is relatively independent in this dataset.
                    """)
