import streamlit as st
import pandas as pd
import io

from PIL import Image

def display_ml_implementation():
    st.header("Machine Learning Implementation")

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

                # Copy cleaned dataset
                df_ml = df_cleaned.copy()
                """, language="python")

    # ------------------------------ DECISION TREE CLASSIFIER ------------------------------

    st.subheader("Decision Tree Classifier: Classify Operating System")
    st.markdown(
        "Decision trees are excellent for classification tasks where interpretability is key. In this context, they can help classify users based on various features such as Age, Gender, Battery Drain, and Device Model, with respect to Operating System (Android/iOS). This can provide insights for app developers or advertisers on which demographic segments favor each OS. Decision trees are easy to understand, visually interpretable, and can capture non-linear relationships between features.")

    with st.expander("🐈 Code for Decision Tree Classifier"):
        st.code("""
                # Prepare the data
                X = df_cleaned[['Age', 'Gender', 'Battery Drain (mAh/day)', 'Device Model']]  # Features
                y = df_cleaned['Operating System']  # Target variable

                # Convert categorical variables to numerical ones
                X = pd.get_dummies(X, drop_first=True)

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Initialize and train the Decision Tree classifier
                dt_model = DecisionTreeClassifier(random_state=42)
                dt_model.fit(X_train, y_train)

                # Predictions
                y_pred = dt_model.predict(X_test)

                # Model evaluation
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
                print("Confusion Matrix:")
                print(confusion_matrix(y_test, y_pred))
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))

                # Visualize the Decision Tree
                plt.figure(figsize=(12, 8))
                plot_tree(dt_model, feature_names=X.columns, class_names=dt_model.classes_, filled=True)
                plt.title("Decision Tree Classifier Visualization")
                plt.show()
                """, language="python")
        
        image_path = "assets/image07.png"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image07.png' is in the correct path.")
            
    with st.expander("😸 Analysis: Decision Tree"):
        st.markdown("""
                    1. `Graph Interpretation`
                        - This bar plot visualizes the importance of each feature in predicting the Operating System (Android or iOS) using the Decision Tree model.
                    
                    2. `Key Observations`
                        - The Device Model_iPhone 12 feature has the highest importance, indicating that this model of phone is strongly associated with one of the operating systems (likely iOS). Other device models, along with Gender_Male and Battery Drain, show minimal importance in comparison.

                    3. `Conclusion`
                        - The Decision Tree model relies heavily on specific device models to predict the operating system, which aligns with the expectation that certain devices are exclusive to particular operating systems. This insight may be less useful for predicting OS in future data if new devices are introduced, as it depends on specific device information rather than more generalized features.
                    """)
        
    # ------------------------------ SUPPORT VECTOR MACHINE ------------------------------

    st.subheader(
        "Support Vector Machine (SVM): Engagement Level Classification")
    st.markdown(
        "SVM is particularly suited for binary or multi-class classification tasks where finding the optimal boundary between classes is critical. Here, SVM can classify users into different engagement levels (low, medium, high) based on features such as App Usage Time, Screen On Time, and Battery Drain.")

    with st.expander("🐈 Code for Decision Tree Classifier"):
        st.code("""
                    low_threshold_app_usage = df['App Usage Time (min/day)'].quantile(0.33)
                    medium_threshold_app_usage = df['App Usage Time (min/day)'].quantile(0.66)

                    low_threshold_screen_time = df['Screen On Time (hours/day)'].quantile(0.33)
                    medium_threshold_screen_time = df['Screen On Time (hours/day)'].quantile(0.66)

                    low_threshold_battery_drain = df['Battery Drain (mAh/day)'].quantile(0.33)
                    medium_threshold_battery_drain = df['Battery Drain (mAh/day)'].quantile(0.66)

                    # Define a function to classify engagement levels
                    def classify_engagement(row):
                        if (row['App Usage Time (min/day)'] <= low_threshold_app_usage and
                            row['Screen On Time (hours/day)'] <= low_threshold_screen_time and
                            row['Battery Drain (mAh/day)'] <= low_threshold_battery_drain):
                            return 'low'
                        elif (row['App Usage Time (min/day)'] <= medium_threshold_app_usage and
                            row['Screen On Time (hours/day)'] <= medium_threshold_screen_time and
                            row['Battery Drain (mAh/day)'] <= medium_threshold_battery_drain):
                            return 'medium'
                        else:
                            return 'high'

                    # Apply the function to create the new column
                    df['Engagement Level'] = df.apply(classify_engagement, axis=1)

                    # Check the distribution of engagement levels
                    print(df['Engagement Level'].value_counts())
                """, language="python")

        engagement_data = {
            'Engagement Level': ['high', 'medium', 'low'],
            'Count': [273, 244, 183]
        }
        df = pd.DataFrame(engagement_data)
        st.table(df)

        st.code("""
                X_svm = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)']]
                y_svm = df['Engagement Level']  # Assuming 'Engagement Level' is a column

                # Encoding target variable for SVM
                y_svm = y_svm.map({'low': 0, 'medium': 1, 'high': 2})

                # Train-test split
                X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.3, random_state=42)

                # SVM Model
                svm_model = SVC(kernel='linear', random_state=42)
                svm_model.fit(X_train_svm, y_train_svm)

                # SVM Predictions and Evaluation
                y_pred_svm = svm_model.predict(X_test_svm)
                svm_accuracy = accuracy_score(y_test_svm, y_pred_svm)
                print("SVM Accuracy:", svm_accuracy)
                print(classification_report(y_test_svm, y_pred_svm))

                # Visualizing SVM (2D PCA for visualization purposes)
                pca = PCA(n_components=2)
                X_svm_pca = pca.fit_transform(X_svm)
                plt.figure(figsize=(6, 6))
                plt.scatter(X_svm_pca[:, 0], X_svm_pca[:, 1], c=y_svm, cmap='viridis', edgecolor='k', s=50)
                plt.title('SVM Classification of Engagement Levels')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                """, language="python")

        image_path = "assets/image08.png"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image08.png' is in the correct path.")
            
    with st.expander("😸 Analysis: SVM"):            
        st.markdown("""
                    1. `Graph Interpretation`
                        - This scatter plot shows the classification of engagement levels (low, medium, high) based on a Support Vector Machine (SVM) model, plotted in a reduced two-dimensional space using PCA (Principal Component Analysis).

                    2. `Key Observations`
                        - The three engagement levels appear as clusters with clear separations, indicating that the SVM model was successful in distinguishing between low, medium, and high engagement users based on App Usage Time, Screen On Time, and Battery Drain. Each engagement level forms a distinct cluster, with some overlap between adjacent levels.

                    3. `Conclusion`
                        - The SVM model is effective in segmenting users into distinct engagement levels. This could be valuable for creating tailored experiences based on engagement patterns, as well as for targeted marketing or user retention strategies. However, the presence of slight overlaps suggests that some users' behaviors fall near the boundaries of engagement categories, which could result in occasional misclassification.
                    """)
    
    # ------------------------------ CLUSTERING ------------------------------

    st.subheader("Decision Tree Classifier: Classify Operating System")
    st.markdown(
        "Clustering is used for unsupervised learning, which is ideal for creating user profiles based on similar characteristics. By clustering users based on Screen On Time, Battery Drain, and Data Usage, we can group them into segments like power users and casual users. This segmentation can be useful for tailoring marketing strategies, optimizing app features, and addressing user needs.")

    with st.expander("🐈 Code for Clustering"):
        st.code("""
                X_clustering = df[['Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)']]

                # KMeans model to identify clusters
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(X_clustering)

                # Naming Clusters based on profiles
                cluster_names = {0: 'Casual Users', 1: 'Moderate Users', 2: 'Power Users'}
                cluster_labels = [cluster_names[label] for label in clusters]

                # Adding cluster labels to dataset for visualization
                df['User Profile'] = cluster_labels

                # Visualize the clusters
                plt.figure(figsize=(6, 6))
                plt.scatter(X_clustering['Screen On Time (hours/day)'], X_clustering['Battery Drain (mAh/day)'], c=clusters, cmap='viridis', edgecolor='k', s=50)
                plt.title('Clustering Analysis of User Profiles')
                plt.xlabel('Screen On Time (hours/day)')
                plt.ylabel('Battery Drain (mAh/day)')
                """, language="python")
        
        image_path = "assets/image09.png"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
        except FileNotFoundError:
            st.write(
                "Image file not found. Make sure 'image09.png' is in the correct path.")

    with st.expander("😸 Analysis: Clustering"):
        st.markdown("""
                    1. `Graph Interpretation`
                        - This scatter plot shows the results of a KMeans clustering analysis on user behavior based on Screen On Time, Battery Drain, and Data Usage. Each color represents a different cluster, corresponding to a specific user profile.

                    2. `Key Observations`
                        - There are three distinct clusters representing different user profiles: Casual Users: Lower screen time and battery usage, likely indicating infrequent or short-duration app usage. Moderate Users: Moderate screen time and battery usage, suggesting more balanced usage patterns. Power Users: High screen time, battery drain, and data usage, indicating heavy engagement with apps and potentially resource-intensive activities.

                    3. `Conclusion`
                        - The clustering analysis reveals clear user segments that can inform personalized app features or marketing strategies. For example, power users may appreciate app performance optimizations, while casual users may benefit from simplified interfaces. Recognizing these user profiles can help businesses target user-specific strategies and enhance user experience for different engagement levels.
                    """)
