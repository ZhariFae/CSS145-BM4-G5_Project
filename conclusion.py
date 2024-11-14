import streamlit as st
import pandas as pd
import io

from PIL import Image

def display_conclusion():
    st.header("Conclusion")

    st.subheader("Exploratory Data Analysis")
    st.markdown("""
                `User Engagement Levels and Battery Usage`
                - This scatter plot shows the results of a KMeans clustering analysis on user behavior based on Screen On Time, Battery Drain, and Data Usage. Each color represents a different cluster, corresponding to a specific user profile.

                `Device Model and Demographic Influence`
                - The breakdown of device models and user demographics, such as age and gender, reveals some trends but limited correlations. While device model choice is a significant factor, age and gender do not strongly impact battery usage or screen time. This observation suggests that while demographics may play a role in device selection, they have less influence on actual usage behavior.

                `Clustering of User Profiles`
                - Clustering analysis based on screen-on time and battery drain identified distinct groups of users, potentially corresponding to different usage intensities. These clusters categorize users into heavy, moderate, and light usage profiles. Heavy users show both high screen-on time and battery drain, indicating intense device use, while lighter users have more conservative usage patterns. This categorization could help in understanding user needs and tailoring device features to better suit different user profiles.

                 `Correlation Analysis`
                - The correlation analysis further highlights the relationship between key variables, particularly battery drain and screen-on time. These two features showed a moderate positive correlation, reinforcing that higher screen usage corresponds to increased battery consumption. Other factors like age and gender showed weaker correlations, indicating that demographic variables have limited influence on battery behavior compared to screen time.
                """)

    st.subheader("Machine Learning Implementation")

    image_path = "/workspaces/CSS145-BM4-G5_Project/assets/image10.png"
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.write(
            "Image file not found. Make sure 'image010.png' is in the correct path.")

    st.subheader("Decision Tree Classifier - Predicting Operating System")
    st.markdown("""
                The Decision Tree Classifier was used to predict a user’s operating system based on demographics and device usage metrics:

                `Features`
                - The model utilized Age, Gender, Battery Drain, and Device Model as input features.

                `Results`
                - The classifier achieved a reasonably high accuracy, indicating its ability to distinguish between Android and iOS users. The Decision Tree model effectively highlighted key features that impact OS preference, showing a clear relationship between device model and OS choice, as expected.

                `Visualization`
                - A decision tree plot offered insights into the decision-making process of the model, where device model and battery usage patterns were primary drivers in OS prediction.

                `Observation`
                - The Decision Tree feature importance graph reveals that the device model is the most critical factor in predicting the operating system (OS). Specifically, models like iPhone 12 have strong predictive power, which likely indicates that certain models are exclusively associated with either iOS or Android platforms. This insight confirms that device-specific attributes are highly indicative of OS choice, aligning with known market trends where certain devices exclusively use a particular OS (e.g., iPhone for iOS, Samsung for Android).

                `Conclusion`
                - The Decision Tree model effectively leverages device models to predict OS, confirming that device-specific attributes dominate OS classification.
                """)

    st.subheader(
        "Support Vector Machine (SVM) - Classifying Engagement Levels")
    st.markdown("""
                The Support Vector Machine (SVM) model classified users into different engagement levels (low, medium, and high) based on thresholds set for App Usage Time, Screen On Time, and Battery Drain:

                `Threshold-Based Classification`
                - Engagement levels were derived based on the 33rd and 66th percentiles, categorizing users into low, medium, and high engagement.

                `Model Performance`
                - The SVM model achieved good accuracy, particularly in distinguishing low and high engagement users, highlighting that usage patterns and device strain differ notably between these groups.

                `Visualization`
                - A PCA (Principal Component Analysis) visualization provided a 2D view of the engagement levels, where clusters of users with similar usage behavior emerged, validating the classification approach.

                `Observation`
                - The SVM PCA visualization of engagement levels illustrates clear clusters, differentiating users based on their engagement level (low, medium, or high). The three distinct colors represent these levels, showing that SVM was able to separate users effectively. The horizontal spread along the PCA components indicates a range of usage behaviors within each level, with high-engagement users positioned farthest to the right, followed by medium, and then low-engagement users. This supports the SVM model’s capability to distinguish engagement intensity based on app usage, screen time, and battery drain metrics..

                `Conclusion`
                - SVM is successful in classifying engagement levels, where high-engagement users show significantly different patterns, suggesting opportunities for targeted app features or resource optimization..
                """)
    
    st.subheader("Clustering Analysis (KMeans) - Segmenting User Profiles")
    st.markdown("""
                KMeans clustering grouped users into distinct profiles, allowing us to explore patterns without pre-defined labels:

                `Features`
                - The clustering algorithm used Screen On Time, Battery Drain, and Data Usage as inputs to differentiate user profiles.

                `User Profiles`
                - Casual Users: Characterized by lower screen time, battery usage, and data consumption.
                - Moderate Users: Showed moderate values across the features, likely representing average users.
                - Power Users: Represented users with the highest screen time, battery drain, and data usage, reflecting high-engagement individuals.

                `Visualization`
                - A scatter plot highlighted the user profiles, showing distinct clusters that aligned well with expected usage behaviors.

                `Observation`
                - The distinct clustering underscores the existence of different user profiles, which can help in creating customized experiences for each segment. For example, power users (high screen time and battery usage) may benefit from battery-saving features or notifications about data usage, while casual users may prioritize simpler, streamlined interactions.

                `Conclusion`
                - KMeans clustering successfully segments users into profiles based on usage intensity, providing valuable insights for product customization and targeted marketing.
                """)
