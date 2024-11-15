import streamlit as st
import pandas as pd
import io

from PIL import Image

def display_conclusion():
    st.header("Conclusion")

    st.subheader("Exploratory Data Analysis")
    st.markdown("""
                `Screen Time and Battery Drain`
                - We noticed a clear link between screen time and battery drain—more time spent with the screen on typically led to higher battery consumption across all users.

                `Engagement Patterns by Device and Demographic`
                - When we looked at gender and age, we found that these factors didn't strongly influence usage patterns.
                - The type of device really stood out. Certain device models showed much longer screen-on times and higher app usage.

                `Distribution Analysis and Outliers`
                - App usage time, battery drain, and data usage had outliers that pointed to a distinct group of "power users" who use their devices far more intensively than most people. 

                 `User Profiles`
                - We identified three main types of users: Casual Users, Moderate Users, and Power Users. These profiles give us a clearer view of different user behaviors and their varying levels of device engagement.
                """)

    st.subheader("Machine Learning Implementation")

    image_path = "assets/image10.png"
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.write(
            "Image file not found. Make sure 'image010.png' is in the correct path.")

    st.subheader("Decision Tree Classifier - Predicting Operating System")
    st.markdown("""
                The Decision Tree model was trained to predict the operating system (OS) based on user demographics and device-specific features.

                `Features`
                - Age, Gender, Battery Drain, and Device Model.

                `Results`
                - The classifier achieved a reasonably high accuracy, indicating its ability to distinguish between Android and iOS users. It effectively showed key features that impact OS preference, showing a clear relationship between device model and OS choice.

                `Observation`
                - The device model is the most critical factor in predicting the operating system (OS). Models like iPhone 12 have strong predictive power, because it is the only IOS device in the data pool.

                `Conclusion`
                - It effectively leverages device models to predict OS, confirming that device-specific attributes dominate OS classification.
                """)

    st.subheader(
        "Support Vector Machine (SVM) - Classifying Engagement Levels")
    st.markdown("""
                The Support Vector Machine (SVM) model classified users into different engagement levels (low, medium, and high) based on thresholds set for App Usage Time, Screen On Time, and Battery Drain:

                `Threshold-Based Classification`
                - Engagement levels were derived based on the 33rd and 66th percentiles.

                `Model Performance`
                - The SVM model achieved good accuracy, particularly in distinguishing low and high engagement users, highlighting that usage patterns and device strain differ notably between these groups.

                `Observation`
                - The SVM PCA visualization of engagement levels illustrates clear clusters, differentiating users based on their engagement level.

                `Conclusion`
                - Successful in classifying engagement levels, where high-engagement users show significantly different patterns.
                """)
    
    st.subheader("Clustering Analysis (KMeans) - Segmenting User Profiles")
    st.markdown("""
                KMeans clustering grouped users into distinct profiles, allowing us to explore patterns without pre-defined labels:

                `Features`
                - Screen On Time, Battery Drain, and Data Usage

                `User Profiles`
                - Casual Users: Lower screen time, battery usage, and data consumption.
                - Moderate Users: Moderate values across the features.
                - Power Users: Highest screen time, battery drain, and data usage.

                `Observation`
                - Clustering shows the existence of different user profiles, which can help in creating customized experiences for each segment.

                `Conclusion`
                - Successfully segments users into profiles based on usage intensity.
                """)
