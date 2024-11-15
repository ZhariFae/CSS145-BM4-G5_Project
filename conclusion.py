import streamlit as st
import pandas as pd
import io

from PIL import Image

def display_conclusion():
    st.header("Conclusion")

    st.markdown("""
                With a focus on using exploratory data analysis (EDA) to identify important use patterns and machine learning models to deliver descriptive and predictive insights, this project provided a comprehensive examination of user behavior and mobile device usage. Thanks to the valuable information from each analysis step, we now have a greater understanding of user engagement, device preferences, and usage intensity.
                """)

    st.subheader("Exploratory Data Analysis")
    st.markdown("""
                The EDA phase uncovered fundamental trends and correlations in user behavior:
                """)
    st.markdown("""
                `Screen Time and Battery Drain`
                - The analysis showed a clear correlation between screen-on time and battery drain across all user groups. High screen time reliably translates to higher battery consumption, identifying a subset of high-engagement users likely to prioritize battery efficiency in their device choices.

                `Engagement Patterns by Device and Demographic`
                - By examining variables like device model, age, and gender, the EDA highlighted differences in engagement levels based on demographic factors. While gender and age had weaker correlations with usage patterns, the device model emerged as a significant differentiator, with certain devices associated with longer screen-on times and higher app usage.

                `Distribution Analysis and Outliers`
                - Visualizations, including box plots and histograms, revealed outliers in variables such as app usage time, battery drain, and data usage. The presence of these outliers suggests a distinct group of power users who engage intensively with their devices, unlike the majority of users whose behavior clusters closer to the mean. This insight provides a clearer picture of the variation within user behavior.

                 `User Profiles`
                - Through clustering analysis, we identified three distinct user profiles—Casual Users, Moderate Users, and Power Users. Each group showed distinct behavior patterns, with Power Users displaying high screen-on time and data usage, while Casual Users had minimal app usage and battery drain. This segmentation is valuable for recognizing varied user needs and engagement levels within the dataset.
                """)

    st.subheader("Machine Learning Implementation")

    st.markdown("""
                Building on the EDA insights, we implemented machine learning models to classify user characteristics and predict engagement levels:
                """)

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
                - The SVM PCA visualization of engagement levels illustrates clear clusters, differentiating users based on their engagement level (low, medium, or high). The three distinct colors represent these levels, showing that SVM was able to separate users effectively. The horizontal spread along the PCA components indicates a range of usage behaviors within each level, with high-engagement users positioned farthest to the right, followed by medium, and then low-engagement users. This supports the SVM model’s capability to distinguish engagement intensity based on app usage, screen time, and battery drain metrics.

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
