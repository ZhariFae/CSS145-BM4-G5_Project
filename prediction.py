import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def display_prediction():
    st.header("Prediction")

    # Load dataset
    df = pd.read_csv('assets/user_behavior_dataset.csv')

    # Quantile thresholds for classification
    low_threshold_app_usage = df['App Usage Time (min/day)'].quantile(0.33)
    medium_threshold_app_usage = df['App Usage Time (min/day)'].quantile(0.66)
    low_threshold_screen_time = df['Screen On Time (hours/day)'].quantile(0.33)
    medium_threshold_screen_time = df['Screen On Time (hours/day)'].quantile(
        0.66)
    low_threshold_battery_drain = df['Battery Drain (mAh/day)'].quantile(0.33)
    medium_threshold_battery_drain = df['Battery Drain (mAh/day)'].quantile(
        0.66)

    # Classification function
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

    # Apply classification
    df['Engagement Level'] = df.apply(classify_engagement, axis=1)

    # Prepare data for SVM
    X_svm = df[[
        'App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)']]
    y_svm = df['Engagement Level'].map({'low': 0, 'medium': 1, 'high': 2})

    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_svm, y_svm, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
                                   'Low', 'Medium', 'High'], output_dict=True)

    # Layout: Inputs on the left, graph on the right
    col1, col2 = st.columns(2, gap="medium")

    # Inputs section
    with col1:
        st.subheader("Input Values for Prediction")
        app_usage = st.number_input(
            "App Usage Time (minutes/day)", min_value=0, max_value=1440, value=100, step=1)
        screen_time = st.number_input(
            "Screen On Time (hours/day)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
        battery_drain = st.number_input(
            "Battery Drain (mAh/day)", min_value=0, max_value=10000, value=500, step=50)

        if st.button("Predict Engagement Level"):
            user_input = np.array([[app_usage, screen_time, battery_drain]])
            user_input_scaled = scaler.transform(user_input)
            prediction = svm_model.predict(user_input_scaled)
            engagement_mapping = {0: "Low", 1: "Medium", 2: "High"}
            st.write(
                f"Predicted Engagement Level: **{engagement_mapping[prediction[0]]}**")

        # Display stats below inputs
        st.subheader("Model Performance Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(
            {'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))

    # Graph section
    with col2:
        st.subheader("Predicted vs. Actual Test Set")
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(y_test, y_pred, alpha=0.7,
                             edgecolors='k', label="Data Points")
        ax.plot([0, 2], [0, 2], color='red',
                linestyle='--', label="Ideal Line")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_yticklabels(["Low", "Medium", "High"])
        ax.set_xlabel("Actual Engagement Level")
        ax.set_ylabel("Predicted Engagement Level")
        ax.set_title("Predicted vs. Actual Engagement Levels")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)