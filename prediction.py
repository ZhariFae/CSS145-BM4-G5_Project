 import streamlit as st
 import numpy as np
 import pandas as pd
 from sklearn.svm import SVC
 from sklearn.preprocessing import StandardScaler
 from sklearn.model_selection import train_test_split
 from PIL import Image

st.set_page_config(page_title="Prediction", layout="wide")

 def display_prediction():
     st.header("Prediction")

 df = pd.read_csv('assets/user_behavior_dataset.csv')

 low_threshold_app_usage = df['App Usage Time (min/day)'].quantile(0.33)
 medium_threshold_app_usage = df['App Usage Time (min/day)'].quantile(0.66)

 low_threshold_screen_time = df['Screen On Time (hours/day)'].quantile(0.33)
 medium_threshold_screen_time = df['Screen On Time (hours/day)'].quantile(0.66)

 low_threshold_battery_drain = df['Battery Drain (mAh/day)'].quantile(0.33)
 medium_threshold_battery_drain = df['Battery Drain (mAh/day)'].quantile(0.66)

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

 df['Engagement Level'] = df.apply(classify_engagement, axis=1)

 # SVM Training
 X_svm = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)']]
 y_svm = df['Engagement Level'].map({'low': 0, 'medium': 1, 'high': 2})

 X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.3, random_state=42)

 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train)
 X_test_scaled = scaler.transform(X_test)

 svm_model = SVC(kernel='linear', random_state=42)
 svm_model.fit(X_train_scaled, y_train)

 # Streamlit Prediction Page
 st.title("Engagement Level Prediction with SVM")
 st.write("Input values for App Usage Time, Screen On Time, and Battery Drain to predict user engagement level.")

 app_usage = st.number_input("App Usage Time (minutes/day)", min_value=0, max_value=1440, value=100, step=1)
 screen_time = st.number_input("Screen On Time (hours/day)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
 battery_drain = st.number_input("Battery Drain (mAh/day)", min_value=0, max_value=10000, value=500, step=50)

 if st.button("Predict Engagement Level"):
     user_input = np.array([[app_usage, screen_time, battery_drain]])
     user_input_scaled = scaler.transform(user_input)
     prediction = svm_model.predict(user_input_scaled)
     engagement_mapping = {0: "Low", 1: "Medium", 2: "High"}
     st.write(f"Predicted Engagement Level: **{engagement_mapping[prediction[0]]}**")
