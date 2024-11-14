import streamlit as st

def display_dataset_information():
    st.header("General Information")
    
    # Group Information
    st.subheader("CSS145-BM4 Group 5")
    st.write("GATMAITAN, Gilbert Jan")
    st.write("PALMA, Gian Carlo")
    st.write("REYES, Jedidiah")
    st.write("VILLAFRANCA, Johan Takkis")
    st.write("VIOLENTA, Erielson Emmanuel")

    # Dataset Information
    st.subheader("Dataset Information")
    st.write("The dataset used for this project can be found here:")
    st.page_link("https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset", label="Mobile Device Usage and User Behavior Dataset", icon=None, help=None, disabled=False, use_container_width=None)

    st.subheader("Dataset Description")
    st.write("This dataset comprehensively analyzes trends in mobile device usage and user behavior classification. It includes 700 user data samples encompassing metrics like data consumption, battery drain, screen-on time, and app usage duration. Each input is categorized into one of five user behavior groups, ranging from mild to excessive usage, to enable meaningful analysis and modeling.")

    st.subheader("Column Description")
    st.write("User ID")
    st.write("A unique identifier is assigned to each user, used to distinguish individual records without revealing personal information.")
    st.write("Device Model")
    st.write("The specific model of the mobile device, which can influence app performance and user behavior.")
    st.write("Operating System")
    st.write("The operating system of the device (e.g., Android, iOS), which could impact app availability and user experience.")
    st.write("App Usage Time (min/day)")
    st.write("The average daily time in minutes that the user spends on apps, reflects engagement and usage levels.")
    st.write("Screen On Time (hours/day)")
    st.write("Total daily screen-on time in hours, providing a broader measure of device usage beyond app activity.")
    st.write("Battery Drain (mAh/day)")
    st.write("Average daily battery consumption in milliampere-hours (mAh), giving insight into device power usage based on activity.")
    st.write("Number of Apps Installed")
    st.write("The total number of apps installed on the device may indicate the user’s app variety and engagement.")
    st.write("Data Usage (MB/day)")
    st.write("Daily data usage in megabytes, showing the extent of online activity and content consumption.")
    st.write("User ID:")
    st.write("User ID:")
    st.write("User ID:")
    st.write("User ID:")
