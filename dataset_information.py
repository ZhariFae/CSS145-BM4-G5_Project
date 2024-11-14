import streamlit as st

def display_dataset_information():
    st.header("General Information")
    
    # Library and Dataset Imports
    st.subheader("CSS145-BM4 Group 5")
    st.write("GATMAITAN, Gilbert Jan")
    st.write("PALMA, Gian Carlo")
    st.write("REYES, Jedidiah")
    st.write("VILLAFRANCA, Johan Takkis")
    st.write("VIOLENTA, Erielson Emmanuel")

    # Dataset Analysis
    st.subheader("Dataset Analysis")
    st.write("Initial view of the dataset to understand its structure and attributes.")
    st.code("""
    # Dataset Read
    df_initial.head()

    # Original Copy
    df = df_initial.copy()

    # Dataset Inspect
    print("Dataset Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))
    """, language="python")
