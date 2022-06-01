from turtle import onclick
import streamlit as st
import pandas as pd
import numpy as np
import os

st.title("Generic Weights")

BASE_DIR = "DataFiles/"

def update_df(df, w):
    df['weight'] = w
    df.to_csv(os.path.join(BASE_DIR, "final/topic_map.csv"), index=False)

df = pd.read_csv(os.path.join(BASE_DIR, "final/topic_map.csv"))

weights = df.weight.values.tolist()
for i, row in df.iterrows():
    st.markdown("""---""")
    st.text(row.topic)
    weights[i] = st.number_input('', min_value=0.1, max_value=2.0, value=row.weight, step=0.05, key=i)
    st.button("update", on_click=update_df, args=(df, weights), key=f"b {i}")

