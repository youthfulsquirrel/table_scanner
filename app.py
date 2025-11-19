import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from processor import process_table

st.title("📊 Table Digitizer – Shaded Cell Counter")

uploaded = st.file_uploader("Upload your table photo", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded)

    with st.spinner("Processing…"):
        try:
            matrix, col_sums = process_table(img)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("Done!")

    st.subheader("Digitized Matrix (0/1)")
    df_matrix = pd.DataFrame(matrix)
    st.dataframe(df_matrix)

    st.subheader("Column Totals")
    df_sums = pd.DataFrame({"Sum": col_sums})
    st.dataframe(df_sums)

    st.download_button(
        "Download matrix as CSV",
        df_matrix.to_csv(index=False),
        file_name="matrix.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download column sums as CSV",
        df_sums.to_csv(),
        file_name="column_sums.csv",
        mime="text/csv"
    )