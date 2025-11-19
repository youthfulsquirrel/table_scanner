import streamlit as st
from PIL import Image
import numpy as np
import io
import pandas as pd
from processor import process_table, draw_overlay
import base64
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("📊 Table Digitizer – Click 4 Corners")

# Session state for clicked corners
if "corners" not in st.session_state:
    st.session_state.corners = []

uploaded = st.file_uploader("Upload table photo", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    img_width, img_height = img.size

    # Embed image in HTML canvas for click detection
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    html_code = f"""
    <html>
    <body>
    <p>Click 4 corners on the image (TL → TR → BR → BL):</p>
    <canvas id="canvas" width={img_width} height={img_height} 
        style="border:1px solid black;background-image:url('data:image/png;base64,{img_str}');background-size:contain;">
    </canvas>
    <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var points = [];
        canvas.addEventListener('click', function(evt){{
            if(points.length >= 4) return;
            var rect = canvas.getBoundingClientRect();
            var x = evt.clientX - rect.left;
            var y = evt.clientY - rect.top;
            points.push([Math.round(x), Math.round(y)]);
            ctx.fillStyle = "red";
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            const el = document.getElementById('corner_input');
            el.value = JSON.stringify(points);
            el.dispatchEvent(new Event('change'));
        }});
    </script>
    <textarea id="corner_input" style="display:none;"></textarea>
    </body>
    </html>
    """
    components.html(html_code, height=img_height + 50)

    # Read clicks from hidden textarea
    coords_str = st.text_input("Clicked points (hidden)", key="corner_input")
    if coords_str:
        try:
            points = np.array(eval(coords_str), dtype="float32")
            st.session_state.corners = points
        except:
            st.session_state.corners = []

    if len(st.session_state.corners) > 0:
        st.write(f"Points clicked: {st.session_state.corners.tolist()}")

    # Submit corners
    if len(st.session_state.corners) == 4:
        if st.button("Submit Corners"):
            st.success("Processing table...")

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            image_bytes = img_byte_arr.getvalue()

            # Process table
            matrix, col_sums, overlay_img = process_table(image_bytes, st.session_state.corners, cols=48)

            # Editable matrix
            st.subheader("Digitized Matrix")
            df_matrix = pd.DataFrame(matrix)
            edited_df = st.data_editor(df_matrix, num_rows="dynamic")
            edited_matrix = edited_df.to_numpy().astype(int)

            # Overlay
            updated_overlay = draw_overlay(overlay_img, edited_matrix)
            st.subheader("Overlay on Image")
            st.image(updated_overlay, channels="BGR")

            # Column sums
            col_sums = edited_matrix.sum(axis=0)
            st.subheader("Column Totals")
            st.dataframe(pd.DataFrame({"Sum": col_sums}))

            # Download CSV
            st.download_button(
                "Download matrix CSV",
                edited_df.to_csv(index=False),
                file_name="matrix.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download column sums CSV",
                pd.DataFrame({"Sum": col_sums}).to_csv(index=False),
                file_name="column_sums.csv",
                mime="text/csv"
            )
