# app.py
import streamlit as st
from PIL import Image
from enhance import Inferer
import os

def main():
    st.title("Low-Light Image Enhancer")

    inferer = Inferer()
    # weights_path = "weights/low_light_weights_best.h5"
    weights_path = "https://drive.google.com/file/d/1k_fve0bGykhUA8mkwObsGmCead76L6Fw/view?usp=sharing"

    if not os.path.exists(weights_path):
        st.error("Model weights not found. Please ensure 'low_light_weights_best.h5' is in the expected directory.")
        return

    inferer.build_model(num_rrg=3, num_mrb=2, channels=64, weights_path=weights_path)

    uploaded_file = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner('Enhancing image...'):
            original, enhanced = inferer.infer_streamlit(image)

        st.image(enhanced, caption="Enhanced Image", use_column_width=True)

        st.download_button(
            label="Download Enhanced Image",
            data=enhanced_to_bytes(enhanced),
            file_name="enhanced_image.png",
            mime="image/png"
        )

def enhanced_to_bytes(image):
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import streamlit.bootstrap
#     streamlit.bootstrap.run("app.py", "", [], {})
