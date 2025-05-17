# app.py

import streamlit as st
from PIL import Image
from enhance import Inferer
import os
import gdown

def download_model(weights_path):
    """Download the model from Google Drive if not exists."""
    if not os.path.exists(weights_path):
        st.warning("Model weights not found locally. Downloading from Google Drive...")
        file_id = "1zygtod4QnQW-8WrMxZMOGUfGjwnEBydO"
        # https://drive.google.com/file/d/1zygtod4QnQW-8WrMxZMOGUfGjwnEBydO/view?usp=drive_link
        gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", weights_path, quiet=False)

def main():
    st.title("Low-Light Image Enhancer")

    weights_path = "weights/low_light_weights_best.h5"
    download_model(weights_path)

    if not os.path.exists(weights_path):
        st.error("Failed to load model weights.")
        return

    inferer = Inferer()
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
    return buf.getvalue()

if __name__ == "__main__":
    main()


# import streamlit as st
# from PIL import Image
# from enhance import Inferer
# import os
# import requests

# MODEL_FILE_ID = "1k_fve0bGykhUA8mkwObsGmCead76L6Fw"
# MODEL_PATH = "weights/low_light_weights_best.h5"

# def download_model():
#     url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
#     os.makedirs("weights", exist_ok=True)
#     response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open(MODEL_PATH, "wb") as f:
#             for chunk in response.iter_content(1024 * 1024):
#                 f.write(chunk)
#     else:
#         st.error("Failed to download model weights.")
#         st.stop()

# def main():
#     st.title("Low-Light Image Enhancer")

#     if not os.path.exists(MODEL_PATH):
#         st.warning("Model file not found, downloading from Google Drive...")
#         download_model()

#     inferer = Inferer()
#     inferer.build_model(num_rrg=3, num_mrb=2, channels=64, weights_path=MODEL_PATH)

#     uploaded_file = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Original Image", use_column_width=True)

#         with st.spinner('Enhancing image...'):
#             original, enhanced = inferer.infer_streamlit(image)

#         st.image(enhanced, caption="Enhanced Image", use_column_width=True)

#         st.download_button(
#             label="Download Enhanced Image",
#             data=enhanced_to_bytes(enhanced),
#             file_name="enhanced_image.png",
#             mime="image/png"
#         )

# def enhanced_to_bytes(image):
#     from io import BytesIO
#     buf = BytesIO()
#     image.save(buf, format="PNG")
#     return buf.getvalue()

# if __name__ == "__main__":
#     main()


# if __name__ == "__main__":
#     import streamlit.bootstrap
#     streamlit.bootstrap.run("app.py", "", [], {})


