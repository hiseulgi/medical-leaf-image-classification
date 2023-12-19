import base64
from typing import List, Tuple

import httpx
import rootutils
import streamlit as st
from PIL import Image

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

SAMPLE_IMAGE = {
    "betel": str(ROOT / "src" / "web" / "asset" / "betel.jpg"),
    "lemon": str(ROOT / "src" / "web" / "asset" / "lemon.jpg"),
    "mint": str(ROOT / "src" / "web" / "asset" / "mint.jpg"),
}


def predict(image_bytes: bytes) -> Tuple[int, List[any]]:
    r = httpx.post("http://api:6969/v1/predictions/knn", files={"image": image_bytes})
    if r.status_code != 200:
        return r.status_code, {}

    return r.status_code, r.json()


def main():
    st.set_page_config(
        page_title="Medical Leaf Classification",
        page_icon=":herb:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Medical Leaf Classification :herb:")

    col1, col2 = st.columns([2, 3])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a segmented leaf image", type=["png", "jpg", "jpeg"]
        )
        sample = st.selectbox(
            "Or choose from sample images:", list(SAMPLE_IMAGE.keys())
        )

        image_bytes = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_bytes = uploaded_file

        elif sample:
            image = Image.open(SAMPLE_IMAGE[sample])
            st.image(image, caption="Sample Image", use_column_width=True)
            image_bytes = open(SAMPLE_IMAGE[sample], "rb")

    with col2:
        if image_bytes is not None:
            if st.button(
                "Predict", type="primary", key="predict", use_container_width=True
            ):
                with st.spinner("Predicting..."):
                    status_code, predictions = predict(image_bytes)

                if status_code != 200:
                    st.error("Error fetching data!", icon="❌")
                    st.stop()
                st.success(f"Status code: {status_code}", icon="✅")

                st.write("Predictions:")
                for data in predictions:
                    st.progress(
                        data["score"], text=f'{data["label"]} - {data["score"]}'
                    )


if __name__ == "__main__":
    main()
