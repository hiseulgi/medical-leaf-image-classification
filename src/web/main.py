import sys

import rootutils

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(ROOT / "src")

from typing import Dict, Tuple

import httpx
import rootutils
import streamlit as st
from PIL import Image

from src.web.schema.api_schema import BlogResponseSchema, PredictionResponseSchema

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

SAMPLE_IMAGE = {
    "betel": str(ROOT / "src" / "web" / "static" / "betel.jpg"),
    "lemon": str(ROOT / "src" / "web" / "static" / "lemon.jpg"),
    "mint": str(ROOT / "src" / "web" / "static" / "mint.jpg"),
}


def get_predict(
    image_bytes: bytes, model_name: str = "knn"
) -> Tuple[int, PredictionResponseSchema]:
    r = httpx.post(
        f"http://api:6969/v1/predictions/{model_name}", files={"image": image_bytes}
    )
    if r.status_code != 200:
        return r.status_code, None

    return r.status_code, PredictionResponseSchema(**r.json())


def get_blog(class_name: str) -> Tuple[int, BlogResponseSchema]:
    r = httpx.get(f"http://api:6969/v1/blog?class_name={class_name}")
    if r.status_code != 200:
        return r.status_code, None
    return r.status_code, BlogResponseSchema(**r.json())


def main():
    st.set_page_config(
        page_title="Medical Leaf Classification",
        page_icon=":herb:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Medical Leaf Classification :herb:")

    col1, _, col3 = st.columns([5, 1, 5])

    with col1:
        model_name = st.selectbox("Choose a model:", ["knn", "mobilenet"])
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

    with col3:
        if image_bytes is not None:
            if st.button(
                "Predict", type="primary", key="predict", use_container_width=True
            ):
                # predict route
                st.markdown("---")
                with st.spinner("Predicting..."):
                    status_code, response = get_predict(image_bytes, model_name)
                    if status_code != 200:
                        st.error("Error fetching data!", icon="❌")
                        st.stop()
                    predictions = response.results[0]

                st.toast(f"Status code (predictions): {status_code}", icon="✅")

                st.markdown(f"## Predictions {model_name}")
                for i, _ in enumerate(predictions.labels):
                    st.progress(
                        predictions.scores[i],
                        text=f"{predictions.labels[i]} - {predictions.scores[i]}",
                    )

                # blog route
                st.markdown("---")
                with st.spinner("Fetching blog..."):
                    status_code, response = get_blog(predictions.labels[0])
                    if status_code != 200:
                        st.error("Error fetching data!", icon="❌")
                        st.stop()
                    blog = response.results

                st.toast(f"Status code (predictions): {status_code}", icon="✅")
                st.markdown(f"## {blog.real_name} (*{blog.binomial_name}*)")
                _, colb, _ = st.columns(3)
                with colb:
                    st.image(blog.image, use_column_width=True)
                st.markdown(blog.description)
                st.markdown("---")


if __name__ == "__main__":
    main()
