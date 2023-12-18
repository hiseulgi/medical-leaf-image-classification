import json
import pickle
from io import BytesIO
from typing import Dict

import numpy as np
import rootutils
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import median, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import erosion, remove_small_holes, remove_small_objects
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from src.api.utils.logger import get_logger

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

log = get_logger()


class KnnCore:
    def __init__(self):
        self.setup()

    def setup(self):
        """Setup KNN Core"""
        # load scaler
        with open(ROOT / "temp" / "scaler.pkl", "rb") as f:
            self.scaler: MinMaxScaler = pickle.load(f)

        # load model
        with open(ROOT / "temp" / "knn_model.pkl", "rb") as f:
            self.model: KNeighborsClassifier = pickle.load(f)

        # load class mapping from json
        with open(ROOT / "temp" / "class_mapping.json", "r") as f:
            self.class_mapping: Dict[str, str] = json.load(f)

    async def preprocess_img_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes."""
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)
        # if PNG, convert to RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img

    async def preprocess_img_knn(self, img_np: np.ndarray) -> np.ndarray:
        """Preprocess image for KNN."""
        # read as grayscale
        img_gray = rgb2gray(img_np)

        # median filter
        img_filtered = median(img_gray)

        # thresholding
        thresh = threshold_otsu(img_gray)
        img_binary = img_filtered > thresh
        img_binary = remove_small_objects(img_binary, 100)
        img_binary = remove_small_holes(img_binary, 100)
        img_binary = erosion(img_binary)

        # invert image
        inverted_img = np.invert(img_binary)

        return inverted_img

    async def feature_extraction_knn(self, img_np: np.ndarray) -> np.ndarray:
        """Extract features for KNN."""
        FEATURES_PROPERTIES = [
            "area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
        ]

        label_np = label(img_np)
        props = regionprops_table(label_np, img_np, properties=FEATURES_PROPERTIES)
        props = {k: v[0] for k, v in props.items()}
        props_np = np.array(list(props.values())).reshape(1, -1)
        return props_np
