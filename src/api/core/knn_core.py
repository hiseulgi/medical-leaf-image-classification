import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import rootutils
from skimage.color import rgb2gray
from skimage.filters import median, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import erosion, remove_small_holes, remove_small_objects
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from src.api.schema.predictions_schema import PredictionsResultSchema
from src.api.utils.logger import get_logger

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

log = get_logger()


class KnnCore:
    def __init__(
        self,
        model_path: str = str(
            ROOT / "src" / "api" / "static" / "model" / "knn_model.pkl"
        ),
        scaler_path: str = str(
            ROOT / "src" / "api" / "static" / "model" / "scaler.pkl"
        ),
        class_mapping_path: str = str(
            ROOT / "src" / "api" / "static" / "class_mapping.json"
        ),
    ) -> None:
        """Initialize KNN Core"""

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.class_mapping_path = Path(class_mapping_path)
        self.setup()

    def setup(self) -> None:
        """Setup KNN Core"""
        # load scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler: MinMaxScaler = pickle.load(f)

        # load model
        with open(self.model_path, "rb") as f:
            self.model: KNeighborsClassifier = pickle.load(f)

        # load class mapping from json
        with open(self.class_mapping_path, "r") as f:
            self.class_mapping: Dict[str, str] = json.load(f)

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

    async def predict(self, img_np: np.ndarray) -> PredictionsResultSchema:
        """Predict using KNN model."""

        # preprocess image
        img_np = await self.preprocess_img_knn(img_np)

        # features extraction
        features = await self.feature_extraction_knn(img_np)

        # scale features
        features = self.scaler.transform(features)

        # predict
        predictions = self.model.predict_proba(features.reshape(1, -1))

        # get result
        labels = []
        scores = []
        top_5_pred = np.argsort(predictions, axis=1)[0, -5:][::-1]

        for i in top_5_pred:
            labels.append(self.class_mapping[str(i)])
            scores.append(float(predictions[0, i]))

        result = PredictionsResultSchema(labels=labels, scores=scores)

        log.info(f"Predictions: {result}")

        return result
