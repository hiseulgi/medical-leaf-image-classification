"""Feature Extraction from dataset"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd
import rootutils
from skimage.color import rgb2gray
from skimage.filters import median, threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops_table
from skimage.morphology import erosion, remove_small_holes, remove_small_objects
from tqdm import tqdm

from src.extraction.data_module import DataModule

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class FeatureExtractor:
    def __init__(
        self,
        raw_dataset: Dict[str, List[str]],
    ) -> None:
        self.features_properties = [
            "area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
        ]
        self.raw_dataset = raw_dataset
        self.extracted_features = self.start()

    def start(self) -> None:
        """Extracts features from dataset"""
        extracted_dataset = self.raw_dataset.copy()

        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(self._extract_feature, self.raw_dataset["images"]),
                    total=len(self.raw_dataset["images"]),
                )
            )

        for prop in self.features_properties:
            extracted_dataset[prop] = [result[prop][0] for result in results]

        self._dump_features(extracted_dataset)

        return extracted_dataset

    def get_extracted_features(self) -> Dict[str, List[float]]:
        """Returns extracted features"""
        return self.extracted_features

    def _extract_feature(self, image_path: str) -> Dict[str, List[float]]:
        """Opens image from path and extracts features"""
        with open(image_path, "rb") as f:
            img = imread(f)
            preprocessed_image = self._preprocess_image(img)

        labels = label(preprocessed_image)
        props = regionprops_table(
            labels, preprocessed_image, properties=self.features_properties
        )

        return props

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocesses image"""
        # read as grayscale
        img = rgb2gray(img)

        # median filter
        img = median(img)

        # thresholding
        thresh = threshold_otsu(img)
        img = img > thresh
        img = remove_small_objects(img, 100)
        img = remove_small_holes(img, 100)
        img = erosion(img)

        # invert image
        inverted_img = np.invert(img)

        return inverted_img

    def _dump_features(self, extracted_dataset: Dict[str, List[any]]) -> None:
        """Dumps extracted features to csv"""
        df = pd.DataFrame(extracted_dataset)
        df.to_csv(ROOT / "data" / "extracted_dataset.csv", index=False)


if __name__ == "__main__":
    dataset_dir = ROOT / "data" / "cleaned_dataset"
    dataset_module = DataModule(dataset_dir).get_dataset()
    feature_extractor = FeatureExtractor(dataset_module)

    for key, value in feature_extractor.get_extracted_features().items():
        print(key, value[0])
