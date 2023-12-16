import argparse
import os
import sys

import rootutils

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(ROOT / "src")

from src.dataset.dataset_module import DatasetModule
from src.extraction.data_module import DataModule
from src.extraction.feature_extractor import FeatureExtractor
from src.model.model_builder import KNNModule


def main() -> None:
    # check if extracted dataset exists
    if not os.path.exists(ROOT / "data" / "extracted_dataset.csv"):
        # extract dataset
        dataset_dir = ROOT / "data" / "cleaned_dataset"
        dataset_module = DataModule(dataset_dir).get_dataset()
        FeatureExtractor(dataset_module)

    # initialize dataset module
    dataset = DatasetModule(ROOT / "data" / "extracted_dataset.csv")

    # initialize knn module
    knn = KNNModule(dataset)

    # train and save knn model
    knn.train()
    knn.save_model()

    # load and test knn model
    knn.load_model(ROOT / "temp" / "knn_model.pkl")
    knn.test()


if __name__ == "__main__":
    main()
