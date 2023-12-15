"""Data module for dataset building"""
from pathlib import Path
from typing import Dict, List

import rootutils

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class DataModule:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.dataset_dict = self._make_dataset()

    def get_dataset(self) -> Dict[str, List[str]]:
        """Returns dataset"""
        return self.dataset_dict

    def _make_dataset(self) -> Dict[str, List[str]]:
        """Creates dataset"""

        image_list = list(self.dataset_path.glob("**/*.jpg"))
        label_list = [str(image).split("/")[-2] for image in image_list]
        dataset_dict = {}
        dataset_dict["images"] = image_list
        dataset_dict["labels"] = label_list

        return dataset_dict


if __name__ == "__main__":
    dataset_dir = ROOT / "data" / "cleaned_dataset"
    dataset = DataModule(dataset_dir)
    print(dataset.get_dataset()["images"][0])
    print(dataset.get_dataset()["labels"][0])
