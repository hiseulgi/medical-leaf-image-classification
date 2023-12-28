"""Exctracted Dataset Module"""

import json
import pickle
from typing import Tuple, Union

import numpy as np
import pandas as pd
import rootutils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class DatasetModule:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.dataset = self._read_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = self._split_data()
        self.scaler = None
        self.label_encoder = None
        self._preprocess_data()

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns train data"""
        return self.x_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns test data"""
        return self.x_test, self.y_test

    def _read_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        return df

    def _split_data(
        self, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = self.dataset.iloc[:, 2:]
        label = self.dataset.iloc[:, 1:2]
        label = self._encode_labels(label)
        X_train, X_test, y_train, y_test = train_test_split(
            data, label, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def _preprocess_data(self) -> None:
        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        # dump scaler for inference
        self._dump_scaler()

    def _dump_scaler(self) -> None:
        with open(ROOT / "temp" / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

    def _encode_labels(self, labels: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        class_mapping = dict(
            zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_)
        )

        # dump label encoder for inference
        with open(ROOT / "temp" / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f)

        return encoded_labels


if __name__ == "__main__":
    dataset_module = DatasetModule(
        dataset_path=str(ROOT / "data" / "extracted_dataset.csv")
    )
    print(dataset_module)

    X_train, y_train = dataset_module.get_train_data()
    X_test, y_test = dataset_module.get_test_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
