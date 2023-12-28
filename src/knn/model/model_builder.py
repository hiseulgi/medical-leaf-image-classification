import pickle

import numpy as np
import rootutils
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier

from src.knn.dataset.dataset_module import DatasetModule

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class KNNModule:
    def __init__(self, dataset: DatasetModule = None) -> None:
        self.model = None
        self.dataset = dataset

    def train(self) -> None:
        assert self.dataset is not None, "Dataset is not provided"

        # parameter tuning
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        parameter = {"n_neighbors": np.arange(2, 30, 1)}
        knn = KNeighborsClassifier()
        knn_cv = GridSearchCV(knn, param_grid=parameter, cv=kf, verbose=3)
        X_train, y_train = self.dataset.get_train_data()
        knn_cv.fit(X_train, y_train)
        print(f"Best score: {knn_cv.best_score_}")
        print(f"Best parameter: {knn_cv.best_params_}")

        # train model
        self.model = knn_cv.best_estimator_
        self.model.fit(X_train, y_train)

    def test(self) -> None:
        assert self.dataset is not None, "Dataset is not provided"
        assert self.model is not None, "Model is not trained"

        X_test, y_test = self.dataset.get_test_data()
        print(f"Test score: {self.model.score(X_test, y_test)}")

    def save_model(self) -> None:
        with open(ROOT / "temp" / "knn_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path: str) -> None:
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)


class LazyPredictModule:
    def __init__(self, dataset: DatasetModule = None) -> None:
        self.model = None
        self.dataset = dataset

    def train(self) -> None:
        assert self.dataset is not None, "Dataset is not provided"

        X_train, y_train = self.dataset.get_train_data()
        X_test, y_test = self.dataset.get_test_data()

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        print(models)


if __name__ == "__main__":
    # debug knn module
    dataset = DatasetModule(ROOT / "data" / "extracted_dataset.csv")
    knn = KNNModule(dataset)
    knn.train()
    knn.test()
    knn.save_model()
    knn.load_model(ROOT / "temp" / "knn_model.pkl")
    knn.test()

    # debug lazy predict module
    lazy_predict = LazyPredictModule(dataset)
    lazy_predict.train()
