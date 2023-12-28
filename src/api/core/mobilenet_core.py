import json
from typing import List, Union

import cv2
import numpy as np
import rootutils

from src.api.core.onnx_core import OnnxCore
from src.api.schema.predictions_schema import PredictionsResultSchema
from src.api.utils.logger import get_logger

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

log = get_logger()


class MobilenetCore(OnnxCore):
    """Mobilenet Core runtime engine module"""

    def __init__(
        self,
        engine_path: str = str(ROOT / "src/api/static/model/mobilenetv3_best.onnx"),
        class_path: str = str(ROOT / "src/api/static/class_mapping.json"),
        provider: str = "cpu",
    ) -> None:
        """
        Initialize Mobilenet Core runtime engine module.

        Args:
            engine_path (str): Path to ONNX runtime engine file.
            class_path (str): Path to class mapping json file.
            provider (str): Provider for ONNX runtime engine.
        """
        super().__init__(engine_path, provider)
        self.class_path = class_path
        self._open_class_mapping()

    def _open_class_mapping(self) -> None:
        """Open class mapping json file."""
        with open(self.class_path, "r") as f:
            self.class_mapping = json.load(f)

    def predict(
        self, imgs: Union[np.ndarray, List[np.ndarray]]
    ) -> List[PredictionsResultSchema]:
        """
        Classify image(s) (batch) and return top 5 predictions.

        Args:
            imgs (np.ndarray): Input image.

        Returns:
            List[PredictionsResultSchema]: List of predictions result, in size (Batch, 5).
        """
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        imgs = self.preprocess_imgs(imgs)
        outputs = self.engine.run(None, {self.metadata[0].input_name: imgs})
        outputs = self.postprocess_imgs(outputs)
        return outputs

    def preprocess_imgs(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Preprocess image(s) (batch) like resize and normalize.

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): Image(s) to preprocess.
            normalize (bool, optional): Whether to normalize image(s). Defaults to True.

        Returns:
            np.ndarray: Preprocessed image(s) in size (B, C, H, W).
        """
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        # resize images
        dst_h, dst_w = self.img_shape
        resized_imgs = np.zeros((len(imgs), dst_h, dst_w, 3), dtype=np.float32)

        for i, img in enumerate(imgs):
            # resize img to 224x224 (according to model input)
            img = cv2.resize(img, dsize=(dst_h, dst_w), interpolation=cv2.INTER_CUBIC)
            resized_imgs[i] = img

        # normalize images
        # resized_imgs = resized_imgs.transpose(0, 3, 1, 2)
        resized_imgs /= 255.0 if normalize else 1.0

        return resized_imgs

    def postprocess_imgs(
        self, outputs: List[np.ndarray]
    ) -> List[PredictionsResultSchema]:
        """
        Postprocess model output(s) into top 5 predictions probability.

        Args:
            outputs (List[np.ndarray]): Model output(s) (batch), in size (Batch, Class).

        Returns:
            List[PredictionsResultSchema]: List of predictions result, in size (Batch, 5).
        """
        results: List[PredictionsResultSchema] = []
        for output in outputs:
            softmax_output = self.softmax(output[0])

            labels = []
            scores = []
            top_5_pred = np.argsort(softmax_output)[::-1][:5]

            for i in top_5_pred:
                labels.append(self.class_mapping[str(i)])
                scores.append(float(softmax_output[i]))

            results.append(PredictionsResultSchema(labels=labels, scores=scores))

            log.info(f"Predictions: {results}")

        return results

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each sets of scores in x.

        Args:
            x (np.ndarray): Input logits.

        Returns:
            np.ndarray: Softmax calculation result.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
