from pathlib import Path
from typing import List, Union

import onnxruntime as ort
import rootutils

from src.api.schema.onnx_schema import OnnxMetadataSchema
from src.api.utils.logger import get_logger

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

log = get_logger()


class OnnxCore:
    """Common ONNX runtime engine module."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """
        Initialize ONNX runtime common engine.

        Args:
            engine_path (str): Path to ONNX runtime engine file.
            provider (str): Provider for ONNX runtime engine.
        """
        self.engine_path = Path(engine_path)
        self.provider = provider
        self.provider = self.check_providers(provider)

    def setup(self) -> None:
        """Setup ONNX runtime engine."""
        log.info(f"Setup ONNX engine")
        self.engine = ort.InferenceSession(
            str(self.engine_path), providers=self.provider
        )
        self.metadata = self.get_metadata()

        # img_shape tergantung pada file onnx-nya (lihat di netron)
        self.img_shape = self.metadata[0].input_shape[1:3]

        log.info(f"ONNX engine is ready!")

    def get_metadata(self) -> List[OnnxMetadataSchema]:
        """
        Get model metadata.

        Returns:
            List[OnnxMetadataSchema]: List of model metadata.
        """
        inputs = self.engine.get_inputs()
        outputs = self.engine.get_outputs()

        result: List[OnnxMetadataSchema] = []
        for inp, out in zip(inputs, outputs):
            result.append(
                OnnxMetadataSchema(
                    input_name=inp.name,
                    input_shape=inp.shape,
                    output_name=out.name,
                    output_shape=out.shape,
                )
            )

        return result

    def check_providers(self, provider: Union[str, List]) -> List:
        """
        Check available providers. If provider is not available, use CPU instead.

        Args:
            provider (Union[str, List]): Provider for ONNX runtime engine.

        Returns:
            List: List of available providers.
        """
        assert provider in ["cpu", "gpu"], "Invalid provider"
        available_providers = ort.get_available_providers()
        log.debug(f"Available providers: {available_providers}")
        if provider == "cpu" and "OpenVINOExecutionProvider" in available_providers:
            provider = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        elif provider == "gpu" and "CUDAExecutionProvider" in available_providers:
            provider = ["CUDAExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        return provider
