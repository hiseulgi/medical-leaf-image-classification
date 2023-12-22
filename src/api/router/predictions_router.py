from typing import List

import numpy as np
import rootutils
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.core.knn_core import KnnCore
from src.api.schema.api_schema import PredictionResponseSchema, PredictionsRequestSchema
from src.api.schema.predictions_schema import PredictionsResultSchema
from src.api.utils.logger import get_logger

log = get_logger()

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


# initialize knn core
knn_core = KnnCore()

# initialize router
router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)


def allowed_file_types(filename: str):
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    ext = filename.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS


@router.post(
    "/knn",
    tags=["predictions"],
    summary="Classify medical leaf image",
    response_model=PredictionResponseSchema,
)
async def knn_predictions(
    request: Request, form: PredictionsRequestSchema = Depends()
) -> PredictionResponseSchema:
    """KNN Predictions from Raw Medical Leaf Image"""

    # Validate file type
    if not allowed_file_types(form.image.filename):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid file type. Only JPG, JPEG, and PNG files are allowed.",
        )
    log.info(f"Processing image: {form.image.filename}")

    predictions: List[PredictionsResultSchema] = await knn_core.predict(
        form.image.file.read()
    )

    response = PredictionResponseSchema(
        status="success", message="Image processed successfully.", results=predictions
    )

    log.info(f"Image processed successfully.")

    return response
