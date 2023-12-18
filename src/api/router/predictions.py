from typing import List

import numpy as np
import rootutils
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.core.knn_core import KnnCore
from src.api.schema.predictions_schema import (
    KnnResponseSchema,
    PredictionsRequestSchema,
)
from src.api.utils.logger import get_logger

log = get_logger()

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


def allowed_file_types(filename: str):
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    ext = filename.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS


# initialize knn core
knn_core = KnnCore()

# initialize router
router = APIRouter(
    prefix="/v1/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/knn",
    tags=["predictions"],
    summary="Classify medical leaf image",
    response_model=List[KnnResponseSchema],
)
async def knn_predictions(
    request: Request, form: PredictionsRequestSchema = Depends()
) -> List[KnnResponseSchema]:
    """KNN Predictions from Raw Medical Leaf Image"""

    # Validate file type
    if not allowed_file_types(form.image.filename):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid file type. Only JPG, JPEG, and PNG files are allowed.",
        )
    log.info(f"Processing image: {form.image.filename}")

    # convert image to numpy array
    img_np = await knn_core.preprocess_img_bytes(form.image.file.read())

    # preprocess image
    img_np = await knn_core.preprocess_img_knn(img_np)

    # features extraction
    features = await knn_core.feature_extraction_knn(img_np)

    # scale features
    features = knn_core.scaler.transform(features)

    # predict
    predictions = knn_core.model.predict_proba(features.reshape(1, -1))

    # get result
    result: List[KnnResponseSchema] = []
    top_5_pred = np.argsort(predictions, axis=1)[0, -5:][::-1]

    for idx in top_5_pred:
        result.append(
            KnnResponseSchema(
                label=knn_core.class_mapping[str(idx)],
                score=predictions[0, idx],
            )
        )

    log.info(f"Predictions: {result}")

    return result
