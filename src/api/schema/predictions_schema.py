import rootutils
from fastapi import File, UploadFile
from pydantic import BaseModel, Field, validator

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class PredictionsRequestSchema(BaseModel):
    image: UploadFile = File(..., description="Raw medical leaf image")


class KnnResponseSchema(BaseModel):
    """KNN Predictions Response Schema"""

    label: str = Field(..., description="Predicted Label", example="curry")
    score: float = Field(..., description="Predicted Score", example=0.6)

    @validator("score", pre=True)
    def score_to_float(cls, v: float) -> float:
        return round(v, 2)
