from typing import List

import rootutils
from fastapi import File, UploadFile
from pydantic import BaseModel, Field

from src.api.schema.blog_schema import BlogSchema
from src.api.schema.predictions_schema import PredictionsResultSchema

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class BaseApiResponseSchema(BaseModel):
    """Base API Response Schema"""

    status: str = Field(..., description="API Response Status", example="success")
    message: str = Field(..., description="API Response Message", example="OK")


class PredictionsRequestSchema(BaseModel):
    """Predictions Request Schema"""

    image: UploadFile = File(..., description="Raw medical leaf image")


class PredictionResponseSchema(BaseApiResponseSchema):
    """Prediction Response Schema"""

    results: List[PredictionsResultSchema] = Field(
        ..., description="List of Predictions Result"
    )


class BlogRequestSchema(BaseModel):
    """Blog Request Schema"""

    class_name: str = Field(
        ..., description="Class Name of Predicted Image", example="curry"
    )


class BlogResponseSchema(BaseApiResponseSchema):
    """Blog Response Schema"""

    results: BlogSchema = Field(..., description="Blog of Predicted Image")
