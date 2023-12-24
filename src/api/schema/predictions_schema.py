from typing import List

import rootutils
from pydantic import BaseModel, Field, validator

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class PredictionsResultSchema(BaseModel):
    """Predictions Result Schema"""

    labels: List[str] = Field(
        ..., description="Predicted Labels", example=["curry", "basil"]
    )
    scores: List[float] = Field(..., description="Predicted Scores", example=[0.6, 0.4])

    @validator("scores", pre=True)
    def round_scores(cls, v):
        return [round(score, 2) for score in v]
