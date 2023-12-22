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

    label: str = Field(..., description="Predicted Label", example="curry")
    score: float = Field(..., description="Predicted Score", example=0.6)

    @validator("score", pre=True)
    def score_to_float(cls, v: float) -> float:
        return round(v, 2)
