import rootutils
from pydantic import BaseModel, Field

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class BlogSchema(BaseModel):
    """Blog Schema"""

    class_name: str = Field(
        ..., description="Class Name of Predicted Image", example="curry"
    )
    real_name: str = Field(
        ..., description="Real Name of Predicted Image", example="Curry Tree"
    )
    binomial_name: str = Field(
        ..., description="Binomial Name of Predicted Image", example="Murraya koenigii"
    )
    image: str = Field(
        ...,
        description="Example Image URL of Predicted Image",
        example="https://example.com/curry.jpg",
    )
    description: str = Field(
        ...,
        description="Description of Predicted Image",
        example="Curry is a leafy vegetable that is used in many dishes in Southeast Asia.",
    )
