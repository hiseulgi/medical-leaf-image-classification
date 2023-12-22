import rootutils
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.core.blog_core import BlogCore
from src.api.schema.api_schema import BlogRequestSchema, BlogResponseSchema
from src.api.utils.logger import get_logger

log = get_logger()

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


# initialize blog core
blog_core = BlogCore()

# initialize router
router = APIRouter(
    tags=["blog"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/blog",
    tags=["blog"],
    summary="Get single blog article by class name",
    response_model=BlogResponseSchema,
)
async def get_blog(
    request: BlogRequestSchema = Depends(),
) -> BlogResponseSchema:
    """
    Get single blog article by class name
    """

    # get class name
    class_name = request.class_name

    # get blog article
    is_found, blog = blog_core.get_blog(class_name=class_name)

    if not is_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blog article for {class_name} not found",
        )

    return BlogResponseSchema(
        status="success",
        message="OK",
        results=blog,
    )
