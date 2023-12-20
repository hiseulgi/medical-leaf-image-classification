import sys

import rootutils

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(ROOT / "src")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import blog_router, predictions_router
from src.api.server.uvicorn import UvicornServer
from src.api.utils.logger import get_logger

log = get_logger()


def main():
    app = FastAPI(
        title="Medical Leaf Image Classification API",
        version="0.1.0",
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predictions_router.router, prefix="/v1")
    app.include_router(blog_router.router, prefix="/v1")

    server = UvicornServer(
        app=app,
        host="0.0.0.0",
        port=6969,
    )
    server.run()


if __name__ == "__main__":
    main()
