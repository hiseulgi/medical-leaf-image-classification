import uvicorn

from src.api.utils.logger import get_logger

log = get_logger()


class UvicornServer:
    """Uvicorn runner."""

    def __init__(
        self, app, host: str, port: int, workers: int = 1, log_level: str = "info"
    ):
        self.app = app
        self.host = host
        self.port = port
        self.workers = workers
        self.log_level = log_level

    def run(self):
        log.info(f"Starting uvicorn server on {self.host}:{self.port}...")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers,
            log_level=self.log_level,
        )
