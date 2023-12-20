import json
from typing import Dict, Tuple

import rootutils

from src.api.schema.blog_schema import BlogSchema
from src.api.utils.logger import get_logger

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

log = get_logger()


class BlogCore:
    def __init__(self):
        self._setup()

    def _setup(self):
        """Setup Blog Core"""

        # open blog article from json
        with open(ROOT / "src" / "api" / "static" / "leaf_information.json", "r") as f:
            self.blog: Dict[str, BlogSchema] = json.load(f)

    def get_blog(self, class_name: str) -> Tuple[bool, BlogSchema]:
        """Get Blog Article

        Args:
            class_name (str): Class Name of Predicted Image

        Returns:
            Tuple[bool, BlogSchema]: Tuple of boolean and Blog Schema. Boolean means whether the class name is found or not.
        """

        for _, value in self.blog.items():
            if class_name == value["class_name"]:
                log.info(f"Found blog article for {class_name}")
                return True, BlogSchema(**value)

        log.info(f"Blog article for {class_name} not found")
        return False, None
