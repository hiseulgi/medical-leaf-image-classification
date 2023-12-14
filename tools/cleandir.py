"""Clean dataset directory in `data`"""

import os
from pathlib import Path
from typing import Dict, List

import rootutils

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


def main() -> None:
    data_dir = ROOT / "data" / "Segmented Medicinal Leaf Images"
    folder_list = list(data_dir.glob("*"))
    for folder in folder_list:
        new_name = str(folder).split("/")[-1]
        new_name = (
            new_name.split("(")[-1]
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(")", "")
        )
        os.rename(folder, data_dir / new_name)
    os.rename(
        ROOT / "data" / "Segmented Medicinal Leaf Images",
        ROOT / "data" / "cleaned_dataset",
    )


if __name__ == "__main__":
    main()
