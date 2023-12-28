from io import BytesIO

import numpy as np
from PIL import Image


async def preprocess_img_bytes(img_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes."""
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    # if PNG, convert to RGB
    if img.shape[-1] == 4:
        img = img[..., :3]

    return img
