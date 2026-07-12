import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.utils.errors import AiServiceError


def decode_base64_image(image_base64: str) -> Image.Image:
    if not image_base64 or not image_base64.strip():
        raise AiServiceError("FRAME_IMAGE_REQUIRED", "Frame image content is required for real local AI analysis.")

    try:
        if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exception:
        raise AiServiceError("INVALID_IMAGE_BASE64", "Frame image content is not valid base64.") from exception

    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exception:
        raise AiServiceError("INVALID_IMAGE_BASE64", "Frame image content is not a readable image.") from exception

    if image.width <= 0 or image.height <= 0:
        raise AiServiceError("INVALID_IMAGE_BASE64", "Frame image content is empty.")

    return image.convert("RGB")
