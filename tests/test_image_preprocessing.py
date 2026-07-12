import base64
from io import BytesIO

from PIL import Image

from app.services.image_preprocessing import decode_base64_image
from app.utils.errors import AiServiceError


def test_valid_base64_image_decodes() -> None:
    image = Image.new("RGB", (2, 2), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    decoded = decode_base64_image(encoded)

    assert decoded.mode == "RGB"
    assert decoded.size == (2, 2)


def test_invalid_base64_returns_invalid_image_error() -> None:
    try:
        decode_base64_image("not-base64")
    except AiServiceError as exception:
        assert exception.error_code == "INVALID_IMAGE_BASE64"
    else:
        raise AssertionError("Expected INVALID_IMAGE_BASE64")
