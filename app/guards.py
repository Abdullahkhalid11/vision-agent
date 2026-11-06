"""Upload validation: size, file signature, decodability, dimensions."""
import io

from PIL import Image

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 6000
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class RejectedUpload(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def check_upload(data: bytes) -> None:
    if len(data) > MAX_BYTES:
        raise RejectedUpload(413, "file exceeds 5 MB")
    # Check the file's own signature, not its extension.
    if not (data.startswith(JPEG_MAGIC) or data.startswith(PNG_MAGIC)):
        raise RejectedUpload(415, "only JPEG and PNG files are accepted")
    try:
        Image.open(io.BytesIO(data)).verify()  # catches truncated or corrupt files
        img = Image.open(io.BytesIO(data))     # verify() invalidates the object, so reopen
    except Exception:
        raise RejectedUpload(400, "file is not a decodable image")
    w, h = img.size
    if w > MAX_SIDE or h > MAX_SIDE:
        raise RejectedUpload(400, "image dimensions exceed 6000 px")