import os


PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "8"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
IMAGE_UPLOAD_DIR = os.getenv("IMAGE_UPLOAD_DIR", "uploaded_images")
SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}
