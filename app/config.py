import os


PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "8"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}
