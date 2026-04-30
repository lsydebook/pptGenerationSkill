import os


PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "8"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
IMAGE_OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "parsed_images")
INCLUDE_IMAGE_BASE64 = os.getenv("INCLUDE_IMAGE_BASE64", "false").lower() in {
	"1",
	"true",
	"yes",
}

TEXT_EXTS = {
	".csv",
	".docx",
	".epub",
	".hwp",
	".ipynb",
	".mbox",
	".md",
	".mp3",
	".mp4",
	".pdf",
	".ppt",
	".pptm",
	".pptx",
	".txt",
}
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}
SUPPORTED_EXTS = TEXT_EXTS | IMAGE_EXTS
