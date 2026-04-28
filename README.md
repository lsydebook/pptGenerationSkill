# pptGenerationSkill
这是一个为了应付研究生周报而生的skill，正在研究中......

## File parsing service (LlamaIndex)

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API

Parse one uploaded file and return extracted documents.

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "file=@/path/to/weekly.md" \
	-F "note=optional note for image-only uploads"
```

### Notes

- Supported extensions: .md, .markdown, .txt, .pdf, .docx, .png, .jpg, .jpeg, .webp, .gif
- Image files return an empty text by default unless `note` is provided.
- Concurrency is controlled by `PARSE_CONCURRENCY` (default: 8).
- Max upload size is controlled by `MAX_FILE_SIZE_MB` (default: 50).
