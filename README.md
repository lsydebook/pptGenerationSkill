# pptGenerationSkill
这是一个为了应付研究生周报而生的skill，正在研究中......

## File parsing service (Kohaku-style)
### Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API

Parse one uploaded file and return extracted documents.

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "file=@/path/to/weekly.md" \
	-F "note=optional note"

Parse free-form text with optional image uploads.

```bash
curl -X POST "http://localhost:8000/v1/parse_text_image" \
	-F "text=free form notes" \
	-F "images=@/path/to/figure1.png" \
	-F "images=@/path/to/figure2.jpg" \
	-F "note=optional note"
```

Parse a single image upload (same file field as `/v1/parse`).

```bash
curl -X POST "http://localhost:8000/v1/parse_image" \
	-F "file=@/path/to/figure1.png" \
	-F "note=optional note"
```
```

### Notes

- Supported extensions: .pdf, .md, .markdown, .txt
- Each parsed document includes `text`, `metadata`, and `assets`.
- Parsing uses Kohaku-style document segmentation (section → paragraph → sentence).
- Uploaded images are saved to `IMAGE_UPLOAD_DIR` (default: `uploaded_images`).
- Concurrency is controlled by `PARSE_CONCURRENCY` (default: 8).
- Max upload size is controlled by `MAX_FILE_SIZE_MB` (default: 50).

## Offline document parsing (Kohaku-style)

This project includes a structured parser that produces `DocumentPayload`
objects (document → section → paragraph → sentence) with unique node IDs like
`docA:sec1:p2:s3`. It supports PDF, Markdown, and plain text inputs. Use
`DocumentIndexer` in `app.kohakurag.indexer` to build the hierarchy.
