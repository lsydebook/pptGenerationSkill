# pptGenerationSkill
这是一个为了应付研究生周报而生的skill，正在研究中......

## File parsing service (Kohaku-style)
### Run

```bash
uv run main.py
```

Or with uvicorn directly:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### API

Parse uploaded file and free-form text, each will be parsed independently if both are provided.

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "file=@/path/to/weekly.md" \
	-F "note=optional note"
```

Parse free-form text.

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "text=free form notes" \
	-F "note=optional note"
```

Upload a single image (currently no parsing).

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "file=@/path/to/figure1.png" \
	-F "note=optional note"
```

Upload a document and text together (both are parsed and merged in one response).

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "file=@/path/to/weekly.md" \
	-F "text=free form notes" \
	-F "note=optional note"
```

### Notes

- Supported extensions: .pdf, .md, .markdown, .txt
- Each parsed document includes `text`, `metadata`, and `assets`.
- Parsing uses Kohaku-style document segmentation (section → paragraph → sentence).
- Image uploads are accepted but not parsed yet.
- Concurrency is controlled by `PARSE_CONCURRENCY` (default: 8).
- Max upload size is controlled by `MAX_FILE_SIZE_MB` (default: 50).

## Offline document parsing (Kohaku-style)

This project includes a structured parser that produces `DocumentPayload`
objects (document → section → paragraph → sentence) with unique node IDs like
`docA:sec1:p2:s3`. It supports PDF, Markdown, and plain text inputs. Use
`DocumentIndexer` in `app.kohakurag.indexer` to build the hierarchy.
