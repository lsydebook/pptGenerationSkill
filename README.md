# pptGenerationSkill
这是一个为了应付研究生周报而生的skill，正在研究中......

## File parsing service (LlamaIndex)
### Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API

Parse one uploaded file and return extracted documents.

```bash
curl -X POST "http://localhost:8000/v1/parse" \
	-F "file=@/path/to/weekly.md" \
	-F "note=optional note for image-only uploads" \
	-F "image_output_dir=path/to/output"
```

### Notes

- Supported extensions (per LlamaIndex SimpleDirectoryReader):
	- .csv, .docx, .epub, .hwp, .ipynb, .mbox, .md, .mp3, .mp4,
		.pdf, .ppt, .pptm, .pptx, .txt, .jpeg, .jpg, .png
- Each parsed document includes `text`, `metadata`, and `assets`.
- Parsing is handled via LlamaIndex `SimpleDirectoryReader` for supported document types.
- `assets` contains paths to saved images plus mime type and sha256.
- Use `IMAGE_OUTPUT_DIR` to set the default output folder (default: `parsed_images`).
- Set `INCLUDE_IMAGE_BASE64=true` to include base64 in assets (default: false).
- Extra dependencies by format:
	- .pdf: `pypdf`
	- .docx: `docx2txt`
	- .ppt/.pptx/.pptm: `python-pptx`
	- .epub: `EbookLib`, `html2text`
	- .mbox: `beautifulsoup4`
	- .ipynb: `nbconvert`
	- .hwp: `olefile`
	- .jpeg/.jpg/.png: `pillow`
	- .mp3/.mp4: `openai-whisper`, `pydub` (requires ffmpeg installed)
- Concurrency is controlled by `PARSE_CONCURRENCY` (default: 8).
- Max upload size is controlled by `MAX_FILE_SIZE_MB` (default: 50).
