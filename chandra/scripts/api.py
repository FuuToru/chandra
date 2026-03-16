"""
FastAPI service wrapping Chandra OCR.

Endpoints:
  POST /ocr/markdown  — returns merged markdown
  POST /ocr/html      — returns merged HTML
  POST /ocr/chunks    — returns chunks JSON (Datalab/Marker format)
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from chandra.input import load_file
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.scripts.cli import build_chunks_json

# ── Singleton model holder ──────────────────────────────────────────
_model: Optional[InferenceManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the InferenceManager once at startup."""
    global _model
    _model = InferenceManager(method="vllm")
    yield
    _model = None


app = FastAPI(
    title="Chandra OCR API",
    description="OCR service that converts PDF / images to Markdown, HTML, or Chunks.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _run_ocr(
    file_bytes: bytes,
    file_name: str,
    page_range: Optional[str] = None,
    include_images: bool = True,
    include_headers_footers: bool = True,
):
    """Save upload to a temp file, run inference, return list of BatchOutputItem."""
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        config = {"page_range": page_range} if page_range else {}
        images = load_file(tmp.name, config)

    batch = [
        BatchInputItem(image=img, prompt_type="ocr_layout")
        for img in images
    ]

    results = _model.generate(
        batch,
        include_images=include_images,
        include_headers_footers=include_headers_footers,
    )
    return results


# ── Endpoints ───────────────────────────────────────────────────────

@app.post("/ocr/markdown")
async def ocr_markdown(
    file: UploadFile = File(...),
    page_range: Optional[str] = Query(None, description="Page range, e.g. '0-2,5'"),
    include_images: bool = Query(True),
    include_headers_footers: bool = Query(True),
):
    """Return merged Markdown for all pages."""
    file_bytes = await file.read()
    results = _run_ocr(
        file_bytes, file.filename,
        page_range=page_range,
        include_images=include_images,
        include_headers_footers=include_headers_footers,
    )

    merged_md = "".join(r.markdown for r in results)
    return JSONResponse({
        "success": True,
        "file_name": file.filename,
        "num_pages": len(results),
        "result": merged_md,
    })


@app.post("/ocr/html")
async def ocr_html(
    file: UploadFile = File(...),
    page_range: Optional[str] = Query(None, description="Page range, e.g. '0-2,5'"),
    include_images: bool = Query(True),
    include_headers_footers: bool = Query(True),
):
    """Return merged HTML for all pages."""
    file_bytes = await file.read()
    results = _run_ocr(
        file_bytes, file.filename,
        page_range=page_range,
        include_images=include_images,
        include_headers_footers=include_headers_footers,
    )

    merged_html = "".join(r.html for r in results)
    return JSONResponse({
        "success": True,
        "file_name": file.filename,
        "num_pages": len(results),
        "result": merged_html,
    })


@app.post("/ocr/chunks")
async def ocr_chunks(
    file: UploadFile = File(...),
    page_range: Optional[str] = Query(None, description="Page range, e.g. '0-2,5'"),
    include_images: bool = Query(True),
    include_headers_footers: bool = Query(True),
):
    """Return chunks JSON (Datalab/Marker compatible format)."""
    file_bytes = await file.read()
    results = _run_ocr(
        file_bytes, file.filename,
        page_range=page_range,
        include_images=include_images,
        include_headers_footers=include_headers_footers,
    )

    chunks_data = build_chunks_json(file.filename, results)
    return JSONResponse(chunks_data)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}
