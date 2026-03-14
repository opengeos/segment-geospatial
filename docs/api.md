# REST API

segment-geospatial includes a built-in REST API powered by [FastAPI](https://fastapi.tiangolo.com/) that allows you to run image segmentation over HTTP. This is useful for integrating segmentation into web applications, pipelines, and non-Python clients.

## Installation

Install the API dependencies with the `api` extra:

```bash
pip install "segment-geospatial[api]"
```

To also install a specific SAM model backend, combine extras:

```bash
pip install "segment-geospatial[api,samgeo3]"
```

## Starting the Server

Use the `samgeo-api` command:

```bash
samgeo-api
```

Options:

```bash
samgeo-api --host 0.0.0.0 --port 8000        # Custom host/port
samgeo-api --preload sam2:sam2-hiera-large     # Preload a model at startup
samgeo-api --reload                            # Auto-reload for development
```

Alternatively, use `uvicorn` directly:

```bash
uvicorn samgeo.api:app --host 0.0.0.0 --port 8000
```

Once running, interactive API docs (Swagger UI) are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Endpoints

### Health Check

```
GET /health
```

Returns the server status and version.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "1.2.3"}
```

### List Models

```
GET /models
```

Returns available model versions/IDs and which models are currently loaded in memory.

```bash
curl http://localhost:8000/models
```

### Clear Models

```
DELETE /models
```

Clears the model cache and frees GPU memory.

```bash
curl -X DELETE http://localhost:8000/models
```

### Automatic Segmentation

```
POST /segment/automatic
```

Runs automatic mask generation on an uploaded image. Supports SAM, SAM2, and SAM3.

**Parameters (multipart form):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Image file (TIFF, PNG, JPEG) |
| `model_version` | string | `sam2` | One of `sam`, `sam2`, `sam3` |
| `model_id` | string | auto | Model identifier (e.g., `sam2-hiera-large`) |
| `output_format` | string | `geojson` | One of `geojson`, `geotiff`, `png` |
| `foreground` | bool | `true` | Extract foreground objects only |
| `unique` | bool | `true` | Assign unique ID to each object |
| `min_size` | int | `0` | Minimum mask size in pixels |
| `max_size` | int | none | Maximum mask size in pixels |
| `points_per_side` | int | `32` | Points sampled per side (SAM/SAM2) |
| `pred_iou_thresh` | float | `0.8` | IoU threshold for filtering |
| `stability_score_thresh` | float | `0.95` | Stability score threshold |

**Example:**

```bash
curl -X POST http://localhost:8000/segment/automatic \
  -F "file=@image.tif" \
  -F "model_version=sam2" \
  -F "output_format=geojson"
```

### Prompt-based Segmentation

```
POST /segment/predict
```

Runs segmentation with point or bounding box prompts. Supports SAM and SAM2.

**Parameters (multipart form):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Image file (TIFF, PNG, JPEG) |
| `model_version` | string | `sam2` | One of `sam`, `sam2` |
| `model_id` | string | auto | Model identifier |
| `output_format` | string | `geojson` | One of `geojson`, `geotiff`, `png` |
| `point_coords` | string | none | JSON array of `[[x, y], ...]` |
| `point_labels` | string | none | JSON array of `[1, 0, ...]` (1=foreground, 0=background) |
| `boxes` | string | none | JSON array of `[[xmin, ymin, xmax, ymax], ...]` |
| `point_crs` | string | none | CRS string (e.g., `EPSG:4326`) |
| `multimask_output` | bool | `false` | Return multiple masks per prompt |

**Example with point prompts:**

```bash
curl -X POST http://localhost:8000/segment/predict \
  -F "file=@image.tif" \
  -F "point_coords=[[100, 200]]" \
  -F "point_labels=[1]" \
  -F "output_format=geojson"
```

**Example with box prompts:**

```bash
curl -X POST http://localhost:8000/segment/predict \
  -F "file=@image.tif" \
  -F "boxes=[[10, 20, 300, 400]]" \
  -F "output_format=geotiff"
```

### Text-prompt Segmentation

```
POST /segment/text
```

Runs text-prompt segmentation using SAM3.

**Parameters (multipart form):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Image file (TIFF, PNG, JPEG) |
| `prompt` | string | required | Text description (e.g., `building`, `tree`) |
| `model_id` | string | auto | SAM3 model identifier |
| `backend` | string | `meta` | One of `meta`, `transformers` |
| `output_format` | string | `geojson` | One of `geojson`, `geotiff`, `png` |
| `confidence_threshold` | float | `0.5` | Detection confidence threshold |
| `min_size` | int | `0` | Minimum mask size in pixels |
| `max_size` | int | none | Maximum mask size in pixels |

**Example:**

```bash
curl -X POST http://localhost:8000/segment/text \
  -F "file=@image.tif" \
  -F "prompt=building" \
  -F "output_format=geojson"
```

## Caching

The API automatically caches models and image encodings for better performance:

- **Model cache**: Models are loaded once and reused across requests. Use `DELETE /models` to free GPU memory.
- **Image cache**: When the same image is sent multiple times (e.g., with different prompts), the expensive image encoding step is skipped. This makes subsequent requests significantly faster.

Example timing with a 13 MB GeoTIFF:

| Request | Description | Time |
|---------|-------------|------|
| 1st | Model load + image encoding | ~7s |
| 2nd | Same image, different prompt | ~0.4s |
| 3rd | Same image, another prompt | ~0.2s |

## Python Client Example

```python
import requests

url = "http://localhost:8000/segment/text"

with open("image.tif", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("image.tif", f, "image/tiff")},
        data={"prompt": "building", "output_format": "geojson"},
    )

geojson = response.json()
print(f"Found {len(geojson['features'])} features")
```

## API Reference

::: samgeo.api
