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
| `output_format` | string | `geojson` | One of `geojson`, `geotiff`, `png`, `json`, `detections` |
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

Runs segmentation with point or bounding box prompts. Supports SAM, SAM2, and SAM3.

For SAM3 with bounding box prompts, the model finds **all similar objects** in the image (not just the object inside the box). Point prompts with SAM3 segment the specific object at the point location.

**Parameters (multipart form):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Image file (TIFF, PNG, JPEG) |
| `model_version` | string | `sam3` | One of `sam`, `sam2`, `sam3` |
| `model_id` | string | auto | Model identifier |
| `output_format` | string | `geojson` | One of `geojson`, `geotiff`, `png`, `json`, `detections` |
| `point_coords` | string | none | JSON array of `[[x, y], ...]` |
| `point_labels` | string | none | JSON array of `[1, 0, ...]` (1=foreground, 0=background) |
| `boxes` | string | none | JSON array of `[[xmin, ymin, xmax, ymax], ...]` |
| `point_crs` | string | none | CRS string (e.g., `EPSG:4326`) for point/box coordinates |
| `multimask_output` | bool | `false` | Return multiple masks per prompt |
| `min_size` | int | `0` | Minimum mask size in pixels |
| `max_size` | int | none | Maximum mask size in pixels |

**Example with point prompts:**

```bash
curl -X POST http://localhost:8000/segment/predict \
  -F "file=@image.tif" \
  -F "point_coords=[[100, 200]]" \
  -F "point_labels=[1]" \
  -F "output_format=geojson"
```

**Example with box prompts (finds all similar objects):**

```bash
curl -X POST http://localhost:8000/segment/predict \
  -F "file=@image.tif" \
  -F "boxes=[[10, 20, 300, 400]]" \
  -F "output_format=geojson"
```

**Example with JSON output (pixel-coordinate bounding boxes):**

```bash
curl -X POST http://localhost:8000/segment/predict \
  -F "file=@image.jpg" \
  -F "boxes=[[10, 20, 300, 400]]" \
  -F "output_format=json"
```

```json
{
  "image_width": 2647,
  "image_height": 1464,
  "num_detections": 12,
  "detections": [
    {"id": 1, "value": 1, "bbox": [50, 80, 200, 250], "width": 150, "height": 170},
    {"id": 2, "value": 2, "bbox": [310, 45, 480, 210], "width": 170, "height": 165}
  ]
}
```

**Example with detections output (geographic-coordinate bounding boxes):**

```bash
curl -X POST http://localhost:8000/segment/predict \
  -F "file=@image.tif" \
  -F "boxes=[[10, 20, 300, 400]]" \
  -F "output_format=detections"
```

```json
{
  "type": "FeatureCollection",
  "crs": "EPSG:3857",
  "num_detections": 12,
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-13609328.39, 4561446.23], [-13609284.55, 4561446.23], [-13609284.55, 4561389.77], [-13609328.39, 4561389.77], [-13609328.39, 4561446.23]]]
      },
      "properties": {"id": 1, "value": 1, "bbox_pixel": [50.0, 80.0, 200.0, 250.0]}
    }
  ]
}
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
| `output_format` | string | `geojson` | One of `geojson`, `geotiff`, `png`, `json`, `detections` |
| `confidence_threshold` | float | `0.5` | Detection confidence threshold |
| `min_size` | int | `0` | Minimum mask size in pixels |
| `max_size` | int | none | Maximum mask size in pixels |

**Example (GeoJSON mask polygons):**

```bash
curl -X POST http://localhost:8000/segment/text \
  -F "file=@image.tif" \
  -F "prompt=building" \
  -F "output_format=geojson"
```

**Example with JSON output (pixel-coordinate bounding boxes):**

```bash
curl -X POST http://localhost:8000/segment/text \
  -F "file=@image.jpg" \
  -F "prompt=building" \
  -F "output_format=json"
```

```json
{
  "image_width": 2647,
  "image_height": 1464,
  "num_detections": 46,
  "detections": [
    {"id": 1, "bbox": [2506, 134, 2653, 324], "width": 147, "height": 190, "score": 0.887},
    {"id": 2, "bbox": [1200, 450, 1380, 620], "width": 180, "height": 170, "score": 0.862}
  ]
}
```

**Example with detections output (geographic-coordinate bounding boxes):**

```bash
curl -X POST http://localhost:8000/segment/text \
  -F "file=@image.tif" \
  -F "prompt=building" \
  -F "output_format=detections"
```

```json
{
  "type": "FeatureCollection",
  "crs": "EPSG:3857",
  "num_detections": 46,
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-13609328.39, 4561446.23], [-13609284.55, 4561446.23], [-13609284.55, 4561389.77], [-13609328.39, 4561389.77], [-13609328.39, 4561446.23]]]
      },
      "properties": {"id": 1, "score": 0.887, "bbox_pixel": [2506.47, 134.43, 2653.27, 323.52]}
    }
  ]
}
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

# Get GeoJSON mask polygons
with open("image.tif", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("image.tif", f, "image/tiff")},
        data={"prompt": "building", "output_format": "geojson"},
    )

geojson = response.json()
print(f"Found {len(geojson['features'])} features")
```

```python
# Get bounding boxes in pixel coordinates (suitable for non-georeferenced images)
with open("image.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("image.jpg", f, "image/jpeg")},
        data={"prompt": "car", "output_format": "json"},
    )

result = response.json()
for det in result["detections"]:
    print(f"Object {det['id']}: bbox={det['bbox']}, score={det['score']:.3f}")
```

## API Reference

::: samgeo.api
