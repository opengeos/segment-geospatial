# Frequently Asked Questions (FAQ)

## Coordinate Reference Systems (CRS)

### Does samgeo automatically convert my imagery to EPSG:4326?

**No.** samgeo does **NOT** automatically reproject or convert your GeoTIFF files to EPSG:4326. Your imagery stays in its native coordinate system throughout the entire segmentation process.

**Key points:**
- When you load a GeoTIFF with `set_image()` or use `generate_masks_tiled()`, the image data is read directly without any CRS transformation
- The output masks inherit the same CRS as your input imagery
- The `reproject()` function in `samgeo.common` has EPSG:4326 as a default parameter, but it's **only used when you explicitly call that function** - it is NOT called automatically during segmentation

**Why might you see distortion?**
1. Your visualization tool is displaying the results in EPSG:4326 (like some web map libraries)
2. You're manually calling the `reproject()` function somewhere in your workflow
3. Your original imagery already has CRS-related issues

**Recommendation:**
- Verify your input TIF has the correct CRS metadata using `gdalinfo` or rasterio (`src.crs`)
- The output masks will preserve the same CRS as your input
- If you need to reproject for visualization, do it as a separate step AFTER segmentation:
  ```python
  from samgeo import common
  common.reproject("input_masks.tif", "output_epsg4326.tif", dst_crs="EPSG:4326")
  ```

### How can I check the CRS of my GeoTIFF?

Using GDAL:
```bash
gdalinfo your_file.tif
```

Using Python with rasterio:
```python
import rasterio
with rasterio.open("your_file.tif") as src:
    print(f"CRS: {src.crs}")
```

### My imagery looks distorted during segmentation

The segmentation algorithm works on pixel values, not geographic coordinates, so CRS should not affect segmentation quality. If you're seeing distortion:

1. **Check if it's a visualization issue:** The distortion might only appear when viewing results in a different CRS
2. **Verify your input data:** Use `gdalinfo` to check if the CRS metadata is correct
3. **Try reprojecting to a more suitable CRS:** Some projections preserve shape better than others for specific regions (e.g., UTM for local areas)
