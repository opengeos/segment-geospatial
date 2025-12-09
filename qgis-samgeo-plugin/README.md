# SamGeo QGIS Plugin

A QGIS plugin for remote sensing image segmentation using [SamGeo](https://samgeo.gishub.org), powered by Meta's Segment Anything Model ([SAM](https://github.com/facebookresearch/sam3)).

## Features

- **Multiple SAM versions**: Support for SamGeo (SAM1), SamGeo2 (SAM2), and SamGeo3 (SAM3)
- **Text-based segmentation**: Describe what you want to segment (e.g., "building", "tree", "road")
- **Interactive point prompts**: Click on the map to add foreground/background points
- **Box prompts**: Draw rectangles to segment specific regions
- **GeoTIFF support**: Works with georeferenced images, preserving spatial reference
- **Multiple output formats**: Save as vector (GeoPackage, Shapefile) or raster (GeoTIFF)

![](https://github.com/user-attachments/assets/21805e83-15a7-4619-92f4-391b90315eff)

## Requirements

- QGIS 3.22 or later
- Python 3.10+
- SamGeo package

## Installation

### 1. Install SamGeo

First, install SamGeo in your QGIS Python environment. It is recommended to create a new conda environment **and** install QGIS and SamGeo:

```bash
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge qgis segment-geospatial
```

Some SamGeo dependencies are only available on PyPI. Run the following command to install all dependencies:

```bash
pip install -U "segment-geospatial[samgeo3]"
```


### 2. Install the Plugin

#### Option A: Using the install script

```bash
# Clone the repository
git clone https://github.com/opengeos/qgis-samgeo-plugin.git
cd qgis-samgeo-plugin
conda activate geo

# Install (Linux/macOS)
python install_plugin.py

# Or use the shell script
chmod +x install_plugin.sh
./install_plugin.sh
```


#### Option B: Manual installation

1. Find your QGIS plugins directory:
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins`
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins`

2. Copy or symlink this entire folder (qgis-samgeo-plugin) to the plugins directory and rename it to `samgeo_plugin`:
   ```bash
   ln -s /path/to/qgis-samgeo-plugin ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/samgeo_plugin
   ```

3. Restart QGIS

4. Enable the plugin: Go to **Plugins > Manage and Install Plugins**, find "SamGeo", and enable it.

## Usage

### Start QGIS

Run the following commands to start QGIS and use the SamGeo plugin:

```bash
conda activate geo
qgis
```

### Opening the Plugin

1. Click the SamGeo icon in the toolbar, or
2. Go to **Raster > SamGeo > SamGeo Segmentation**

### Workflow

#### 1. Load the Model

1. In the **Model** tab, select your preferred SAM version:
   - **SamGeo3 (SAM3)**: Latest version with text prompts support
   - **SamGeo2 (SAM2)**: Second generation SAM
   - **SamGeo (SAM1)**: Original SAM model
2. Choose the backend (meta or transformers)
3. Select the device (auto, cuda, or cpu)
4. Set the confidence threshold (default: 0.5)
5. Enable interactive mode if you want to use point/box prompts
6. Click **Load Model**

   ![](https://github.com/user-attachments/assets/f1e4e356-e460-4d94-83ec-1f596d65156e)

#### 2. Set an Image

- Select a raster layer from the dropdown and click **Set Image from Layer**, or
- Browse for an image file and click **Set Image from File**
- You can download a sample image from [here](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif)

   ![](https://github.com/user-attachments/assets/b7c6a430-c4c3-4359-855c-b198cdcf2c91)

#### 3. Segment the Image

##### Text-Based Segmentation (Text tab)
1. Enter a text prompt describing what to segment (e.g., "tree", "building")
2. Optionally set min/max size filters
3. Click **Segment by Text**

   ![](https://github.com/user-attachments/assets/37012722-e1aa-4abe-9b25-8955064cfd8d)

##### Point-Based Segmentation (Interactive tab)
1. Click **Add Foreground Points** and click on objects to include
2. Click **Add Background Points** and click on areas to exclude
3. Click **Segment by Points**
4. Right-click to finish adding points

   ![](https://github.com/user-attachments/assets/0eb1174d-5e22-4555-be7d-aec0714c147d)

##### Box-Based Segmentation (Interactive tab)
1. Click **Draw Box**
2. Draw a rectangle around the area to segment
3. Click **Segment by Box**

   ![](https://github.com/user-attachments/assets/5aa962ae-4ed2-4696-a609-c586006a2ed8)

##### Batch Point Segmentation (Batch tab)
Use this mode to process many points at once (e.g., building centroids):

1. Select a point vector layer from the dropdown, or browse for a vector file (GeoJSON, Shapefile, etc.)
2. Optionally specify the CRS if not auto-detected (e.g., "EPSG:4326")
3. Set min/max size filters if needed
4. Optionally specify an output raster file path
5. Click **Run Batch Segmentation**


   ![](https://github.com/user-attachments/assets/b7714361-2122-4953-9fe0-61f2e9ae9b1d)

Each point generates a separate mask, making this efficient for segmenting many individual objects.

#### 4. Save Results (Output tab)

1. Choose output format (vector or raster)
2. Set the output file path
3. Click **Save Masks**

   ![](https://github.com/user-attachments/assets/84d00598-56a2-42c3-8cea-bfbf739c4e65)

## Programmatic Usage

You can also use SamGeo programmatically in the QGIS Python console:

```python
from samgeo import SamGeo3

# Initialize the model
sam = SamGeo3(
    backend="meta",
    enable_inst_interactivity=True,
    confidence_threshold=0.5,
)

# Set an image (GeoTIFF)
sam.set_image("/path/to/your/image.tif")

# Text-based segmentation
sam.generate_masks("building")
sam.save_masks("buildings.tif", unique=True)

# Point-based segmentation
point_coords = [[500, 300], [600, 400]]  # pixel coordinates
point_labels = [1, 1]  # 1=foreground, 0=background
sam.generate_masks_by_points(point_coords, point_labels)
sam.save_masks("points_result.tif")

# Batch point segmentation (from vector file)
sam.generate_masks_by_points_patch(
    point_coords_batch="building_centroids.geojson",
    point_crs="EPSG:4326",
    output="building_masks.tif",
    unique=True,
)

# Box-based segmentation
boxes = [[100, 100, 500, 500]]  # [xmin, ymin, xmax, ymax] in pixels
sam.generate_masks_by_boxes(boxes)
sam.save_masks("box_result.tif")

# Convert raster mask to vector
from samgeo import common
common.raster_to_vector("buildings.tif", "buildings.gpkg")
```

## Troubleshooting

### Model fails to load

- Ensure segment-geospatial is installed with the appropriate SAM support
- Check that you have enough GPU memory (or switch to CPU)
- Try restarting QGIS

### Segmentation is slow

- Use a GPU if available (select "cuda" device)
- Reduce image resolution before processing
- Use the transformers backend which may be faster for some tasks

### No objects found

- Try a different text prompt
- Lower the confidence threshold
- Adjust min/max size filters
- Ensure the image is properly loaded

## License

This plugin is released under the MIT License.

## Citation

If you use this plugin in your research, please cite:

```bibtex
@article{wu2023samgeo,
  title     = "{samgeo: A Python package for segmenting geospatial data with the
               Segment Anything Model (SAM)}",
  author    = "Wu, Qiusheng and Osco, Lucas Prado",
  journal   = "Journal of Open Source Software",
  volume    =  8,
  number    =  89,
  pages     =  5663,
  year      =  2023,
  url       = "https://doi.org/10.21105/joss.05663",
  doi       = "10.21105/joss.05663",
  issn      = "2475-9066"
}
```

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/opengeos/qgis-samgeo-plugin) for guidelines.
