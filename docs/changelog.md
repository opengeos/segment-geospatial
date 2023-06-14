# Changelog

## v0.8.2 - Jun 14, 2023

**New Features**

-   Added regularized option for vector output (#104)
-   Added text prompt GUI (#80)

**Improvements**

-   Added more deep learning resources (#90)
-   Use the force_filename parameter with hf_hub_download() (#93)
-   Fixed typo (#94)

## v0.8.1 - May 24, 2023

**Improvements**

-   Added huggingface_hub and remove onnx (#87)
-   Added more demos to docs (#82)

## v0.8.0 - May 24, 2023

**New Features**

-   Added support for using text prompts with SAM (#73)
-   Added text prompt GUI (#80)

**Improvements**

-   Improved text prompt notebook (#79)
-   Fixed notebook typos (#78)
-   Added ArcGIS tutorial to docs (#72)

## v0.7.0 - May 20, 2023

**New Features**

-   Added unittest (#58)
-   Added JOSS paper draft (#61)
-   Added ArcGIS notebook example (#63)
-   Added text prompting segmentation (#65)
-   Added support for segmenting non-georeferenced imagery (#66)

**Improvements**

-   Added blend option for show_anns method (#59)
-   Updated ArcGIS installation instructions (#68, #70)

**Contributors**

@p-vdp @LucasOsco

## v0.6.2 - May 17, 2023

**Improvements**

-   Added jupyter-server-proxy to Dockerfile for supporting add_raster (#57)

## v0.6.1 - May 16, 2023

**New Features**

-   Added Dockerfile (#51)

## v0.6.0 - May 16, 2023

**New Features**

-   Added interactive GUI for creating foreground and background markers (#44)
-   Added support for custom projection bbox (#39)

**Improvements**

-   Fixed Colab Marker AwesomeIcon bug (#50)
-   Added info about using SAM with Desktop GIS (#48)
-   Use proper extension in the usage documentation (#43)

**Demos**

-   Interactive segmentation with input prompts

![](https://i.imgur.com/2Nyg9uW.gif)

-   Input prompts from existing files

![](https://i.imgur.com/Cb4ZaKY.gif)

## v0.5.0 - May 10, 2023

**New Features**

-   Added support for input prompts (#30)

**Improvements**

-   Fixed the batch processing bug (#29)

**Demos**

![](https://i.imgur.com/GV7Rzxt.gif)

## v0.4.0 - May 6, 2023

**New Features**

-   Added new methods to `SamGeo` class, including `show_masks`, `save_masks`, `show_anns`, making it much easier to save segmentation results in GeoTIFF and vector formats.
-   Added new functions to `common` module, including `array_to_image`, `show_image`, `download_file`, `overlay_images`, `blend_images`, and `update_package`
-   Added tow more notebooks, including [automatic_mask_generator](https://samgeo.gishub.org/examples/automatic_mask_generator/) and [satellite-predictor](https://samgeo.gishub.org/examples/satellite-predictor/)
-   Added `SamGeoPredictor` class

**Improvements**

-   Improved `SamGeo.generate()` method
-   Improved docstrings and API reference
-   Added demos to docs

**Demos**

-   Automatic mask generator

![](https://i.imgur.com/I1IhDgz.gif)

**Contributors**

@darrenwiens

## v0.3.0 - Apr 26, 2023

**New Features**

-   Added several new functions, including `get_basemaps`, `reproject`, `tiff_to_shp`, and `tiff_to_geojson`
-   Added hundereds of new basemaps through xyzservices

**Improvement**

-   Fixed `tiff_to_vector` crs bug #12
-   Add `crs` parameter to `tms_to_geotiff`

## v0.2.0 - Apr 21, 2023

**New Features**

-   Added notebook example
-   Added `SamGeo.generate` method
-   Added `SamGeo.tiff_to_vector` method

## v0.1.0 - Apr 19, 2023

**New Features**

-   Added `SamGeo` class
-   Added GitHub Actions
-   Added notebook example

## v0.0.1 - Apr 18, 2023

Initial release
