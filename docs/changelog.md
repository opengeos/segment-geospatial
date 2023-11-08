# Changelog

## v0.10.2 - Nov 7, 2023

**What's Changed**

-   Add JOSS paper by @giswqs in [#197](https://github.com/opengeos/segment-geospatial/pull/197)
-   Add notebook for using Maxar Open Data by @giswqs in [#198](https://github.com/opengeos/segment-geospatial/pull/198)
-   Add checkpoint to textsam.LangSAM() by @forestbat in [#204](https://github.com/opengeos/segment-geospatial/pull/204)
-   Add workshop notebook by @giswqs in [#209](https://github.com/opengeos/segment-geospatial/pull/209)

**New Contributors**

-   @forestbat made their first contribution in [#204](https://github.com/opengeos/segment-geospatial/pull/204)

**Full Changelog**: [v0.10.1...v0.10.2](https://github.com/opengeos/segment-geospatial/compare/v0.10.1...v0.10.2)

## v0.10.1 - Sep 1, 2023

**What's Changed**

-   Fix basemap issue by @giswqs in [#190](https://github.com/opengeos/segment-geospatial/pull/190)

**Full Changelog**: [v0.10.0...v0.10.1)](https://github.com/opengeos/segment-geospatial/compare/v0.10.0...v0.10.1)

## v0.10.0 - Aug 24, 2023

### What's Changed

-   Added fastsam module by @giswqs #167
-   Update optional dependencies by @giswqs in #68
-   Improve contributing guidelines by @giswqs in #169
-   [FIX] Added missing conversions from BGR to RGB by @lbferreira in #171
-   Address JOSS review comments by @giswqs in #175

### New Contributors

-   @lbferreira made their first contribution in #171

## v0.9.1 - Aug 14, 2023

**New Features**

-   Added support for HQ-SAM (#161)
-   Added HQ-SAM notebooks (#162)

## v0.9.0 - Aug 6, 2023

**New Features**

-   Added support for multiple input boxes (#159)

**Improvements**

-   UpdateD groundingdino installation (#147)
-   Updated README (#152)

## v0.8.5 - Jul 19, 2023

**Improvements**

-   Updated installation docs (#146)
-   Updated leafmap and localtileserver to dependencies (#146)
-   Added info about libgl1 dependency install on Linux systems (#141)
-   Fixed save_masks bug without source image (#139)

## v0.8.4 - Jul 5, 2023

**Improvements**

-   Fixed model download bug (#136)
-   Added legal notice (#133)
-   Fixed image source bug for show_anns (#131)
-   Improved exception handling for LangSAM GUI (#130)
-   Added to return pixel coordinates of masks (#129)
-   Added text_sam to docs (#123)
-   Fixed file deletion error on Windows (#122)
-   Fixed mask bug in text_sam/predict when the input is PIL image (#117)

## v0.8.3 - Jun 20, 2023

**New Features**

-   Added support for batch segmentation (#116)
-   Added swimming pools example (#106)

**Improvements**

-   Removed 'flag' and 'param' arguments (#112)
-   Used sorted function instead of if statements (#109)

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
