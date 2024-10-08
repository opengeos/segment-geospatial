# Welcome to samgeo

[![image](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/opengeos/segment-geospatial/blob/main/docs/examples/satellite.ipynb)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/segment-geospatial/blob/main/docs/examples/satellite.ipynb)
[![image](https://img.shields.io/pypi/v/segment-geospatial.svg)](https://pypi.python.org/pypi/segment-geospatial)
[![image](https://img.shields.io/conda/vn/conda-forge/segment-geospatial.svg)](https://anaconda.org/conda-forge/segment-geospatial)
[![Docker Pulls](https://badgen.net/docker/pulls/giswqs/segment-geospatial?icon=docker&label=pulls)](https://hub.docker.com/r/giswqs/segment-geospatial)
[![PyPI Downloads](https://static.pepy.tech/badge/segment-geospatial)](https://pepy.tech/project/segment-geospatial)
[![Conda Recipe](https://img.shields.io/badge/recipe-segment--geospatial-green.svg)](https://github.com/conda-forge/segment-geospatial-feedstock)
[![Conda Downloads](https://anaconda.org/conda-forge/segment-geospatial/badges/downloads.svg)](https://anaconda.org/conda-forge/segment-geospatial)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05663/status.svg)](https://doi.org/10.21105/joss.05663)

[![logo](https://raw.githubusercontent.com/opengeos/segment-geospatial/main/docs/assets/logo_rect.png)](https://github.com/opengeos/segment-geospatial/blob/main/docs/assets/logo.png)

**A Python package for segmenting geospatial data with the Segment Anything Model (SAM)** 🗺️

## Introduction

The **segment-geospatial** package draws its inspiration from [segment-anything-eo](https://github.com/aliaksandr960/segment-anything-eo) repository authored by [Aliaksandr Hancharenka](https://github.com/aliaksandr960). To facilitate the use of the Segment Anything Model (SAM) for geospatial data, I have developed the [segment-anything-py](https://github.com/opengeos/segment-anything) and [segment-geospatial](https://github.com/opengeos/segment-geospatial) Python packages, which are now available on PyPI and conda-forge. My primary objective is to simplify the process of leveraging SAM for geospatial data analysis by enabling users to achieve this with minimal coding effort. I have adapted the source code of segment-geospatial from the [segment-anything-eo](https://github.com/aliaksandr960/segment-anything-eo) repository, and credit for its original version goes to Aliaksandr Hancharenka.

-   🆓 Free software: MIT license
-   📖 Documentation: <https://samgeo.gishub.org>

## Citations

-   Wu, Q., & Osco, L. (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). _Journal of Open Source Software_, 8(89), 5663. <https://doi.org/10.21105/joss.05663>
-   Osco, L. P., Wu, Q., de Lemos, E. L., Gonçalves, W. N., Ramos, A. P. M., Li, J., & Junior, J. M. (2023). The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot. _International Journal of Applied Earth Observation and Geoinformation_, 124, 103540. <https://doi.org/10.1016/j.jag.2023.103540>

## Features

-   Download map tiles from Tile Map Service (TMS) servers and create GeoTIFF files
-   Segment GeoTIFF files using the Segment Anything Model ([SAM](https://github.com/facebookresearch/segment-anything)) and [HQ-SAM](https://github.com/SysCV/sam-hq)
-   Segment remote sensing imagery with text prompts
-   Create foreground and background markers interactively
-   Load existing markers from vector datasets
-   Save segmentation results as common vector formats (GeoPackage, Shapefile, GeoJSON)
-   Save input prompts as GeoJSON files
-   Visualize segmentation results on interactive maps
-   Segment objects from timeseries remote sensing imagery

## Examples

-   [Segmenting remote sensing imagery](https://samgeo.gishub.org/examples/satellite)
-   [Automatically generating object masks](https://samgeo.gishub.org/examples/automatic_mask_generator)
-   [Segmenting remote sensing imagery with input prompts](https://samgeo.gishub.org/examples/input_prompts)
-   [Segmenting remote sensing imagery with box prompts](https://samgeo.gishub.org/examples/box_prompts)
-   [Segmenting remote sensing imagery with text prompts](https://samgeo.gishub.org/examples/text_prompts)
-   [Batch segmentation with text prompts](https://samgeo.gishub.org/examples/text_prompts_batch)
-   [Using segment-geospatial with ArcGIS Pro](https://samgeo.gishub.org/examples/arcgis)
-   [Segmenting swimming pools with text prompts](https://samgeo.gishub.org/examples/swimming_pools)
-   [Segmenting satellite imagery from the Maxar Open Data Program](https://samgeo.gishub.org/examples/max_open_data)

## Demos

-   Automatic mask generator

![](https://i.imgur.com/I1IhDgz.gif)

-   Interactive segmentation with input prompts

![](https://i.imgur.com/2Nyg9uW.gif)

-   Input prompts from existing files

![](https://i.imgur.com/Cb4ZaKY.gif)

-   Interactive segmentation with text prompts

![](https://i.imgur.com/wydt5Xt.gif)

## Tutorials

Video tutorials are available on my [YouTube Channel](https://youtube.com/@giswqs).

-   Automatic mask generation

[![Alt text](https://img.youtube.com/vi/YHA_-QMB8_U/0.jpg)](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcrg5RnZjkB_KY6tv96WO2h)

-   Using SAM with ArcGIS Pro

[![Alt text](https://img.youtube.com/vi/VvyInoQ6N8Q/0.jpg)](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcrg5RnZjkB_KY6tv96WO2h)

-   Interactive segmentation with text prompts

[![Alt text](https://img.youtube.com/vi/cSDvuv1zRos/0.jpg)](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcrg5RnZjkB_KY6tv96WO2h)

## Using SAM with Desktop GIS

-   **QGIS**: Check out the [Geometric Attributes plugin for QGIS](https://github.com/BjornNyberg/Geometric-Attributes-Toolbox/wiki/User-Guide#segment-anything-model). Credit goes to [Bjorn Nyberg](https://github.com/BjornNyberg).
-   **ArcGIS**: Check out the [Segment Anything Model (SAM) Toolbox for ArcGIS](https://www.arcgis.com/home/item.html?id=9b67b441f29f4ce6810979f5f0667ebe) and the [Resources for Unlocking the Power of Deep Learning Applications Using ArcGIS](https://community.esri.com/t5/education-blog/resources-for-unlocking-the-power-of-deep-learning/ba-p/1293098). Credit goes to [Esri](https://www.esri.com).

## Computing Resources

The Segment Anything Model is computationally intensive, and a powerful GPU is recommended to process large datasets. It is recommended to have a GPU with at least 8 GB of GPU memory. You can utilize the free GPU resources provided by Google Colab. Alternatively, you can apply for [AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research), which offers cloud credits to support academic research. If you are in the Greater China region, apply for the AWS Cloud Credit [here](https://aws.amazon.com/cn/events/educate_cloud/research-credits).

## Legal Notice

This repository and its content are provided for educational purposes only. By using the information and code provided, users acknowledge that they are using the APIs and models at their own risk and agree to comply with any applicable laws and regulations. Users who intend to download a large number of image tiles from any basemap are advised to contact the basemap provider to obtain permission before doing so. Unauthorized use of the basemap or any of its components may be a violation of copyright laws or other applicable laws and regulations.

## Acknowledgements

This project is based upon work partially supported by the National Aeronautics and Space Administration (NASA) under Grant No. 80NSSC22K1742 issued through the [Open Source Tools, Frameworks, and Libraries 2020 Program](https://bit.ly/3RVBRcQ).

This project is also supported by Amazon Web Services ([AWS](https://aws.amazon.com/)). In addition, this package was made possible by the following open source projects. Credit goes to the developers of these projects.

-   [segment-anything](https://github.com/facebookresearch/segment-anything) 💻
-   [segment-anything-eo](https://github.com/aliaksandr960/segment-anything-eo) 🛰️
-   [tms2geotiff](https://github.com/gumblex/tms2geotiff) 📷
-   [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 🦖
-   [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything) 📝
