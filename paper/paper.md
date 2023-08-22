---
title: "samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM)"
tags:
    - Python
    - geospatial
    - segment anything
    - deep learning
    - satellite

authors:
    - name: Qiusheng Wu
      orcid: 0000-0001-5437-4073
      affiliation: 1
    - name: Lucas Prado Osco
      orcid: 0000-0002-0258-536X
      affiliation: 2

affiliations:
    - name: Department of Geography & Sustainability, University of Tennessee, Knoxville, TN 37996, United States
      index: 1
    - name: Faculty of Engineering and Architecture and Urbanism, University of Western São Paulo, Rod. Raposo Tavares, km 572 - Limoeiro, Pres. Prudente 19067-175, SP, Brazil
      index: 2

date: 18 May 2023
bibliography: paper.bib
---

# Summary

Segment-Geospatial (samgeo) is an open-source Python package designed to simplify the process of segmenting geospatial data with the Segment Anything Model [@Kirillov2023]. The package leverages popular Python libraries, such as leafmap [@Wu2021], ipywidgets [@Grout2021], rasterio [@Gillies2013], geopandas [@Jordahl2021], and segment-anything-py [@Wu2023], to provide a straightforward interface for users to segment remote sensing imagery and export the results in various formats, including vector and raster data. The segmentation can be run automatically, interactively in a graphical user interface (GUI), or by text prompts built upon Grounding DINO [@liu2023]. However, it's worth noting that the text prompt approach has its limitations, which may require parameter fine-tuning. Additionally, the package provides functionality for downloading remote sensing imagery and visualizing segmentation results interactively in a Jupyter environment. Segment-Geospatial aims to fill the gap in the Python ecosystem by providing a user-friendly, efficient, and flexible geospatial segmentation tool without the need for training deep learning models.

# Statement of need

Image segmentation is a critical task in geospatial analysis as it enables the identification and extraction of relevant features from satellite or aerial imagery. By segmenting an image into meaningful regions, it becomes possible to extract information about the spatial distribution and characteristics of various objects and land cover types. This information can then be used to support decision-making in a wide range of fields, from agriculture and forestry to environmental monitoring and national security.

Traditionally, image segmentation has been performed using manual or semi-automatic methods, which are time-consuming and labor-intensive. In recent years, deep learning models have been developed to automate the segmentation process. However, these models generally require large amounts of training data and are computationally expensive, making them impractical for many applications. The Segment Anything Model (SAM) [@Kirillov2023] recently released by Meta AI is a promptable segmentation system with zero-shot generalization to unfamiliar objects and images, without the need for additional training. The model was trained on 11 million images with over 1 billion masks. Users can segment any object in any image using existing model checkpoints without the need for additional training. Since its release in April 2023, there has been a plethora of applications of SAM in various fields [@Zhang2023; @Ji2023], such as medical imaging [@Ma2023]. Currently, there are few Python packages available on PyPI and conda-forge for segmenting images in the geospatial domain with SAM. Therefore, we developed the Segment-Geospatial Python package to fill this gap and provide a low-code and no-code solution for segmenting geospatial data with SAM.

# Acknowledgements

The segment-geospatial package draws its inspiration from segment-anything-eo repository [@Hancharenka2023]. We thank the author Aliaksandr Hancharenka for his excellent work. We also thank the authors of the Segment Anything Model [@Kirillov2023] for making the model available to the public.

This project is based upon work partially supported by the National Aeronautics and Space Administration (NASA) under Grant No. 80NSSC22K1742 issued through the [Open Source Tools, Frameworks, and Libraries 2020 Program](https://bit.ly/3RVBRcQ). This project is also supported by Amazon Web Services ([AWS](https://aws.amazon.com/)).

# References
