#!/usr/bin/env python

"""Tests for `samgeo` package."""

import os
import unittest

from samgeo import samgeo


class TestSamgeo(unittest.TestCase):
    """Tests for `samgeo` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        bbox = [-122.1497, 37.6311, -122.1203, 37.6458]
        image = "satellite.tif"
        samgeo.tms_to_geotiff(
            output=image, bbox=bbox, zoom=15, source="Satellite", overwrite=True
        )
        self.source = image

        out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")
        self.checkpoint = checkpoint

        sam = samgeo.SamGeo(
            model_type="vit_h",
            checkpoint=checkpoint,
            sam_kwargs=None,
        )

        self.sam = sam

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_generate(self):
        """Test the automatic generation of masks and annotations."""
        sam = self.sam
        source = self.source

        sam.generate(source, output="masks.tif", foreground=True, unique=True)
        self.assertTrue(os.path.exists("masks.tif"))

        sam.show_anns(axis="off", alpha=1, output="annotations.tif")
        self.assertTrue(os.path.exists("annotations.tif"))

        sam.tiff_to_vector("masks.tif", "masks.gpkg")
        self.assertTrue(os.path.exists("masks.gpkg"))

    def test_predict(self):
        """Test the prediction of masks and annotations based on input prompts."""
        sam = samgeo.SamGeo(
            model_type="vit_h",
            checkpoint=self.checkpoint,
            automatic=False,
            sam_kwargs=None,
        )

        sam.set_image(self.source)
        point_coords = [[-122.1419, 37.6383]]
        sam.predict(
            point_coords, point_labels=1, point_crs="EPSG:4326", output="mask1.tif"
        )
        self.assertTrue(os.path.exists("mask1.tif"))

        point_coords = [
            [-122.1464, 37.6431],
            [-122.1449, 37.6415],
            [-122.1451, 37.6395],
        ]
        sam.predict(
            point_coords, point_labels=1, point_crs="EPSG:4326", output="mask2.tif"
        )
        self.assertTrue(os.path.exists("mask2.tif"))
