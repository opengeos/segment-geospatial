#!/usr/bin/env python

"""Tests for `samgeo` package."""

import os
import unittest

from samgeo import samgeo


class TestSamgeo(unittest.TestCase):
    """Tests for `samgeo` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        bbox = [-122.2659, 37.8682, -122.2521, 37.8741]
        image = "satellite.tif"
        samgeo.tms_to_geotiff(
            output=image, bbox=bbox, zoom=17, source="Satellite", overwrite=True
        )
        self.source = image

        out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")

        sam = samgeo.SamGeo(
            model_type="vit_h",
            checkpoint=checkpoint,
            sam_kwargs=None,
        )

        self.sam = sam

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_generate(self):
        """Test something."""

        sam = self.sam
        source = self.source

        sam.generate(source, output="masks.tif", foreground=True, unique=True)
        self.assertTrue(os.path.exists("masks.tif"))
