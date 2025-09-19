#!/usr/bin/env python

"""Tests for `samgeo` package."""

import os
import unittest

from samgeo import common


class TestCommon(unittest.TestCase):
    """Tests for the common.py module."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_is_colab(self):
        self.assertFalse(common.is_colab())

    def test_check_file_path(self):
        self.assertTrue(common.check_file_path("tests/test_common.py"))

    def test_temp_file_path(self):
        self.assertFalse(os.path.exists(common.temp_file_path(extension=".tif")))

    def test_github_raw_url(self):
        self.assertEqual(
            common.github_raw_url(
                "https://github.com/opengeos/segment-geospatial/blob/main/samgeo/samgeo.py"
            ),
            "https://raw.githubusercontent.com/opengeos/segment-geospatial/main/samgeo/samgeo.py",
        )

    def test_download_file(self):
        self.assertTrue(
            common.download_file(
                url="https://github.com/opengeos/leafmap/raw/master/examples/data/world_cities.csv"
            )
        )

    def test_image_to_cog(self):
        image = "https://github.com/opengeos/data/raw/main/raster/landsat7.tif"
        cog = "tests/data/landsat7_cog.tif"

        common.image_to_cog(image, cog)

        self.assertTrue(os.path.exists(cog))

    def test_vector_to_geojson(self):
        vector = "https://github.com/opengeos/leafmap/raw/master/examples/data/world_cities.geojson"
        self.assertIsInstance(common.vector_to_geojson(vector), dict)

    def test_tms_to_geotiff(self):
        bbox = [-95.3704, 29.6762, -95.368, 29.6775]
        image = "satellite.tif"
        common.tms_to_geotiff(
            output=image, bbox=bbox, zoom=20, source="Satellite", overwrite=True
        )
        self.assertTrue(os.path.exists(image))
