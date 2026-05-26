#!/usr/bin/env python

"""Tests for `samgeo` package."""

import os
import tempfile
import unittest

import numpy as np
import rasterio

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

    def test_prepare_image_for_sam_multichannel_array(self):
        image = np.zeros((4, 5, 5), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        image[..., 3] = 40
        image[..., 4] = 50

        rgb = common.prepare_image_for_sam(image)
        self.assertEqual(rgb.shape, (4, 5, 3))
        np.testing.assert_array_equal(rgb[..., 0], image[..., 0])
        np.testing.assert_array_equal(rgb[..., 1], image[..., 1])
        np.testing.assert_array_equal(rgb[..., 2], image[..., 2])

        false_color = common.prepare_image_for_sam(image, bands=[5, 3, 1])
        np.testing.assert_array_equal(false_color[..., 0], image[..., 4])
        np.testing.assert_array_equal(false_color[..., 1], image[..., 2])
        np.testing.assert_array_equal(false_color[..., 2], image[..., 0])

    def test_read_image_for_sam_multiband_geotiff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image = os.path.join(tmpdir, "multiband.tif")
            data = np.zeros((5, 4, 6), dtype=np.uint8)
            for band in range(data.shape[0]):
                data[band] = (band + 1) * 10

            with rasterio.open(
                image,
                "w",
                driver="GTiff",
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
            ) as dst:
                dst.write(data)

            rgb = common.read_image_for_sam(image)
            self.assertEqual(rgb.shape, (4, 6, 3))
            np.testing.assert_array_equal(rgb[..., 0], data[0])
            np.testing.assert_array_equal(rgb[..., 1], data[1])
            np.testing.assert_array_equal(rgb[..., 2], data[2])

            false_color = common.read_image_for_sam(image, bands=[5, 3, 1])
            np.testing.assert_array_equal(false_color[..., 0], data[4])
            np.testing.assert_array_equal(false_color[..., 1], data[2])
            np.testing.assert_array_equal(false_color[..., 2], data[0])

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
