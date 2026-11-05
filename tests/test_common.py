#!/usr/bin/env python

"""Tests for `samgeo` package."""

import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

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
                crs="EPSG:4326",
                transform=from_origin(-180, 90, 1, 1),
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

    def test_raster_to_vector_empty_mask(self):
        """An all-zero mask vectorizes to an empty layer, not a crash.

        Regression: when a segmentation produces no foreground pixels,
        raster_to_vector built a GeoDataFrame with no geometry column and
        raised "no active geometry column" from geopandas. It should instead
        write a valid, empty GeoJSON FeatureCollection.
        """
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "empty_mask.tif")
            data = np.zeros((1, 8, 8), dtype=np.uint8)
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=from_origin(-180, 90, 1, 1),
            ) as dst:
                dst.write(data)

            out = os.path.join(tmpdir, "out.geojson")
            common.raster_to_vector(mask_path, out)
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                fc = json.load(f)
            self.assertEqual(fc["type"], "FeatureCollection")
            self.assertEqual(fc["features"], [])

    def test_raster_to_vector_attributes_and_default_output_path(self):
        """Vector exports support Name/Type/custom attributes and default local paths."""
        import geopandas as gpd

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "segmented_tile.tif")
            data = np.zeros((1, 8, 8), dtype=np.uint8)
            data[0, :4, :4] = 1
            data[0, 4:, 4:] = 2
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=from_origin(-180, 90, 1, 1),
            ) as dst:
                dst.write(data)

            # Default output path in the same local directory as source image (.shp)
            shp_path = common.raster_to_shp(
                mask_path,
                attributes={
                    "Name": "segmented_tile.tif",
                    "Type": "point",
                    "score": {1: 0.95, 2: 0.8},
                },
            )
            self.assertEqual(shp_path, os.path.join(tmpdir, "segmented_tile.shp"))
            self.assertTrue(os.path.exists(shp_path))

            gdf = gpd.read_file(shp_path)
            self.assertEqual(len(gdf), 2)
            self.assertTrue((gdf["Name"] == "segmented_tile.tif").all())
            self.assertTrue((gdf["Type"] == "point").all())
            scores_by_val = dict(zip(gdf["value"].astype(int), gdf["score"]))
            self.assertAlmostEqual(scores_by_val[1], 0.95, places=5)
            self.assertAlmostEqual(scores_by_val[2], 0.8, places=5)

            # Default output paths for gpkg, geojson, and raster_to_vector
            gpkg_path = common.raster_to_gpkg(mask_path)
            self.assertEqual(gpkg_path, os.path.join(tmpdir, "segmented_tile.gpkg"))
            self.assertTrue(os.path.exists(gpkg_path))

            geojson_path = common.raster_to_geojson(
                mask_path, attributes={"Name": "segmented_tile.tif", "Type": "text"}
            )
            self.assertEqual(
                geojson_path, os.path.join(tmpdir, "segmented_tile.geojson")
            )
            self.assertTrue(os.path.exists(geojson_path))
            gdf_json = gpd.read_file(geojson_path)
            self.assertEqual(list(gdf_json["Type"]), ["text", "text"])

            # Empty mask with attributes preserves attribute schema in shapefile
            empty_mask = os.path.join(tmpdir, "empty_tile.tif")
            with rasterio.open(
                empty_mask,
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype=np.uint8,
                crs="EPSG:4326",
                transform=from_origin(-180, 90, 1, 1),
            ) as dst:
                dst.write(np.zeros((1, 4, 4), dtype=np.uint8))

            empty_shp = common.raster_to_shp(
                empty_mask, attributes={"Name": "empty_tile.tif", "Type": "box"}
            )
            gdf_empty = gpd.read_file(empty_shp)
            self.assertEqual(len(gdf_empty), 0)
            self.assertIn("Name", gdf_empty.columns)
            self.assertIn("Type", gdf_empty.columns)

            with self.assertRaises(ValueError):
                common.raster_to_vector(mask_path, attributes=["invalid"])
            with self.assertRaises(ValueError):
                common.raster_to_vector(mask_path, attributes={"bad_len": [1, 2, 3]})

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
