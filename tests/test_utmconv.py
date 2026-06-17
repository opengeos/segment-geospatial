"""Tests for the dependency-free UTM conversion helpers."""

import importlib.util
import math
from pathlib import Path
import unittest


def _load_utmconv():
    module_path = Path(__file__).resolve().parents[1] / "samgeo" / "utmconv.py"
    spec = importlib.util.spec_from_file_location("utmconv", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utmconv = _load_utmconv()


class TestUtmconv(unittest.TestCase):
    """Tests for samgeo/utmconv.py."""

    def assert_close(self, actual, expected, tolerance):
        self.assertLessEqual(abs(actual - expected), tolerance)

    def test_degree_radian_conversions_round_trip(self):
        for degrees in [-180.0, -45.5, 0.0, 37.25, 180.0]:
            radians = utmconv.deg2rad(degrees)
            self.assert_close(radians, math.radians(degrees), 1e-12)
            self.assert_close(utmconv.rad2deg(radians), degrees, 1e-12)

    def test_utm_central_meridian_for_edge_zones(self):
        expected_degrees = {
            1: -177.0,
            30: -3.0,
            31: 3.0,
            60: 177.0,
        }

        for zone, central_meridian in expected_degrees.items():
            self.assert_close(
                utmconv.rad2deg(utmconv.utm_central_meridian(zone)),
                central_meridian,
                1e-12,
            )

    def test_latlon_to_utm_known_equator_point(self):
        easting, northing = utmconv.latlon2utmxy(
            utmconv.deg2rad(0.0),
            utmconv.deg2rad(0.0),
            31,
        )

        self.assert_close(easting, 166021.443096, 1e-6)
        self.assert_close(northing, 0.0, 1e-9)

    def test_latlon_to_utm_known_central_meridian_point(self):
        easting, northing = utmconv.latlon2utmxy(
            utmconv.deg2rad(-75.0),
            utmconv.deg2rad(40.0),
            18,
        )

        self.assert_close(easting, 500000.0, 1e-6)
        self.assert_close(northing, 4427757.218472, 1e-6)

    def test_latlon_utm_round_trip_for_northern_and_southern_points(self):
        cases = [
            (-75.0, 40.0, 18, False),
            (151.0, -33.0, 56, True),
            (179.9, 0.1, 60, False),
            (-177.0, -79.9, 1, True),
        ]

        for longitude, latitude, zone, is_south_hemi in cases:
            with self.subTest(
                longitude=longitude,
                latitude=latitude,
                zone=zone,
                is_south_hemi=is_south_hemi,
            ):
                easting, northing = utmconv.latlon2utmxy(
                    utmconv.deg2rad(longitude),
                    utmconv.deg2rad(latitude),
                    zone,
                )
                round_trip_longitude, round_trip_latitude = utmconv.utmxy2latlon(
                    easting,
                    northing,
                    zone,
                    is_south_hemi,
                )

                self.assert_close(
                    utmconv.rad2deg(round_trip_longitude), longitude, 1e-6
                )
                self.assert_close(utmconv.rad2deg(round_trip_latitude), latitude, 2e-5)

    def test_southern_hemisphere_false_northing_is_reversible(self):
        longitude = 151.0
        latitude = -33.0
        zone = 56

        easting, northing = utmconv.latlon2utmxy(
            utmconv.deg2rad(longitude),
            utmconv.deg2rad(latitude),
            zone,
        )

        self.assertGreater(northing, 0.0)

        round_trip_longitude, round_trip_latitude = utmconv.utmxy2latlon(
            easting,
            northing,
            zone,
            True,
        )

        self.assert_close(utmconv.rad2deg(round_trip_longitude), longitude, 1e-6)
        self.assert_close(utmconv.rad2deg(round_trip_latitude), latitude, 2e-5)


if __name__ == "__main__":
    unittest.main()
