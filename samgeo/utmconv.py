"""The module is used to convert between lat/lon and UTM coordinates.
The source code is adapted from https://doi.org/10.6084/m9.figshare.25347187.
Credits to the original authors Yang et al. (2024).
Reference: https://doi.org/10.1080/10106049.2024.2370322
"""

# -*- coding:utf-8 -*-
from math import pi, sin, cos, tan, sqrt

sm_a = 6378137.0
sm_b = 6356752.314
sm_EccSquared = 6.69437999013e-03

UTMScaleFactor = 0.9996


def deg2rad(deg):
    return deg / 180.0 * pi


def rad2deg(rad):
    return rad / pi * 180.0


def arc_length_of_meridian(phi):
    result = -1

    n = (sm_a - sm_b) / (sm_a + sm_b)

    alpha = ((sm_a + sm_b) / 2.0) * (1.0 + (pow(n, 2.0) / 4.0) + (pow(n, 4.0) / 64.0))

    beta = (-3.0 * n / 2.0) + (9.0 * pow(n, 3.0) / 16.0) + (-3.0 * pow(n, 5.0) / 32.0)

    gamma = (15.0 * pow(n, 2.0) / 16.0) + (-15.0 * pow(n, 4.0) / 32.0)

    delta = (-35.0 * pow(n, 3.0) / 48.0) + (105.0 * pow(n, 5.0) / 256.0)

    epsilon = 315.0 * pow(n, 4.0) / 512.0

    result = alpha * (
        phi
        + (beta * sin(2.0 * phi))
        + (gamma * sin(4.0 * phi))
        + (delta * sin(6.0 * phi))
        + (epsilon * sin(8.0 * phi))
    )

    return result


def utm_central_meridian(zone):
    return deg2rad(-183.0 + (zone * 6.0))


def footpoint_latitude(y):

    result = -1

    n = (sm_a - sm_b) / (sm_a + sm_b)

    alpha_ = ((sm_a + sm_b) / 2.0) * (1 + (pow(n, 2.0) / 4) + (pow(n, 4.0) / 64))

    y_ = y / alpha_

    beta_ = (
        (3.0 * n / 2.0) + (-27.0 * pow(n, 3.0) / 32.0) + (269.0 * pow(n, 5.0) / 512.0)
    )

    gamma_ = (21.0 * pow(n, 2.0) / 16.0) + (-55.0 * pow(n, 4.0) / 32.0)

    delta_ = (151.0 * pow(n, 3.0) / 96.0) + (-417.0 * pow(n, 5.0) / 128.0)

    epsilon_ = 1097.0 * pow(n, 4.0) / 512.0

    result = (
        y_
        + (beta_ * sin(2.0 * y_))
        + (gamma_ * sin(4.0 * y_))
        + (delta_ * sin(6.0 * y_))
        + (epsilon_ * sin(8.0 * y_))
    )

    return result


def map_latlon2xy(lng, lat, cent_meridian):

    ep2 = (pow(sm_a, 2.0) - pow(sm_b, 2.0)) / pow(sm_b, 2.0)

    nu2 = ep2 * pow(cos(lat), 2.0)

    N = pow(sm_a, 2.0) / (sm_b * sqrt(1 + nu2))

    t = tan(lat)
    t2 = t * t
    tmp = (t2 * t2 * t2) - pow(t, 6.0)

    l = lng - cent_meridian

    l3coef = 1.0 - t2 + nu2

    l4coef = 5.0 - t2 + 9 * nu2 + 4.0 * (nu2 * nu2)

    l5coef = 5.0 - 18.0 * t2 + (t2 * t2) + 14.0 * nu2 - 58.0 * t2 * nu2

    l6coef = 61.0 - 58.0 * t2 + (t2 * t2) + 270.0 * nu2 - 330.0 * t2 * nu2

    l7coef = 61.0 - 479.0 * t2 + 179.0 * (t2 * t2) - (t2 * t2 * t2)

    l8coef = 1385.0 - 3111.0 * t2 + 543.0 * (t2 * t2) - (t2 * t2 * t2)

    x = (
        N * cos(lat) * l
        + (N / 6.0 * pow(cos(lat), 3.0) * l3coef * pow(l, 3.0))
        + (N / 120.0 * pow(cos(lat), 5.0) * l5coef * pow(l, 5.0))
        + (N / 5040.0 * pow(cos(lat), 7.0) * l7coef * pow(l, 7.0))
    )
    y = (
        arc_length_of_meridian(lat)
        + (t / 2.0 * N * pow(cos(lat), 2.0) * pow(l, 2.0))
        + (t / 24.0 * N * pow(cos(lat), 4.0) * l4coef * pow(l, 4.0))
        + (t / 720.0 * N * pow(cos(lat), 6.0) * l6coef * pow(l, 6.0))
        + (t / 40320.0 * N * pow(cos(lat), 8.0) * l8coef * pow(l, 8.0))
    )
    return x, y


def map_xy2latlon(x, y, lambda0):

    phif = footpoint_latitude(y)

    ep2 = (pow(sm_a, 2.0) - pow(sm_b, 2.0)) / pow(sm_b, 2.0)

    cf = cos(phif)

    nuf2 = ep2 * pow(cf, 2.0)

    Nf = pow(sm_a, 2.0) / (sm_b * sqrt(1 + nuf2))
    Nfpow = Nf

    tf = tan(phif)
    tf2 = tf * tf
    tf4 = tf2 * tf2

    x1frac = 1.0 / (Nfpow * cf)

    Nfpow *= Nf
    x2frac = tf / (2.0 * Nfpow)

    Nfpow *= Nf
    x3frac = 1.0 / (6.0 * Nfpow * cf)

    Nfpow *= Nf
    x4frac = tf / (24.0 * Nfpow)

    Nfpow *= Nf
    x5frac = 1.0 / (120.0 * Nfpow * cf)

    Nfpow *= Nf
    x6frac = tf / (720.0 * Nfpow)

    Nfpow *= Nf
    x7frac = 1.0 / (5040.0 * Nfpow * cf)

    x8frac = tf / (40320.0 * Nfpow)

    x2poly = -1.0 - nuf2

    x3poly = -1.0 - 2 * tf2 - nuf2

    x4poly = (
        5.0
        + 3.0 * tf2
        + 6.0 * nuf2
        - 6.0 * tf2 * nuf2
        - 3.0 * (nuf2 * nuf2)
        - 9.0 * tf2 * (nuf2 * nuf2)
    )

    x5poly = 5.0 + 28.0 * tf2 + 24.0 * tf4 + 6.0 * nuf2 + 8.0 * tf2 * nuf2

    x6poly = -61.0 - 90.0 * tf2 - 45.0 * tf4 - 107.0 * nuf2 + 162.0 * tf2 * nuf2

    x7poly = -61.0 - 662.0 * tf2 - 1320.0 * tf4 - 720.0 * (tf4 * tf2)

    x8poly = 1385.0 + 3633.0 * tf2 + 4095.0 * tf4 + 1575 * (tf4 * tf2)

    lat = (
        phif
        + x2frac * x2poly * (x * x)
        + x4frac * x4poly * pow(x, 4.0)
        + x6frac * x6poly * pow(x, 6.0)
        + x8frac * x8poly * pow(x, 8.0)
    )

    lng = (
        lambda0
        + x1frac * x
        + x3frac * x3poly * pow(x, 3.0)
        + x5frac * x5poly * pow(x, 5.0)
        + x7frac * x7poly * pow(x, 7.0)
    )

    return lng, lat


def latlon2utmxy(lng, lat, zone):
    x1, y1 = map_latlon2xy(lng, lat, utm_central_meridian(zone))
    x1 = x1 * UTMScaleFactor + 500000.0
    y1 = y1 * UTMScaleFactor
    if y1 < 0.0:
        y1 += 10000000.0
    return x1, y1


def utmxy2latlon(x, y, zone, is_south_hemi):

    x -= 500000.0
    x /= UTMScaleFactor

    if is_south_hemi:
        y -= 10000000.0

    y /= UTMScaleFactor

    cmeridian = utm_central_meridian(zone)
    lng, lat = map_xy2latlon(x, y, cmeridian)
    return lng, lat


def main():
    x0 = 116
    y0 = 40
    x1, y1 = coordconvert.wgs84_to_gcj02(x0, y0)

    x2, y2 = latlon2utmxy(deg2rad(x1), deg2rad(y1), 50)
    x4, y4 = utmxy2latlon(x2, y2, 50, False)

    x3, y3 = coordconvert.gcj02_to_utm50(x1, y1)


if __name__ == "__main__":
    main()
