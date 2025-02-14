"""The module is used to regularize building footprints using the feature
edge reconstruction algorithm (FER).
The source code is adapted from https://doi.org/10.6084/m9.figshare.25347187.
Credits to the original authors Yang et al. (2024).
Reference: https://doi.org/10.1080/10106049.2024.2370322
"""

import os, time, shutil
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import polygonize, unary_union
from utmconv import latlon2utmxy, deg2rad, utmxy2latlon, rad2deg
from osgeo import gdal, ogr, osr


class Point:

    x = 0.0
    y = 0.0
    index = 0

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index


class Vector:
    x1 = 0.0
    y1 = 0.0
    x2 = 0.0
    y2 = 0.0
    x = 0.0
    y = 0.0
    index = 0

    def __init__(self, x1, y1, x2, y2, index):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.index = index
        self.x = self.x2 - self.x1
        self.y = self.y2 - self.y1

    def length(self):
        return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))

    def cos(self, v):
        dot = self.x * v.x + self.y * v.y
        cos = dot / (math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2)) * v.length())

        if cos > 1:
            cos = 1
        if cos < -1:
            cos = -1
        return 180 / math.pi * math.acos(cos)

    def k(self):
        k = (self.y2 - self.y1) / (self.x2 - self.x1)
        return k

    def checkV(self):
        if self.x1 == self.x2 and self.y1 == self.y2:
            return False
        return True

    def equals(self, v):

        if self.x1 != v.x1 and self.y1 != v.y1 and self.x1 != v.x1 and self.y1 != v.y1:
            return False
        return True


class AppendV:
    v1 = Vector(0, 0, 0, 0, 0)
    v2 = Vector(0, 0, 0, 0, 0)
    domain = 0
    priority = 0
    fit = 0

    def __init__(self, v1, v2, domain, priority, fit):
        self.v1 = v1
        self.v2 = v2
        self.domain = domain
        self.priority = priority
        self.fit = fit


def time_interval(time_end, time_start):
    seconds = time_end - time_start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def CheckFileExists(filePath):

    if os.path.exists(filePath):
        return 1
    print(f"not exut！{filePath}")
    return 0


def ReadVectorLayer(strVectorFile):
    SUPPORTED_FORMATS = {
        ".shp": "ESRI Shapefile",
        ".geojson": "GeoJSON",
        ".kml": "KML",
        ".gml": "GML",
        ".gpkg": "GPKG",
        ".sqlite": "SQLite",
        ".csv": "CSV",
    }

    ext = os.path.splitext(strVectorFile)[1].lower()
    driver_name = SUPPORTED_FORMATS.get(ext)
    if not driver_name:
        print(f"Unsupported file format: {ext}")
        return None

    driver = ogr.GetDriverByName(driver_name)
    ds = driver.Open(strVectorFile, 0)
    if ds is None:
        print(f"Failed to open file: {strVectorFile}")
        return None
    print(f"Successfully opened file: {strVectorFile}")
    return ds


def ReadVectorMessage(ds):

    layer = ds.GetLayer(0)
    lydefn = layer.GetLayerDefn()
    spatialref = layer.GetSpatialRef()
    geomtype = lydefn.GetGeomType()

    fieldlist = []
    for i in range(lydefn.GetFieldCount()):
        fddefn = lydefn.GetFieldDefn(i)
        fddict = {
            "name": fddefn.GetName(),
            "type": fddefn.GetType(),
            "width": fddefn.GetWidth(),
            "decimal": fddefn.GetPrecision(),
        }
        fieldlist += [fddict]

    return (spatialref, geomtype, fieldlist)


def CreateVectorFile(strVectorFile, sourceData):
    SUPPORTED_FORMATS = {
        ".shp": "ESRI Shapefile",
        ".geojson": "GeoJSON",
        ".kml": "KML",
        ".gml": "GML",
        ".gpkg": "GPKG",
        ".sqlite": "SQLite",
        ".csv": "CSV",
    }

    ext = os.path.splitext(strVectorFile)[1].lower()
    driver_name = SUPPORTED_FORMATS.get(ext)
    if not driver_name:
        print(f"Unsupported file format: {ext}")
        return None

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    spatialref, geomtype, fieldlist = sourceData

    driver = ogr.GetDriverByName(driver_name)
    DeleteVectorFile(driver, strVectorFile)
    ds = driver.CreateDataSource(strVectorFile)
    layer = ds.CreateLayer(
        os.path.basename(strVectorFile).replace(ext, ""),
        srs=spatialref,
        geom_type=geomtype,
    )
    for fd in fieldlist:
        field = ogr.FieldDefn(fd["name"], fd["type"])
        if "width" in fd:
            field.SetWidth(fd["width"])
        if "decimal" in fd:
            field.SetPrecision(fd["decimal"])
        layer.CreateField(field)

    return ds


def CreatePtVectorFile(strVectorFile, sourceData):

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    spatialref, geomtype, fieldlist = sourceData

    driver = ogr.GetDriverByName("ESRI Shapefile")
    DeleteVectorFile(driver, strVectorFile)
    ds = driver.CreateDataSource(strVectorFile)
    layer = ds.CreateLayer(
        os.path.basename(strVectorFile)[:-4], srs=spatialref, geom_type=ogr.wkbPoint
    )

    return ds


def DeleteVectorFile(driver, strVectorFile):

    try:
        if os.path.exists(strVectorFile):
            driver.DeleteDataSource(strVectorFile)
    except Exception as e:
        print("fail!" % strVectorFile)
        print("ERROR:", e)


def Ring2Pts(ring):
    pts = []
    ptCount = ring.GetPointCount()
    for i in range(ptCount):
        if i != ptCount - 1:
            pts.append(Point(ring.GetX(i), ring.GetY(i), i))
    return pts


def Pts2Ring(pts):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in pts:
        ring.AddPoint(i.x, i.y)
    ring.CloseRings()
    return ring


def Pts2Polygon(pts):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in pts:
        ring.AddPoint(i.x, i.y)
    ring.CloseRings()
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon, ring


def compress(p1, p2, points, deleteIds):
    D = 0.5
    swichvalue = False

    A = p1.y - p2.y
    B = p2.x - p1.x
    C = p1.x * p2.y - p2.x * p1.y

    m = points.index(p1)
    n = points.index(p2)
    distance = []

    middle = None

    if n == m + 1:
        return

    for i in range(m + 1, n):
        d = abs(A * points[i].x + B * points[i].y + C) / math.sqrt(
            math.pow(A, 2) + math.pow(B, 2)
        )
        distance.append(d)

    dmax = max(distance)

    if dmax > D:
        swichvalue = True
    else:
        swichvalue = False
    if not swichvalue:
        for i in range(m + 1, n):
            deleteIds.append(i)

    else:
        for i in range(m + 1, n):
            if (
                abs(A * points[i].x + B * points[i].y + C)
                / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
                == dmax
            ):
                middle = points[i]
        compress(p1, middle, points, deleteIds)
        compress(middle, p2, points, deleteIds)

    return deleteIds


def CreateGemetry(ds, polygon):
    layer = ds.GetLayer(0)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(polygon)
    layer.CreateFeature(feature)


def CirSimilar(polygon):
    area = polygon.Area()
    if area == 0.0:
        return 0
    length = polygon.Boundary().Length()
    r1 = 2 * area / length
    r2 = length / (2 * math.pi)
    if r1 / r2 > 0.9 and r1 / r2 < 1.05:
        return 1
    return 0


def CreateCircle(polygon):
    r = polygon.Boundary().Length() / (2 * math.pi)
    centerPt = polygon.Centroid()
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in range(0, 360, 10):
        x = centerPt.GetX() + r * math.cos(i * math.pi / 180)
        y = centerPt.GetY() + r * math.sin(i * math.pi / 180)
        ring.AddPoint(x, y)
    ring.CloseRings()
    return ring


def Ring2Pts1(ring):
    pts = []
    ptCount = ring.GetPointCount()
    for i in range(ptCount):
        if i != ptCount - 1:
            pts.append((ring.GetX(i), ring.GetY(i)))
    return pts


def RecSimilar(polygon):
    pts = Ring2Pts1(polygon.GetGeometryRef(0))
    if len(pts) < 3:
        return 1, polygon.GetGeometryRef(0)
    npolygon = Polygon(pts)

    # Ensure the polygon is valid
    if not npolygon.is_valid:
        print("Invalid polygon detected. Attempting to fix with buffer(1)...")
        npolygon = npolygon.buffer(100)

    # Ensure the exterior ring is closed
    if isinstance(npolygon, Polygon):
        exterior_coords = list(npolygon.exterior.coords)
        if exterior_coords[0] != exterior_coords[-1]:
            print("Closing polygon by ensuring the first and last points are the same.")
            exterior_coords.append(exterior_coords[0])  # Close the ring
            npolygon = Polygon(exterior_coords)

    try:
        minRect = npolygon.minimum_rotated_rectangle
        # print("Minimum rotated rectangle calculated.")
    except Exception as e:
        print("Error while calculating minimum rotated rectangle:")
        print(e)
        return None
    pArea = minRect.area
    areaDiff = minRect.area / polygon.Area() - 1
    if (
        (pArea < 100 and areaDiff < 0.25)
        or (pArea >= 100 and pArea < 300 and areaDiff < 0.2)
        or (pArea >= 300 and pArea < 500 and areaDiff < 0.15)
        or (pArea >= 500 and pArea < 1000 and areaDiff < 0.1)
        or (pArea >= 1000 and pArea < 2000 and areaDiff < 0.1)
        or (pArea >= 2000 and pArea < 5000 and areaDiff < 0.05)
        or (pArea >= 5000 and areaDiff < 0.02)
    ):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in minRect.exterior.coords:
            ring.AddPoint(x, y)
        return 1, ring
    return 0, polygon.GetGeometryRef(0)


def Ring2VList(ring):
    ptCount = ring.GetPointCount()
    vList = []
    for i in range(ptCount):
        if i < ptCount - 1:
            if (
                ring.GetX(i) == ring.GetX(i + 1) and ring.GetY(i) == ring.GetY(i + 1)
            ) == False:
                vList.append(
                    Vector(
                        ring.GetX(i),
                        ring.GetY(i),
                        ring.GetX(i + 1),
                        ring.GetY(i + 1),
                        i,
                    )
                )
    return vList


def VListDiret(vList):
    minWeight = 10000
    ovx = vList[0]
    for i in range(0, 90, 1):
        x = vList[0].x1 + 10 * math.cos(i * math.pi / 180)
        y = vList[0].y1 + 10 * math.sin(i * math.pi / 180)
        vx = Vector(vList[0].x1, vList[0].y1, x, y, 0)
        weight = 0
        for v in vList:
            angle = vx.cos(v)
            angleList = [
                math.fabs(90 - angle),
                math.fabs(180 - angle),
                math.fabs(angle),
            ]
            weight = weight + (min(angleList) / 90) * v.length()
        if weight < minWeight:
            minWeight = weight
            ovx = vx
    return ovx


def CreatePt(ptlayer, x, y):
    feature = ogr.Feature(ptlayer.GetLayerDefn())
    wkt = "POINT(%f %f)" % (x, y)
    point = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(point)
    ptlayer.CreateFeature(feature)


def vList2Ring(vList):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in vList:
        ring.AddPoint(i.x1, i.y1)
    ring.CloseRings()
    return ring


def vList2Ring2(vList):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if len(vList) == 0:
        return ring

    if not vList[0].checkV():
        ring.AddPoint(vList[0].x1, vList[0].y1)
    else:
        ring.AddPoint(vList[0].x1, vList[0].y1)
        ring.AddPoint(vList[0].x2, vList[0].y2)
    for i in range(1, len(vList)):
        if not vList[i].checkV():
            if vList[i].x1 == vList[i - 1].x2 and vList[i].y1 == vList[i - 1].y2:
                continue
        else:
            if vList[i].x1 == vList[i - 1].x2 and vList[i].y1 == vList[i - 1].y2:
                ring.AddPoint(vList[i].x2, vList[i].y2)
            else:
                ring.AddPoint(vList[i].x1, vList[i].y1)
                ring.AddPoint(vList[i].x2, vList[i].y2)
    ring.CloseRings()
    return ring


def FeatureLine2vList(vList):
    ovList = []
    for i in range(len(vList)):
        k = -1
        if i < len(vList) - 1:
            k = i + 1
        if i == len(vList) - 1:
            k = 0
        if vList[i].x2 != vList[k].x1 or vList[i].y2 != vList[k].y1:
            ovList.append(vList[i])
            ovList.append(
                Vector(
                    vList[i].x2, vList[i].y2, vList[k].x1, vList[k].y1, i * vList[i].x2
                )
            )
        else:
            ovList.append(vList[i])
    return ovList


def PrintvList(vList):
    print("VList:")
    for i in vList:
        print(i.x1, i.y1, "  ", i.x2, i.y2, i.index)


def IntersectPt(v1, v2, vx):
    k = vx.k()
    if k == 0:
        x = v1.x1
        y = v2.y2
        x1 = v2.x2
        y1 = v1.y1
    else:
        x = (k * v1.x1 - v1.y1 + 1 / k * v2.x2 + v2.y2) / (k + 1 / k)
        y = k * (x - v1.x1) + v1.y1
        x1 = (-1 / k * v1.x1 - v1.y1 - k * v2.x2 + v2.y2) / (-1 / k - k)
        y1 = -1 / k * (x1 - v1.x1) + v1.y1
    xc = (v1.x2 + v2.x1) / 2
    yc = (v1.y2 + v2.y1) / 2
    if math.sqrt((x - xc) ** 2 + (y - yc) ** 2) > math.sqrt(
        (x1 - xc) ** 2 + (y1 - yc) ** 2
    ):
        x = x1
        y = y1
    return x, y


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def Smooth(vList, vx, angleDif, lengthDif):
    end = False
    while not end:
        if len(vList) < 4:
            break
        k = -1
        for i in range(len(vList)):
            if i < len(vList) - 1:
                k = i + 1
            if i == len(vList) - 1:
                end = True
                k = 0

            v1 = vList[i]
            v2 = vList[k]
            angle = v1.cos(v2)

            if min([angle, math.fabs(180 - angle)]) < angleDif:

                v = Vector(v1.x1, v1.y1, v2.x2, v2.y2, v1.index)
                vList.remove(v1)
                vList.remove(v2)
                if v.checkV():
                    vList.insert(i, v)
                break

            if (
                math.fabs(90 - angle) < angleDif - 5
                and math.fabs(90 - angle) > 1
                and v1.length() < lengthDif
                and v2.length() < lengthDif
            ):

                v1Angle = v1.cos(vx)
                v2Angle = v2.cos(vx)

                if (
                    math.fabs(90 - v1Angle) < angleDif and math.fabs(90 - v1Angle) > 1
                ) or (
                    math.fabs(90 - v2Angle) < angleDif and math.fabs(90 - v2Angle) > 1
                ):
                    x, y = IntersectPt(v1, v2, vx)
                    vList.remove(v1)
                    vList.remove(v2)
                    ov1 = Vector(v1.x1, v1.y1, x, y, v1.index)
                    if ov1.checkV():
                        vList.insert(i, ov1)
                    ov2 = Vector(x, y, v2.x2, v2.y2, v1.index + 1)
                    if ov2.checkV():
                        vList.insert(k, ov2)
                    break

    return vList


def Domain(v1, v2, angleDiff, area, lengthDiff):
    a1 = v1.cos(v2)
    a2 = math.fabs(v1.cos(v2) - 90)
    a3 = 180 - v1.cos(v2)

    if a1 < angleDiff and a1 < a2 and a1 < a3:
        return 0, a1
    if (
        a2 < angleDiff - 5
        and a2 < a1
        and a2 < a3
        and ((v1.x2 != v2.x1 and v1.y2 != v2.y1) or a2 >= 0.01)
    ):
        if (
            area >= 200 and v1.length() < lengthDiff and v2.length() < lengthDiff
        ) or area < 200:
            return 90, a2
        else:
            if v1.x2 != v2.x1 and v1.y2 != v2.y1:
                return 901, a2
    if a3 < angleDiff and a3 < a1 and a3 < a2:
        return 180, a3

    return -1, a1


def FeatureLine(vList, vx, angleDiff, lengthdiff1, lengthdiff2):
    avList = []
    for i in range(len(vList)):
        v1 = vList[i]
        minAngle = min([v1.cos(vx), math.fabs(v1.cos(vx) - 90), 180 - v1.cos(vx)])
        if (minAngle < angleDiff and v1.length() > lengthdiff1) or (
            v1.length() > lengthdiff2
        ):
            avList.append(v1)

    return avList


def LineRelation(v1, v2, vx, angleDiff, lenthDiff, area):
    domain, angle = Domain(v1, v2, angleDiff, lenthDiff, area)
    domainV1, angleV1 = Domain(v1, vx, angleDiff, lenthDiff, area)
    domainV2, angleV2 = Domain(v2, vx, angleDiff, lenthDiff, area)
    if angleV1 >= angleV2:
        v1 = v2
        angleV1 = angleV2
    return AppendV(v1, v2, domain, v1, angle)


def ParaDistance(v1, v2, vx, lengthDiff, area, lengthdiff2):
    x = (v1.x2 + v2.x1) / 2
    y = (v1.y2 + v2.y1) / 2

    if v1.x1 == v1.x2:
        k1 = 0
        k2 = 0
        b1 = y - k1 * x
        b2 = y - k2 * x
        yn = k1 * (x + 1) + b1
        vn1 = Vector(x + 1, yn, x, y, 0)
        yn1 = k2 * (x + 1) + b2
        vn2 = Vector(x, y, x + 1, yn1, 1)
    else:
        if v1.k() != 0:
            k1 = -1 / v1.k()
            k2 = k1
            b1 = y - k1 * x
            b2 = y - k2 * x
            yn = k1 * (x + 1) + b1
            vn1 = Vector(x + 1, yn, x, y, 0)
            yn1 = k2 * (x + 1) + b2
            vn2 = Vector(x, y, x + 1, yn1, 1)

        if v1.k() == 0:
            vn1 = Vector(x, v1.y1, x, y, 0)
            vn2 = Vector(x, y, x, v2.y1, 1)

    x1, y1 = IntersectPt(v1, vn1, vx)
    d1 = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    x2, y2 = IntersectPt(v2, vn2, vx)
    d2 = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5

    vp1 = Vector(v1.x2, v1.y2, x1, y1, 0)
    vp2 = Vector(v2.x1, v2.y1, x2, y2, 0)
    v12 = Vector(v1.x2, v1.y2, v2.x1, v2.y1, 0)

    if (vp1.length() == 0 or vp2.length() == 0) and d1 + d2 > lengthDiff:
        return 0, 0, 0, 0, -1

    if d1 + d2 <= lengthDiff or v12.length() < lengthDiff / 1.5:
        if (
            area >= 100 and v1.length() < lengthdiff2 and v2.length() < lengthdiff2
        ) or area < 100:
            return 0, 0, 0, 0, 0
        else:
            return 0, 0, 0, 0, 3

    if d1 + d2 > lengthDiff:
        if (
            area >= 200 and v1.length() < lengthdiff2 and v2.length() < lengthdiff2
        ) or area < 200:
            return x1, y1, x2, y2, 1
        else:
            return 0, 0, 0, 0, -1


def LocalResc(vList, vx, angleDiff, lengthdiff1, lengthdiff2, area):
    end = False
    vl_num = -1

    while not end:
        if len(vList) < 2 or vl_num == len(vList):
            return vList
        for i in range(len(vList)):
            k = -1
            if i == len(vList) - 1:
                end = True
                k = 0
            if i < len(vList) - 1:
                k = i + 1
            v1 = vList[i]
            v2 = vList[k]
            aV = LineRelation(v1, v2, vx, angleDiff, area, lengthdiff2 * 3)

            if aV.domain == 90:
                x, y = IntersectPt(v1, v2, vx)
                ov1 = Vector(v1.x1, v1.y1, x, y, v1.index)
                ov2 = Vector(x, y, v2.x2, v2.y2, v2.index + 1)
                vList.remove(v1)
                vList.remove(v2)
                if ov1.checkV():
                    vList.insert(i, ov1)
                if ov2.checkV():
                    vList.insert(k, ov2)
                break

            if aV.domain == 901:

                x, y = line_intersection(
                    ((v1.x1, v1.y1), (v1.x2, v1.y2)), ((v2.x1, v2.y1), (v2.x2, v2.y2))
                )
                ov1 = Vector(v1.x1, v1.y1, x, y, v1.index)
                ov2 = Vector(x, y, v2.x2, v2.y2, v2.index + 1)
                vList.remove(v1)
                vList.remove(v2)
                if ov1.checkV():
                    vList.insert(i, ov1)
                if ov2.checkV():
                    vList.insert(k, ov2)
                break

            if aV.domain == 0 or aV.domain == 180:
                x1, y1, x2, y2, d = ParaDistance(
                    v1, v2, vx, lengthdiff1, area, lengthdiff2 * 3
                )
                if d == -1:
                    continue
                if d == 1:
                    if v1.index == v2.index:
                        continue

                    ov1 = Vector(v1.x1, v1.y1, x1, y1, v1.x1 * lengthdiff1 + v2.x2)
                    ov2 = Vector(x2, y2, v2.x2, v2.y2, v1.x1 * lengthdiff1 + v2.x2)
                    vList.remove(v1)
                    vList.remove(v2)
                    if ov1.checkV():
                        vList.insert(i, ov1)
                    if ov2.checkV():
                        vList.insert(k, ov2)
                    break

                if d == 3 or d == 0:
                    ov1 = Vector(v1.x1, v1.y1, v2.x2, v2.y2, v1.index)
                    vList.remove(v1)
                    vList.remove(v2)
                    if ov1.checkV():
                        vList.insert(i, ov1)
                    break

        _list = FeatureLine2vList(vList)
        vl_num = len(vList)
        vList = FeatureLine(_list, vx, angleDiff, lengthdiff1 + 0.5, lengthdiff2)

    return vList


def AreaControl(iring, oring):
    ipolygon = ogr.Geometry(ogr.wkbPolygon)
    ipolygon.AddGeometry(iring)
    iArea = ipolygon.Area()
    ix = ipolygon.Centroid().GetX()
    iy = ipolygon.Centroid().GetY()

    opolygon = ogr.Geometry(ogr.wkbPolygon)
    opolygon.AddGeometry(oring)
    oArea = opolygon.Area()
    ox = opolygon.Centroid().GetX()
    oy = opolygon.Centroid().GetY()

    oidDis = ((ix - ox) ** 2 + (iy - oy) ** 2) ** 0.5
    if iArea == 0:
        return True
    if iArea > 1000 and (math.fabs(iArea - oArea) / iArea < 0.25):
        return True
    if (math.fabs(iArea - oArea) / iArea >= 0.25) or oidDis > 1:
        return False
    return True


def TopoControl(ipolygon, ifeatureOid, olayer):
    if ipolygon == None:
        return ipolygon
    buffergeom = ipolygon.Buffer(0.1)
    olayer.SetSpatialFilter(buffergeom)
    ofeat = olayer.GetNextFeature()

    while ofeat:
        ovalue = ofeat.GetFID()
        if ifeatureOid != ovalue:
            ogeom = ofeat.GetGeometryRef()
            diffGeom = ipolygon.Difference(ogeom)
            if diffGeom == None:
                return ipolygon
            ipolygon = diffGeom
        ofeat = olayer.GetNextFeature()

    return ipolygon


def SelfIntersection(ring):
    pts = Ring2Pts1(ring)
    oring = ogr.Geometry(ogr.wkbLinearRing)
    if len(pts) == 0:
        return ring, 0
    ls = LineString(pts)
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    if not lr.is_simple:
        mls = unary_union(lr)
        area = 0
        opoly = None
        for i in polygonize(mls):
            parea = i.area
            if area < parea:
                area = parea
                opoly = i

        if opoly == None:
            return ring, 0
        for x, y in opoly.exterior.coords:
            oring.AddPoint(x, y)
        oring.CloseRings()
        return oring, 1
    return ring, 0


def GetAttribute(feat):
    fieldCount = feat.GetFieldCount()
    attrList = []
    for i in range(fieldCount):
        attrList.append(feat.GetField(i))
    return attrList


def SetAttribute(feat, attrList):
    for i in range(len(attrList)):
        feat.SetField(i, attrList[i])


def CheckFieldExist(layerDefn, fieldName):
    for i in range(0, layerDefn.GetFieldCount()):
        if layerDefn.GetFieldDefn(i).GetNameRef() == fieldName:
            return True
    return False


def TranformPrj2(igeom):
    ogeom = ogr.Geometry(ogr.wkbPolygon)
    for i in range(igeom.GetGeometryCount()):
        iring = igeom.GetGeometryRef(i)
        oring = ogr.Geometry(ogr.wkbLinearRing)
        ptCount = iring.GetPointCount()
        for i in range(ptCount):
            if iring.GetX(i) < 180:
                x, y = latlon2utmxy(deg2rad(iring.GetX(i)), deg2rad(iring.GetY(i)), 50)
            else:
                x, y = iring.GetX(i), iring.GetY(i)
            oring.AddPoint(x, y)
        oring.CloseRings()
        ogeom.AddGeometry(oring)
    return ogeom


def TranformPrj(igeom, target_srs=None):
    """
    Transform the input geometry to the target projection using spatial reference metadata.
    If the input geometry does not have a spatial reference, WGS84 (EPSG:4326) is assumed.
    If target_srs is None, the geometry is returned unchanged (a clone is returned).

    Parameters:
        igeom (ogr.Geometry): The input geometry.
        target_srs (osr.SpatialReference, optional): The target spatial reference.
            This can be any projection; if omitted, no transformation is applied.

    Returns:
        ogr.Geometry: The transformed geometry (or a clone if no transformation is performed).
    """
    # Ensure the input geometry has a spatial reference; if not, assume WGS84.
    source_srs = igeom.GetSpatialReference()
    if source_srs is None:
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)
        igeom.AssignSpatialReference(source_srs)

    # If no target spatial reference is provided, return a clone of the original geometry.
    if target_srs is None:
        return igeom.Clone()

    # Create the coordinate transformation object.
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Clone and transform the geometry.
    geom_clone = igeom.Clone()
    geom_clone.Transform(transform)
    return geom_clone


def regularize(
    filepath, output, tempath=None, prj_flag=False, min_length=6, min_area=6
):
    """
    Regularize building footprints from a vector file.

    Args:
        filepath (str): Path to the input vector file.
        output (str): Path to the output vector file.
        tempath (str, optional): Path to the temporary directory. Defaults to None.
        prj_flag (bool, optional): Flag to indicate if projection transformation is needed. Defaults to False.
        min_length (int, optional): Minimum length for regularization. Defaults to 6.
        min_area (int, optional): Minimum area for regularization. Defaults to 6.

    Returns:
        None
    """
    ids = ReadVectorLayer(filepath)
    idsMessage = ReadVectorMessage(ids)
    ilayer = ids.GetLayer(0)
    ilayerDefn = ilayer.GetLayerDefn()
    ifeatureCount = ilayer.GetFeatureCount()
    delepolygon = ogr.Geometry(ogr.wkbPolygon)
    r = 0

    if tempath is None:
        tempath = os.path.join(os.path.dirname(output), "temp")
    if not os.path.exists(tempath):
        os.makedirs(tempath)

    for i in range(0, ifeatureCount):

        ifeature = ilayer.GetFeature(i)
        ifeatureOid = ifeature.GetFID()
        if r == 0 or r % 3000 == 0:
            npath = os.path.join(tempath, str(int(r / 3000)) + ".shp")
            ods = CreateVectorFile(npath, idsMessage)
            olayer = ods.GetLayer(0)
        r = r + 1

        attrList = GetAttribute(ifeature)

        igeom = ifeature.GetGeometryRef()
        if not prj_flag:
            igeom = TranformPrj(igeom)

        # aa = igeom.Area()
        if igeom.Area() < min_area or igeom is None:
            continue

        iringCount = igeom.GetGeometryCount()

        oupolygon = ogr.Geometry(ogr.wkbPolygon)

        for j in range(iringCount):
            new = 0
            iring = igeom.GetGeometryRef(j)
            delepolygon.AddGeometry(iring)
            iringpts = Ring2Pts(iring)

            if delepolygon.Area() < min_area:
                if oupolygon.GetGeometryCount() > 0:
                    new = 1
                delepolygon.Empty()
                continue
            delepolygon.Empty()

            ideleteIds = []
            ideleteIds = compress(
                iringpts[0], iringpts[len(iringpts) - 1], iringpts, ideleteIds
            )

            dringpts = []
            for k in iringpts:
                if k.index not in ideleteIds:
                    dringpts.append(k)

            if len(dringpts) == 3:
                if oupolygon.GetGeometryCount() > 0:
                    new = 1
                continue

            dpolygon, dring = Pts2Polygon(dringpts)

            if dpolygon.Boundary() == None:

                continue

            averageLength = (
                dpolygon.Boundary().Length()
                / dpolygon.GetGeometryRef(0).GetPointCount()
            )
            averageLength = averageLength * 1.5
            area = dpolygon.Area()

            try:
                RecSim, rering = RecSimilar(dpolygon)
            except:
                continue
            if RecSim == 1:
                oupolygon.AddGeometry(rering)
                oupolygon = TopoControl(oupolygon, ifeatureOid, olayer)
                new = 1
                continue

            if dring.GetPointCount() == 4:
                new = 1
                oupolygon.AddGeometry(dring)
                oupolygon = TopoControl(oupolygon, ifeatureOid, olayer)
                continue

            vList = Ring2VList(dring)

            try:
                vx = VListDiret(vList)
            except:
                continue
            svList = Smooth(vList, vx, 25, averageLength)

            sring = vList2Ring(svList)
            if not AreaControl(dring, sring):
                oupolygon.AddGeometry(dring)
                oupolygon = TopoControl(oupolygon, ifeatureOid, olayer)
                new = 1
                continue

            if vList2Ring(svList).GetPointCount() == 4:
                oupolygon.AddGeometry(sring)
                oupolygon = TopoControl(oupolygon, ifeatureOid, olayer)
                new = 1
                continue

            m = 0.5
            rring = sring
            while m <= min_length:
                svList = FeatureLine(svList, vx, 20, m, averageLength)

                if len(svList) < 4:
                    m = m + 10
                    continue

                svList = LocalResc(svList, vx, 20, m, averageLength, area)
                oring = vList2Ring2(svList)
                oring, selfInsect = SelfIntersection(oring)

                if m == 0.5:

                    if not AreaControl(sring, oring):
                        oupolygon.AddGeometry(sring)
                        new = 1
                        break
                    rring = oring
                else:
                    if not AreaControl(rring, oring):
                        oupolygon.AddGeometry(rring)
                        new = 1
                        break
                    rring = oring

                m = m + 0.5

            if new == 0:
                oring = vList2Ring2(svList)
                oring, selfInsect = SelfIntersection(oring)
                oupolygon.AddGeometry(oring)
                oupolygon = TopoControl(oupolygon, ifeatureOid, olayer)
                new = 1

        if new == 1:
            ofeature = ogr.Feature(olayer.GetLayerDefn())
            ofeature.SetGeometry(oupolygon)
            SetAttribute(ofeature, attrList)
            olayer.CreateFeature(ofeature)

    ods.Destroy()

    eds = CreateVectorFile(output, idsMessage)
    elayer = eds.GetLayer(0)
    ifeatureList = []
    files = os.listdir(tempath)
    for i in range(0, len(files)):
        if files[i].split(".")[1] != "shp":
            continue
        ipath = os.path.join(tempath, files[i])
        ifileDs = ReadVectorLayer(ipath)
        ifileLayer = ifileDs.GetLayer(0)
        ifeatureCount = ifileLayer.GetFeatureCount()
        for j in range(ifeatureCount):
            ifeature = ifileLayer.GetFeature(j)
            igeom = ifeature.GetGeometryRef()
            ogeom = ogr.Geometry(ogr.wkbPolygon)
            if igeom == None:
                print(0)
                j = j + 1
                continue
            for i in range(igeom.GetGeometryCount()):
                iring = igeom.GetGeometryRef(i)
                oring = ogr.Geometry(ogr.wkbLinearRing)
                ptCount = iring.GetPointCount()
                for i in range(ptCount):
                    if prj_flag:
                        x, y = utmxy2latlon(iring.GetX(i), iring.GetY(i), 50, False)
                        x, y = rad2deg(x), rad2deg(y)
                    else:
                        x, y = iring.GetX(i), iring.GetY(i)
                    oring.AddPoint(x, y)
                ogeom.AddGeometry(oring)
            ofeature = ogr.Feature(elayer.GetLayerDefn())
            ofeature.SetGeometry(ogeom)
            attrList = GetAttribute(ifeature)
            SetAttribute(ofeature, attrList)
            elayer.CreateFeature(ofeature)
            ofeature = None

        ifileDs.Destroy()
    eds.Destroy()

    shutil.rmtree(tempath)


if __name__ == "__main__":
    time_begin = time.time()
    input_path = "data/building_vector.geojson"
    result_path = "data/building_vector_result.gpkg"
    tempath = "data/temp"
    prjFlag = 0
    regularize(input_path, result_path, tempath, prjFlag)
    time_end = time.time()
    print("Time:%s" % time_interval(time_end, time_begin))
