"""
The source code is adapted from https://github.com/aliaksandr960/segment-anything-eo. Credit to the author Aliaksandr Hancharenka.
"""

import os
import tempfile
import cv2
import numpy as np
from tqdm import tqdm

import shapely
import pyproj
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

os.environ["GOOGLE_MAPS_API_KEY"] = "API-KEY"


def is_colab():
    """Tests if the code is being executed within Google Colab."""
    import sys

    if "google.colab" in sys.modules:
        return True
    else:
        return False


def check_file_path(file_path, make_dirs=True):
    """Gets the absolute file path.

    Args:
        file_path (str): The path to the file.
        make_dirs (bool, optional): Whether to create the directory if it does not exist. Defaults to True.

    Raises:
        FileNotFoundError: If the directory could not be found.
        TypeError: If the input directory path is not a string.

    Returns:
        str: The absolute path to the file.
    """
    if isinstance(file_path, str):
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        else:
            file_path = os.path.abspath(file_path)

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir) and make_dirs:
            os.makedirs(file_dir)

        return file_path

    else:
        raise TypeError("The provided file path must be a string.")


def temp_file_path(extension):
    """Returns a temporary file path.

    Args:
        extension (str): The file extension.

    Returns:
        str: The temporary file path.
    """

    import tempfile
    import uuid

    if not extension.startswith("."):
        extension = "." + extension
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{extension}")

    return file_path


def github_raw_url(url):
    """Get the raw URL for a GitHub file.

    Args:
        url (str): The GitHub URL.
    Returns:
        str: The raw URL.
    """
    if isinstance(url, str) and url.startswith("https://github.com/") and "blob" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "blob/", ""
        )
    return url


def download_file(
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
    unzip=True,
    overwrite=False,
    subfolder=False,
):
    """Download a file from URL, including Google Drive shared URL.

    Args:
        url (str, optional): Google Drive URL is also supported. Defaults to None.
        output (str, optional): Output filename. Default is basename of URL.
        quiet (bool, optional): Suppress terminal output. Default is False.
        proxy (str, optional): Proxy. Defaults to None.
        speed (float, optional): Download byte size per second (e.g., 256KB/s = 256 * 1024). Defaults to None.
        use_cookies (bool, optional): Flag to use cookies. Defaults to True.
        verify (bool | str, optional): Either a bool, in which case it controls whether the server's TLS certificate is verified, or a string,
            in which case it must be a path to a CA bundle to use. Default is True.. Defaults to True.
        id (str, optional): Google Drive's file ID. Defaults to None.
        fuzzy (bool, optional): Fuzzy extraction of Google Drive's file Id. Defaults to False.
        resume (bool, optional): Resume the download from existing tmp file if possible. Defaults to False.
        unzip (bool, optional): Unzip the file. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.
        subfolder (bool, optional): Create a subfolder with the same name as the file. Defaults to False.

    Returns:
        str: The output file path.
    """
    import zipfile

    try:
        import gdown
    except ImportError:
        print(
            "The gdown package is required for this function. Use `pip install gdown` to install it."
        )
        return

    if output is None:
        if isinstance(url, str) and url.startswith("http"):
            output = os.path.basename(url)

    out_dir = os.path.abspath(os.path.dirname(output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if isinstance(url, str):
        if os.path.exists(os.path.abspath(output)) and (not overwrite):
            print(
                f"{output} already exists. Skip downloading. Set overwrite=True to overwrite."
            )
            return os.path.abspath(output)
        else:
            url = github_raw_url(url)

    if "https://drive.google.com/file/d/" in url:
        fuzzy = True

    output = gdown.download(
        url, output, quiet, proxy, speed, use_cookies, verify, id, fuzzy, resume
    )

    if unzip and output.endswith(".zip"):
        with zipfile.ZipFile(output, "r") as zip_ref:
            if not quiet:
                print("Extracting files...")
            if subfolder:
                basename = os.path.splitext(os.path.basename(output))[0]

                output = os.path.join(out_dir, basename)
                if not os.path.exists(output):
                    os.makedirs(output)
                zip_ref.extractall(output)
            else:
                zip_ref.extractall(os.path.dirname(output))

    return os.path.abspath(output)


def download_checkpoint(model_type="vit_h", checkpoint_dir=None, hq=False):
    """Download the SAM model checkpoint.

    Args:
        model_type (str, optional): The model type. Can be one of ['vit_h', 'vit_l', 'vit_b'].
            Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
        checkpoint_dir (str, optional): The checkpoint_dir directory. Defaults to None, "~/.cache/torch/hub/checkpoints".
        hq (bool, optional): Whether to use HQ-SAM model (https://github.com/SysCV/sam-hq). Defaults to False.
    """

    if not hq:
        model_types = {
            "vit_h": {
                "name": "sam_vit_h_4b8939.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            },
            "vit_l": {
                "name": "sam_vit_l_0b3195.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            },
            "vit_b": {
                "name": "sam_vit_b_01ec64.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            },
        }
    else:
        model_types = {
            "vit_h": {
                "name": "sam_hq_vit_h.pth",
                "url": "https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing",
            },
            "vit_l": {
                "name": "sam_hq_vit_l.pth",
                "url": "https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view?usp=sharing",
            },
            "vit_b": {
                "name": "sam_hq_vit_b.pth",
                "url": "https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing",
            },
            "vit_tiny": {
                "name": "sam_hq_vit_tiny.pth",
                "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
            },
        }

    if model_type not in model_types:
        raise ValueError(
            f"Invalid model_type: {model_type}. It must be one of {', '.join(model_types)}"
        )

    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get(
            "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
        )

    checkpoint = os.path.join(checkpoint_dir, model_types[model_type]["name"])
    if not os.path.exists(checkpoint):
        print(f"Model checkpoint for {model_type} not found.")
        url = model_types[model_type]["url"]
        download_file(url, checkpoint)
    return checkpoint


def download_checkpoint_legacy(url=None, output=None, overwrite=False, **kwargs):
    """Download a checkpoint from URL. It can be one of the following: sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth, sam_vit_b_01ec64.pth.

    Args:
        url (str, optional): The checkpoint URL. Defaults to None.
        output (str, optional): The output file path. Defaults to None.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.

    Returns:
        str: The output file path.
    """
    checkpoints = {
        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    if isinstance(url, str) and url in checkpoints:
        url = checkpoints[url]

    if url is None:
        url = checkpoints["sam_vit_h_4b8939.pth"]

    if output is None:
        output = os.path.basename(url)

    return download_file(url, output, overwrite=overwrite, **kwargs)


def image_to_cog(source, dst_path=None, profile="deflate", **kwargs):
    """Converts an image to a COG file.

    Args:
        source (str): A dataset path, URL or rasterio.io.DatasetReader object.
        dst_path (str, optional): An output dataset path or or PathLike object. Defaults to None.
        profile (str, optional): COG profile. More at https://cogeotiff.github.io/rio-cogeo/profile. Defaults to "deflate".

    Raises:
        ImportError: If rio-cogeo is not installed.
        FileNotFoundError: If the source file could not be found.
    """
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

    except ImportError:
        raise ImportError(
            "The rio-cogeo package is not installed. Please install it with `pip install rio-cogeo` or `conda install rio-cogeo -c conda-forge`."
        )

    if not source.startswith("http"):
        source = check_file_path(source)

        if not os.path.exists(source):
            raise FileNotFoundError("The provided input file could not be found.")

    if dst_path is None:
        if not source.startswith("http"):
            dst_path = os.path.splitext(source)[0] + "_cog.tif"
        else:
            dst_path = temp_file_path(extension=".tif")

    dst_path = check_file_path(dst_path)

    dst_profile = cog_profiles.get(profile)
    cog_translate(source, dst_path, dst_profile, **kwargs)


def reproject(
    image, output, dst_crs="EPSG:4326", resampling="nearest", to_cog=True, **kwargs
):
    """Reprojects an image.

    Args:
        image (str): The input image filepath.
        output (str): The output image filepath.
        dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
        resampling (Resampling, optional): The resampling method. Defaults to "nearest".
        to_cog (bool, optional): Whether to convert the output image to a Cloud Optimized GeoTIFF. Defaults to True.
        **kwargs: Additional keyword arguments to pass to rasterio.open.

    """
    import rasterio as rio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    if isinstance(resampling, str):
        resampling = getattr(Resampling, resampling)

    image = os.path.abspath(image)
    output = os.path.abspath(output)

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    with rio.open(image, **kwargs) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rio.open(output, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    **kwargs,
                )

    if to_cog:
        image_to_cog(output, output)


def tms_to_geotiff(
    output,
    bbox,
    zoom=None,
    resolution=None,
    source="OpenStreetMap",
    crs="EPSG:3857",
    to_cog=False,
    return_image=False,
    overwrite=False,
    quiet=False,
    **kwargs,
):
    """Download TMS tiles and convert them to a GeoTIFF. The source is adapted from https://github.com/gumblex/tms2geotiff.
        Credits to the GitHub user @gumblex.

    Args:
        output (str): The output GeoTIFF file.
        bbox (list): The bounding box [minx, miny, maxx, maxy], e.g., [-122.5216, 37.733, -122.3661, 37.8095]
        zoom (int, optional): The map zoom level. Defaults to None.
        resolution (float, optional): The resolution in meters. Defaults to None.
        source (str, optional): The tile source. It can be one of the following: "OPENSTREETMAP", "ROADMAP",
            "SATELLITE", "TERRAIN", "HYBRID", or an HTTP URL. Defaults to "OpenStreetMap".
        crs (str, optional): The output CRS. Defaults to "EPSG:3857".
        to_cog (bool, optional): Convert to Cloud Optimized GeoTIFF. Defaults to False.
        return_image (bool, optional): Return the image as PIL.Image. Defaults to False.
        overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
        quiet (bool, optional): Suppress output. Defaults to False.
        **kwargs: Additional arguments to pass to gdal.GetDriverByName("GTiff").Create().

    """

    import os
    import io
    import math
    import itertools
    import concurrent.futures

    import numpy
    from PIL import Image

    try:
        from osgeo import gdal, osr
    except ImportError:
        raise ImportError("GDAL is not installed. Install it with pip install GDAL")

    try:
        import httpx

        SESSION = httpx.Client()
    except ImportError:
        import requests

        SESSION = requests.Session()

    if not overwrite and os.path.exists(output):
        print(
            f"The output file {output} already exists. Use `overwrite=True` to overwrite it."
        )
        return

    xyz_tiles = {
        "OPENSTREETMAP": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "ROADMAP": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "SATELLITE": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "TERRAIN": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        "HYBRID": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    }

    basemaps = get_basemaps()

    if isinstance(source, str):
        if source.upper() in xyz_tiles:
            source = xyz_tiles[source.upper()]
        elif source in basemaps:
            source = basemaps[source]
        elif source.startswith("http"):
            pass
    else:
        raise ValueError(
            'source must be one of "OpenStreetMap", "ROADMAP", "SATELLITE", "TERRAIN", "HYBRID", or a URL'
        )

    def resolution_to_zoom_level(resolution):
        """
        Convert map resolution in meters to zoom level for Web Mercator (EPSG:3857) tiles.
        """
        # Web Mercator tile size in meters at zoom level 0
        initial_resolution = 156543.03392804097

        # Calculate the zoom level
        zoom_level = math.log2(initial_resolution / resolution)

        return int(zoom_level)

    if isinstance(bbox, list) and len(bbox) == 4:
        west, south, east, north = bbox
    else:
        raise ValueError(
            "bbox must be a list of 4 coordinates in the format of [xmin, ymin, xmax, ymax]"
        )

    if zoom is None and resolution is None:
        raise ValueError("Either zoom or resolution must be provided")
    elif zoom is not None and resolution is not None:
        raise ValueError("Only one of zoom or resolution can be provided")

    if resolution is not None:
        zoom = resolution_to_zoom_level(resolution)

    EARTH_EQUATORIAL_RADIUS = 6378137.0

    Image.MAX_IMAGE_PIXELS = None

    gdal.UseExceptions()
    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)

    WKT_3857 = web_mercator.ExportToWkt()

    def from4326_to3857(lat, lon):
        xtile = math.radians(lon) * EARTH_EQUATORIAL_RADIUS
        ytile = (
            math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
        )
        return (xtile, ytile)

    def deg2num(lat, lon, zoom):
        lat_r = math.radians(lat)
        n = 2**zoom
        xtile = (lon + 180) / 360 * n
        ytile = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
        return (xtile, ytile)

    def is_empty(im):
        extrema = im.getextrema()
        if len(extrema) >= 3:
            if len(extrema) > 3 and extrema[-1] == (0, 0):
                return True
            for ext in extrema[:3]:
                if ext != (0, 0):
                    return False
            return True
        else:
            return extrema[0] == (0, 0)

    def paste_tile(bigim, base_size, tile, corner_xy, bbox):
        if tile is None:
            return bigim
        im = Image.open(io.BytesIO(tile))
        mode = "RGB" if im.mode == "RGB" else "RGBA"
        size = im.size
        if bigim is None:
            base_size[0] = size[0]
            base_size[1] = size[1]
            newim = Image.new(
                mode, (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1]))
            )
        else:
            newim = bigim

        dx = abs(corner_xy[0] - bbox[0])
        dy = abs(corner_xy[1] - bbox[1])
        xy0 = (size[0] * dx, size[1] * dy)
        if mode == "RGB":
            newim.paste(im, xy0)
        else:
            if im.mode != mode:
                im = im.convert(mode)
            if not is_empty(im):
                newim.paste(im, xy0)
        im.close()
        return newim

    def finish_picture(bigim, base_size, bbox, x0, y0, x1, y1):
        xfrac = x0 - bbox[0]
        yfrac = y0 - bbox[1]
        x2 = round(base_size[0] * xfrac)
        y2 = round(base_size[1] * yfrac)
        imgw = round(base_size[0] * (x1 - x0))
        imgh = round(base_size[1] * (y1 - y0))
        retim = bigim.crop((x2, y2, x2 + imgw, y2 + imgh))
        if retim.mode == "RGBA" and retim.getextrema()[3] == (255, 255):
            retim = retim.convert("RGB")
        bigim.close()
        return retim

    def get_tile(url):
        retry = 3
        while 1:
            try:
                r = SESSION.get(url, timeout=60)
                break
            except Exception:
                retry -= 1
                if not retry:
                    raise
        if r.status_code == 404:
            return None
        elif not r.content:
            return None
        r.raise_for_status()
        return r.content

    def draw_tile(
        source, lat0, lon0, lat1, lon1, zoom, filename, quiet=False, **kwargs
    ):
        x0, y0 = deg2num(lat0, lon0, zoom)
        x1, y1 = deg2num(lat1, lon1, zoom)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        corners = tuple(
            itertools.product(
                range(math.floor(x0), math.ceil(x1)),
                range(math.floor(y0), math.ceil(y1)),
            )
        )
        totalnum = len(corners)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            for x, y in corners:
                futures.append(
                    executor.submit(get_tile, source.format(z=zoom, x=x, y=y))
                )
            bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
            bigim = None
            base_size = [256, 256]
            for k, (fut, corner_xy) in enumerate(zip(futures, corners), 1):
                bigim = paste_tile(bigim, base_size, fut.result(), corner_xy, bbox)
                if not quiet:
                    print(
                        f"Downloaded image {str(k).zfill(len(str(totalnum)))}/{totalnum}"
                    )

        if not quiet:
            print("Saving GeoTIFF. Please wait...")
        img = finish_picture(bigim, base_size, bbox, x0, y0, x1, y1)
        imgbands = len(img.getbands())
        driver = gdal.GetDriverByName("GTiff")

        if "options" not in kwargs:
            kwargs["options"] = [
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "ZLEVEL=9",
                "TILED=YES",
            ]

        gtiff = driver.Create(
            filename,
            img.size[0],
            img.size[1],
            imgbands,
            gdal.GDT_Byte,
            **kwargs,
        )
        xp0, yp0 = from4326_to3857(lat0, lon0)
        xp1, yp1 = from4326_to3857(lat1, lon1)
        pwidth = abs(xp1 - xp0) / img.size[0]
        pheight = abs(yp1 - yp0) / img.size[1]
        gtiff.SetGeoTransform((min(xp0, xp1), pwidth, 0, max(yp0, yp1), 0, -pheight))
        gtiff.SetProjection(WKT_3857)
        for band in range(imgbands):
            array = numpy.array(img.getdata(band), dtype="u8")
            array = array.reshape((img.size[1], img.size[0]))
            band = gtiff.GetRasterBand(band + 1)
            band.WriteArray(array)
        gtiff.FlushCache()

        if not quiet:
            print(f"Image saved to {filename}")
        return img

    try:
        image = draw_tile(
            source, south, west, north, east, zoom, output, quiet, **kwargs
        )
        if return_image:
            return image
        if crs.upper() != "EPSG:3857":
            reproject(output, output, crs, to_cog=to_cog)
        elif to_cog:
            image_to_cog(output, output)
    except Exception as e:
        raise Exception(e)


def get_profile(src_fp):
    with rasterio.open(src_fp) as src:
        return src.profile


def get_crs(src_fp):
    with rasterio.open(src_fp) as src:
        return src.crs


def get_features(src_fp, bidx=1):
    from rasterio import features

    with rasterio.open(src_fp) as src:
        features = features.dataset_features(
            src,
            bidx=bidx,
            sampling=1,
            band=True,
            as_mask=False,
            with_nodata=False,
            geographic=True,
            precision=-1,
        )
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf.set_crs(src.crs)
        return gdf


def set_transform(geo_box, width, height):
    return rasterio.transform.from_bounds(*geo_box, width, height)


def transform_coords(x, y, src_crs, dst_crs, **kwargs):
    """Transform coordinates from one CRS to another.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.
        src_crs (str): The source CRS, e.g., "EPSG:4326".
        dst_crs (str): The destination CRS, e.g., "EPSG:3857".

    Returns:
        dict: The transformed coordinates in the format of (x, y)
    """
    transformer = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True, **kwargs
    )
    return transformer.transform(x, y)


def vector_to_geojson(filename, output=None, **kwargs):
    """Converts a vector file to a geojson file.

    Args:
        filename (str): The vector file path.
        output (str, optional): The output geojson file path. Defaults to None.

    Returns:
        dict: The geojson dictionary.
    """

    if not filename.startswith("http"):
        filename = download_file(filename)

    gdf = gpd.read_file(filename, **kwargs)
    if output is None:
        return gdf.__geo_interface__
    else:
        gdf.to_file(output, driver="GeoJSON")


def get_vector_crs(filename, **kwargs):
    """Gets the CRS of a vector file.

    Args:
        filename (str): The vector file path.

    Returns:
        str: The CRS of the vector file.
    """
    gdf = gpd.read_file(filename, **kwargs)
    epsg = gdf.crs.to_epsg()
    if epsg is None:
        return gdf.crs
    else:
        return f"EPSG:{epsg}"


def geojson_to_coords(
    geojson: str, src_crs: str = "epsg:4326", dst_crs: str = "epsg:4326"
) -> list:
    """Converts a geojson file or a dictionary of feature collection to a list of centroid coordinates.

    Args:
        geojson (str | dict): The geojson file path or a dictionary of feature collection.
        src_crs (str, optional): The source CRS. Defaults to "epsg:4326".
        dst_crs (str, optional): The destination CRS. Defaults to "epsg:4326".

    Returns:
        list: A list of centroid coordinates in the format of [[x1, y1], [x2, y2], ...]
    """

    import json
    import warnings

    warnings.filterwarnings("ignore")

    if isinstance(geojson, dict):
        geojson = json.dumps(geojson)
    gdf = gpd.read_file(geojson, driver="GeoJSON")
    centroids = gdf.geometry.centroid
    centroid_list = [[point.x, point.y] for point in centroids]
    if src_crs != dst_crs:
        centroid_list = transform_coords(
            [x[0] for x in centroid_list],
            [x[1] for x in centroid_list],
            src_crs,
            dst_crs,
        )
        centroid_list = [[x, y] for x, y in zip(centroid_list[0], centroid_list[1])]
    return centroid_list


def coords_to_xy(
    src_fp: str, coords: list, coord_crs: str = "epsg:4326", **kwargs
) -> list:
    """Converts a list of coordinates to pixel coordinates, i.e., (col, row) coordinates.

    Args:
        src_fp: The source raster file path.
        coords: A list of coordinates in the format of [[x1, y1], [x2, y2], ...]
        coord_crs: The coordinate CRS of the input coordinates. Defaults to "epsg:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.rowcol.

    Returns:
        A list of pixel coordinates in the format of [[x1, y1], [x2, y2], ...]
    """
    if isinstance(coords, np.ndarray):
        coords = coords.tolist()

    xs, ys = zip(*coords)
    with rasterio.open(src_fp) as src:
        width = src.width
        height = src.height
        if coord_crs != src.crs:
            xs, ys = transform_coords(xs, ys, coord_crs, src.crs, **kwargs)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys, **kwargs)
    result = [[col, row] for col, row in zip(cols, rows)]

    result = [
        [x, y] for x, y in result if x >= 0 and y >= 0 and x < width and y < height
    ]
    if len(result) == 0:
        print("No valid pixel coordinates found.")
    elif len(result) < len(coords):
        print("Some coordinates are out of the image boundary.")

    return result


def boxes_to_vector(coords, src_crs, dst_crs="EPSG:4326", output=None, **kwargs):
    """
    Convert a list of bounding box coordinates to vector data.

    Args:
        coords (list): A list of bounding box coordinates in the format [[left, top, right, bottom], [left, top, right, bottom], ...].
        src_crs (int or str): The EPSG code or proj4 string representing the source coordinate reference system (CRS) of the input coordinates.
        dst_crs (int or str, optional): The EPSG code or proj4 string representing the destination CRS to reproject the data (default is "EPSG:4326").
        output (str or None, optional): The full file path (including the directory and filename without the extension) where the vector data should be saved.
                                       If None (default), the function returns the GeoDataFrame without saving it to a file.
        **kwargs: Additional keyword arguments to pass to geopandas.GeoDataFrame.to_file() when saving the vector data.

    Returns:
        geopandas.GeoDataFrame or None: The GeoDataFrame with the converted vector data if output is None, otherwise None if the data is saved to a file.
    """

    from shapely.geometry import box

    # Create a list of Shapely Polygon objects based on the provided coordinates
    polygons = [box(*coord) for coord in coords]

    # Create a GeoDataFrame with the Shapely Polygon objects
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=src_crs)

    # Reproject the GeoDataFrame to the specified EPSG code
    gdf_reprojected = gdf.to_crs(dst_crs)

    if output is not None:
        gdf_reprojected.to_file(output, **kwargs)
    else:
        return gdf_reprojected


def rowcol_to_xy(
    src_fp,
    rows=None,
    cols=None,
    boxes=None,
    zs=None,
    offset="center",
    output=None,
    dst_crs="EPSG:4326",
    **kwargs,
):
    """Converts a list of (row, col) coordinates to (x, y) coordinates.

    Args:
        src_fp (str): The source raster file path.
        rows (list, optional): A list of row coordinates. Defaults to None.
        cols (list, optional): A list of col coordinates. Defaults to None.
        boxes (list, optional): A list of (row, col) coordinates in the format of [[left, top, right, bottom], [left, top, right, bottom], ...]
        zs: zs (list or float, optional): Height associated with coordinates. Primarily used for RPC based coordinate transformations.
        offset (str, optional): Determines if the returned coordinates are for the center of the pixel or for a corner.
        output (str, optional): The output vector file path. Defaults to None.
        dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.xy.

    Returns:
        A list of (x, y) coordinates.
    """

    if boxes is not None:
        rows = []
        cols = []

        for box in boxes:
            rows.append(box[1])
            rows.append(box[3])
            cols.append(box[0])
            cols.append(box[2])

    if rows is None or cols is None:
        raise ValueError("rows and cols must be provided.")

    with rasterio.open(src_fp) as src:
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, zs, offset, **kwargs)
        src_crs = src.crs

    if boxes is None:
        return [[x, y] for x, y in zip(xs, ys)]
    else:
        result = [[xs[i], ys[i + 1], xs[i + 1], ys[i]] for i in range(0, len(xs), 2)]

        if output is not None:
            boxes_to_vector(result, src_crs, dst_crs, output)
        else:
            return result


def bbox_to_xy(
    src_fp: str, coords: list, coord_crs: str = "epsg:4326", **kwargs
) -> list:
    """Converts a list of coordinates to pixel coordinates, i.e., (col, row) coordinates.
        Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
        While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright

    Args:
        src_fp (str): The source raster file path.
        coords (list): A list of coordinates in the format of [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...]
        coord_crs (str, optional): The coordinate CRS of the input coordinates. Defaults to "epsg:4326".

    Returns:
        list: A list of pixel coordinates in the format of [[minx, maxy, maxx, miny], ...] from top left to bottom right.
    """

    if isinstance(coords, str):
        gdf = gpd.read_file(coords)
        coords = gdf.geometry.bounds.values.tolist()
        if gdf.crs is not None:
            coord_crs = f"epsg:{gdf.crs.to_epsg()}"
    elif isinstance(coords, np.ndarray):
        coords = coords.tolist()
    if isinstance(coords, dict):
        import json

        geojson = json.dumps(coords)
        gdf = gpd.read_file(geojson, driver="GeoJSON")
        coords = gdf.geometry.bounds.values.tolist()

    elif not isinstance(coords, list):
        raise ValueError("coords must be a list of coordinates.")

    if not isinstance(coords[0], list):
        coords = [coords]

    new_coords = []

    with rasterio.open(src_fp) as src:
        width = src.width
        height = src.height

        for coord in coords:
            minx, miny, maxx, maxy = coord

            if coord_crs != src.crs:
                minx, miny = transform_coords(minx, miny, coord_crs, src.crs, **kwargs)
                maxx, maxy = transform_coords(maxx, maxy, coord_crs, src.crs, **kwargs)

                rows1, cols1 = rasterio.transform.rowcol(
                    src.transform, minx, miny, **kwargs
                )
                rows2, cols2 = rasterio.transform.rowcol(
                    src.transform, maxx, maxy, **kwargs
                )

                new_coords.append([cols1, rows1, cols2, rows2])

            else:
                new_coords.append([minx, miny, maxx, maxy])

    result = []

    for coord in new_coords:
        minx, miny, maxx, maxy = coord

        if (
            minx >= 0
            and miny >= 0
            and maxx >= 0
            and maxy >= 0
            and minx < width
            and miny < height
            and maxx < width
            and maxy < height
        ):
            # Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
            # While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright
            result.append([minx, maxy, maxx, miny])

    if len(result) == 0:
        print("No valid pixel coordinates found.")
        return None
    elif len(result) == 1:
        return result[0]
    elif len(result) < len(coords):
        print("Some coordinates are out of the image boundary.")

    return result


def geojson_to_xy(
    src_fp: str, geojson: str, coord_crs: str = "epsg:4326", **kwargs
) -> list:
    """Converts a geojson file or a dictionary of feature collection to a list of pixel coordinates.

    Args:
        src_fp: The source raster file path.
        geojson: The geojson file path or a dictionary of feature collection.
        coord_crs: The coordinate CRS of the input coordinates. Defaults to "epsg:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.rowcol.

    Returns:
        A list of pixel coordinates in the format of [[x1, y1], [x2, y2], ...]
    """
    with rasterio.open(src_fp) as src:
        src_crs = src.crs
    coords = geojson_to_coords(geojson, coord_crs, src_crs)
    return coords_to_xy(src_fp, coords, src_crs, **kwargs)


def get_pixel_coords(src_fp, xs, ys):
    with rasterio.open(src_fp) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        box = np.array([min(cols), min(rows), max(cols), max(rows)])
    return box


def write_features(gdf, dst_fp):
    gdf.to_file(dst_fp)


def write_raster(dst_fp, dst_arr, profile, width, height, transform, crs):
    profile.update({"driver": "GTiff", "nodata": "0"})
    with rasterio.open(dst_fp, "w", **profile) as dst:
        if len(dst_arr.shape) == 2:
            dst_arr = dst_arr[np.newaxis, ...]
        for i in range(dst_arr.shape[0]):
            dst.write(dst_arr[i], i + 1)


def chw_to_hwc(block):
    # Grab first 3 channels
    block = block[:3, ...]
    # CHW to HWC
    block = np.transpose(block, (1, 2, 0))
    return block


def hwc_to_hw(block, channel=0):
    # Grab first 3 channels
    block = block[..., channel].astype(np.uint8)
    return block


def calculate_sample_grid(raster_h, raster_w, sample_h, sample_w, bound):
    h, w = sample_h, sample_w
    blocks = []
    height = h + 2 * bound
    width = w + 2 * bound

    for y in range(-bound, raster_h, h):
        for x in range(-bound, raster_w, w):
            right_x_bound = max(bound, x + width - raster_w)
            bottom_y_bound = max(bound, y + height - raster_h)

            blocks.append(
                {
                    "x": x,
                    "y": y,
                    "height": height,
                    "width": width,
                    "bounds": [[bound, bottom_y_bound], [bound, right_x_bound]],
                }
            )
    return blocks


def read_block(src, x, y, height, width, nodata=0, **kwargs):
    return src.read(
        window=((y, y + height), (x, x + width)), boundless=True, fill_value=nodata
    )


def write_block(dst, raster, y, x, height, width, bounds=None):
    if bounds:
        raster = raster[
            bounds[0][0] : raster.shape[0] - bounds[0][1],
            bounds[1][0] : raster.shape[1] - bounds[1][1],
        ]
        x += bounds[1][0]
        y += bounds[0][0]
        width = width - bounds[1][1] - bounds[1][0]
        height = height - bounds[0][1] - bounds[0][0]
    dst.write(raster, 1, window=((y, y + height), (x, x + width)))


def tiff_to_tiff(
    src_fp,
    dst_fp,
    func,
    data_to_rgb=chw_to_hwc,
    sample_size=(512, 512),
    sample_resize=None,
    bound=128,
    foreground=True,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
    **kwargs,
):
    with rasterio.open(src_fp) as src:
        profile = src.profile

        # Computer blocks
        rh, rw = profile["height"], profile["width"]
        sh, sw = sample_size
        bound = bound

        resize_hw = sample_resize

        # Subdivide image into tiles
        sample_grid = calculate_sample_grid(
            raster_h=rh, raster_w=rw, sample_h=sh, sample_w=sw, bound=bound
        )
        # set 1 channel uint8 output
        profile["count"] = 1
        profile["dtype"] = "uint8"

        if erosion_kernel is not None:
            erosion_kernel = np.ones(erosion_kernel, np.uint8)

        with rasterio.open(dst_fp, "w", **profile) as dst:
            for b in tqdm(sample_grid):
                # Read each tile from the source
                r = read_block(src, **b)
                # Extract the first 3 channels as RGB
                uint8_rgb_in = data_to_rgb(r)
                orig_size = uint8_rgb_in.shape[:2]
                if resize_hw is not None:
                    uint8_rgb_in = cv2.resize(
                        uint8_rgb_in, resize_hw, interpolation=cv2.INTER_LINEAR
                    )

                # Run the model, call the __call__ method of SamGeo class
                uin8_out = func(
                    uint8_rgb_in,
                    foreground=foreground,
                    erosion_kernel=erosion_kernel,
                    mask_multiplier=mask_multiplier,
                    **kwargs,
                )

                if resize_hw is not None:
                    uin8_out = cv2.resize(
                        uin8_out, orig_size, interpolation=cv2.INTER_NEAREST
                    )
                # Write the output to the destination
                write_block(dst, uin8_out, **b)


def image_to_image(image, func, sample_size=(384, 384), sample_resize=None, bound=128):
    with tempfile.NamedTemporaryFile() as src_tmpfile:
        s, b = cv2.imencode(".tif", image)
        src_tmpfile.write(b.tobytes())
        src_fp = src_tmpfile.name
        with tempfile.NamedTemporaryFile() as dst_tmpfile:
            dst_fp = dst_tmpfile.name
            tiff_to_tiff(
                src_fp,
                dst_fp,
                func,
                data_to_rgb=chw_to_hwc,
                sample_size=sample_size,
                sample_resize=sample_resize,
                bound=bound,
            )

            result = cv2.imread(dst_fp)
    return result[..., 0]


def tiff_to_image(
    src_fp,
    func,
    data_to_rgb=chw_to_hwc,
    sample_size=(512, 512),
    sample_resize=None,
    bound=128,
):
    with tempfile.NamedTemporaryFile() as dst_tmpfile:
        dst_fp = dst_tmpfile.name
        tiff_to_tiff(
            src_fp,
            dst_fp,
            func,
            data_to_rgb=data_to_rgb,
            sample_size=sample_size,
            sample_resize=sample_resize,
            bound=bound,
        )

        result = cv2.imread(dst_fp)
    return result[..., 0]


def tiff_to_shapes(tiff_path, simplify_tolerance=None):
    from rasterio import features

    with rasterio.open(tiff_path) as src:
        band = src.read()

        mask = band != 0
        shapes = features.shapes(band, mask=mask, transform=src.transform)
    result = [shapely.geometry.shape(shape) for shape, _ in shapes]
    if simplify_tolerance is not None:
        result = [shape.simplify(tolerance=simplify_tolerance) for shape in result]
    return result


def draw_tile(source, lat0, lon0, lat1, lon1, zoom, filename, **kwargs):
    bbox = [lon0, lat0, lon1, lat1]
    image = tms_to_geotiff(
        filename,
        bbox,
        zoom=zoom,
        resolution=None,
        source=source,
        to_cog=False,
        return_image=True,
        quiet=False,
        **kwargs,
    )
    return image


def raster_to_vector(source, output, simplify_tolerance=None, dst_crs=None, **kwargs):
    """Vectorize a raster dataset.

    Args:
        source (str): The path to the tiff file.
        output (str): The path to the vector file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """
    from rasterio import features

    with rasterio.open(source) as src:
        band = src.read()

        mask = band != 0
        shapes = features.shapes(band, mask=mask, transform=src.transform)

    fc = [
        {"geometry": shapely.geometry.shape(shape), "properties": {"value": value}}
        for shape, value in shapes
    ]
    if simplify_tolerance is not None:
        for i in fc:
            i["geometry"] = i["geometry"].simplify(tolerance=simplify_tolerance)

    gdf = gpd.GeoDataFrame.from_features(fc)
    if src.crs is not None:
        gdf.set_crs(crs=src.crs, inplace=True)

    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)

    gdf.to_file(output, **kwargs)


def raster_to_gpkg(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a gpkg file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the gpkg file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".gpkg"):
        output += ".gpkg"

    raster_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def raster_to_shp(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a shapefile.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the shapefile.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".shp"):
        output += ".shp"

    raster_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def raster_to_geojson(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a GeoJSON file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the GeoJSON file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".geojson"):
        output += ".geojson"

    raster_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def get_xyz_dict(free_only=True):
    """Returns a dictionary of xyz services.

    Args:
        free_only (bool, optional): Whether to return only free xyz tile services that do not require an access token. Defaults to True.

    Returns:
        dict: A dictionary of xyz services.
    """
    import collections
    import xyzservices.providers as xyz

    def _unpack_sub_parameters(var, param):
        temp = var
        for sub_param in param.split("."):
            temp = getattr(temp, sub_param)
        return temp

    xyz_dict = {}
    for item in xyz.values():
        try:
            name = item["name"]
            tile = _unpack_sub_parameters(xyz, name)
            if _unpack_sub_parameters(xyz, name).requires_token():
                if free_only:
                    pass
                else:
                    xyz_dict[name] = tile
            else:
                xyz_dict[name] = tile

        except Exception:
            for sub_item in item:
                name = item[sub_item]["name"]
                tile = _unpack_sub_parameters(xyz, name)
                if _unpack_sub_parameters(xyz, name).requires_token():
                    if free_only:
                        pass
                    else:
                        xyz_dict[name] = tile
                else:
                    xyz_dict[name] = tile

    xyz_dict = collections.OrderedDict(sorted(xyz_dict.items()))
    return xyz_dict


def get_basemaps(free_only=True):
    """Returns a dictionary of xyz basemaps.

    Args:
        free_only (bool, optional): Whether to return only free xyz tile services that do not require an access token. Defaults to True.

    Returns:
        dict: A dictionary of xyz basemaps.
    """

    basemaps = {}
    xyz_dict = get_xyz_dict(free_only=free_only)
    for item in xyz_dict:
        name = xyz_dict[item].name
        url = xyz_dict[item].build_url()
        basemaps[name] = url

    return basemaps


def array_to_image(
    array, output, source=None, dtype=None, compress="deflate", **kwargs
):
    """Save a NumPy array as a GeoTIFF using the projection information from an existing GeoTIFF file.

    Args:
        array (np.ndarray): The NumPy array to be saved as a GeoTIFF.
        output (str): The path to the output image.
        source (str, optional): The path to an existing GeoTIFF file with map projection information. Defaults to None.
        dtype (np.dtype, optional): The data type of the output array. Defaults to None.
        compress (str, optional): The compression method. Can be one of the following: "deflate", "lzw", "packbits", "jpeg". Defaults to "deflate".
    """

    from PIL import Image

    if isinstance(array, str) and os.path.exists(array):
        array = cv2.imread(array)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    if output.endswith(".tif") and source is not None:
        with rasterio.open(source) as src:
            crs = src.crs
            transform = src.transform
            if compress is None:
                compress = src.compression

        # Determine the minimum and maximum values in the array

        min_value = np.min(array)
        max_value = np.max(array)

        if dtype is None:
            # Determine the best dtype for the array
            if min_value >= 0 and max_value <= 1:
                dtype = np.float32
            elif min_value >= 0 and max_value <= 255:
                dtype = np.uint8
            elif min_value >= -128 and max_value <= 127:
                dtype = np.int8
            elif min_value >= 0 and max_value <= 65535:
                dtype = np.uint16
            elif min_value >= -32768 and max_value <= 32767:
                dtype = np.int16
            else:
                dtype = np.float64

        # Convert the array to the best dtype
        array = array.astype(dtype)

        # Define the GeoTIFF metadata
        if array.ndim == 2:
            metadata = {
                "driver": "GTiff",
                "height": array.shape[0],
                "width": array.shape[1],
                "count": 1,
                "dtype": array.dtype,
                "crs": crs,
                "transform": transform,
            }
        elif array.ndim == 3:
            metadata = {
                "driver": "GTiff",
                "height": array.shape[0],
                "width": array.shape[1],
                "count": array.shape[2],
                "dtype": array.dtype,
                "crs": crs,
                "transform": transform,
            }

        if compress is not None:
            metadata["compress"] = compress
        else:
            raise ValueError("Array must be 2D or 3D.")

        # Create a new GeoTIFF file and write the array to it
        with rasterio.open(output, "w", **metadata) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            elif array.ndim == 3:
                for i in range(array.shape[2]):
                    dst.write(array[:, :, i], i + 1)

    else:
        img = Image.fromarray(array)
        img.save(output, **kwargs)


def show_image(
    source, figsize=(12, 10), cmap=None, axis="off", fig_args={}, show_args={}, **kwargs
):
    if isinstance(source, str):
        if source.startswith("http"):
            source = download_file(source)

        if not os.path.exists(source):
            raise ValueError(f"Input path {source} does not exist.")

        source = cv2.imread(source)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize, **fig_args)
    plt.imshow(source, cmap=cmap, **show_args)
    plt.axis(axis)
    plt.show(**kwargs)


def show_mask(mask, random_color=False):
    ax = plt.gca()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(
    image,
    coords,
    labels,
    marker_size=375,
    figsize=(12, 10),
    axis="on",
    title=None,
    mask=None,
    **kwargs,
):
    if isinstance(image, str):
        if image.startswith("http"):
            image = download_file(image)

        if not os.path.exists(image):
            raise ValueError(f"Input path {image} does not exist.")

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    if title is not None:
        plt.title(title)
    plt.axis(axis)
    plt.show()


def show_box(box, ax):
    ax = plt.gca()
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def overlay_images(
    image1,
    image2,
    alpha=0.5,
    backend="TkAgg",
    height_ratios=[10, 1],
    show_args1={},
    show_args2={},
):
    """Overlays two images using a slider to control the opacity of the top image.

    Args:
        image1 (str | np.ndarray): The first input image at the bottom represented as a NumPy array or the path to the image.
        image2 (_type_): The second input image on top represented as a NumPy array or the path to the image.
        alpha (float, optional): The alpha value of the top image. Defaults to 0.5.
        backend (str, optional): The backend of the matplotlib plot. Defaults to "TkAgg".
        height_ratios (list, optional): The height ratios of the two subplots. Defaults to [10, 1].
        show_args1 (dict, optional): The keyword arguments to pass to the imshow() function for the first image. Defaults to {}.
        show_args2 (dict, optional): The keyword arguments to pass to the imshow() function for the second image. Defaults to {}.

    """
    import sys
    import matplotlib
    import matplotlib.widgets as mpwidgets

    if "google.colab" in sys.modules:
        backend = "inline"
        print(
            "The TkAgg backend is not supported in Google Colab. The overlay_images function will not work on Colab."
        )
        return

    matplotlib.use(backend)

    if isinstance(image1, str):
        if image1.startswith("http"):
            image1 = download_file(image1)

        if not os.path.exists(image1):
            raise ValueError(f"Input path {image1} does not exist.")

    if isinstance(image2, str):
        if image2.startswith("http"):
            image2 = download_file(image2)

        if not os.path.exists(image2):
            raise ValueError(f"Input path {image2} does not exist.")

    # Load the two images
    x = plt.imread(image1)
    y = plt.imread(image2)

    # Create the plot
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": height_ratios})
    img0 = ax0.imshow(x, **show_args1)
    img1 = ax0.imshow(y, alpha=alpha, **show_args2)

    # Define the update function
    def update(value):
        img1.set_alpha(value)
        fig.canvas.draw_idle()

    # Create the slider
    slider0 = mpwidgets.Slider(ax=ax1, label="alpha", valmin=0, valmax=1, valinit=alpha)
    slider0.on_changed(update)

    # Display the plot
    plt.show()


def blend_images(
    img1,
    img2,
    alpha=0.5,
    output=False,
    show=True,
    figsize=(12, 10),
    axis="off",
    **kwargs,
):
    """
    Blends two images together using the addWeighted function from the OpenCV library.

    Args:
        img1 (numpy.ndarray): The first input image on top represented as a NumPy array.
        img2 (numpy.ndarray): The second input image at the bottom represented as a NumPy array.
        alpha (float): The weighting factor for the first image in the blend. By default, this is set to 0.5.
        output (str, optional): The path to the output image. Defaults to False.
        show (bool, optional): Whether to display the blended image. Defaults to True.
        figsize (tuple, optional): The size of the figure. Defaults to (12, 10).
        axis (str, optional): The axis of the figure. Defaults to "off".
        **kwargs: Additional keyword arguments to pass to the cv2.addWeighted() function.

    Returns:
        numpy.ndarray: The blended image as a NumPy array.
    """
    # Resize the images to have the same dimensions
    if isinstance(img1, str):
        if img1.startswith("http"):
            img1 = download_file(img1)

        if not os.path.exists(img1):
            raise ValueError(f"Input path {img1} does not exist.")

        img1 = cv2.imread(img1)

    if isinstance(img2, str):
        if img2.startswith("http"):
            img2 = download_file(img2)

        if not os.path.exists(img2):
            raise ValueError(f"Input path {img2} does not exist.")

        img2 = cv2.imread(img2)

    if img1.dtype == np.float32:
        img1 = (img1 * 255).astype(np.uint8)

    if img2.dtype == np.float32:
        img2 = (img2 * 255).astype(np.uint8)

    if img1.dtype != img2.dtype:
        img2 = img2.astype(img1.dtype)

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Blend the images using the addWeighted function
    beta = 1 - alpha
    blend_img = cv2.addWeighted(img1, alpha, img2, beta, 0, **kwargs)

    if output:
        array_to_image(blend_img, output, img2)

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(blend_img)
        plt.axis(axis)
        plt.show()
    else:
        return blend_img


def update_package(out_dir=None, keep=False, **kwargs):
    """Updates the package from the GitHub repository without the need to use pip or conda.

    Args:
        out_dir (str, optional): The output directory. Defaults to None.
        keep (bool, optional): Whether to keep the downloaded package. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the download_file() function.
    """

    import shutil

    try:
        if out_dir is None:
            out_dir = os.getcwd()
        url = (
            "https://github.com/opengeos/segment-geospatial/archive/refs/heads/main.zip"
        )
        filename = "segment-geospatial-main.zip"
        download_file(url, filename, **kwargs)

        pkg_dir = os.path.join(out_dir, "segment-geospatial-main")
        work_dir = os.getcwd()
        os.chdir(pkg_dir)

        if shutil.which("pip") is None:
            cmd = "pip3 install ."
        else:
            cmd = "pip install ."

        os.system(cmd)
        os.chdir(work_dir)

        if not keep:
            shutil.rmtree(pkg_dir)
            try:
                os.remove(filename)
            except:
                pass

        print("Package updated successfully.")

    except Exception as e:
        raise Exception(e)


def sam_map_gui(sam, basemap="SATELLITE", repeat_mode=True, out_dir=None, **kwargs):
    """Display the SAM Map GUI.

    Args:
        sam (SamGeo):
        basemap (str, optional): The basemap to use. Defaults to "SATELLITE".
        repeat_mode (bool, optional): Whether to use the repeat mode for the draw control. Defaults to True.
        out_dir (str, optional): The output directory. Defaults to None.

    """
    try:
        import shutil
        import tempfile
        import leafmap
        import ipyleaflet
        import ipyevents
        import ipywidgets as widgets
        from ipyfilechooser import FileChooser
    except ImportError:
        raise ImportError(
            "The sam_map function requires the leafmap package. Please install it first."
        )

    if out_dir is None:
        out_dir = tempfile.gettempdir()

    m = leafmap.Map(repeat_mode=repeat_mode, **kwargs)
    m.default_style = {"cursor": "crosshair"}
    m.add_basemap(basemap, show=False)

    # Skip the image layer if localtileserver is not available
    try:
        m.add_raster(sam.source, layer_name="Image")
    except:
        pass

    m.fg_markers = []
    m.bg_markers = []

    fg_layer = ipyleaflet.LayerGroup(layers=m.fg_markers, name="Foreground")
    bg_layer = ipyleaflet.LayerGroup(layers=m.bg_markers, name="Background")
    m.add(fg_layer)
    m.add(bg_layer)
    m.fg_layer = fg_layer
    m.bg_layer = bg_layer

    widget_width = "280px"
    button_width = "90px"
    padding = "0px 0px 0px 4px"  # upper, right, bottom, left
    style = {"description_width": "initial"}

    toolbar_button = widgets.ToggleButton(
        value=True,
        tooltip="Toolbar",
        icon="gear",
        layout=widgets.Layout(width="28px", height="28px", padding=padding),
    )

    close_button = widgets.ToggleButton(
        value=False,
        tooltip="Close the tool",
        icon="times",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding=padding),
    )

    plus_button = widgets.ToggleButton(
        value=False,
        tooltip="Load foreground points",
        icon="plus-circle",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding=padding),
    )

    minus_button = widgets.ToggleButton(
        value=False,
        tooltip="Load background points",
        icon="minus-circle",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding=padding),
    )

    radio_buttons = widgets.RadioButtons(
        options=["Foreground", "Background"],
        description="Class Type:",
        disabled=False,
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    fg_count = widgets.IntText(
        value=0,
        description="Foreground #:",
        disabled=True,
        style=style,
        layout=widgets.Layout(width="135px", padding=padding),
    )
    bg_count = widgets.IntText(
        value=0,
        description="Background #:",
        disabled=True,
        style=style,
        layout=widgets.Layout(width="135px", padding=padding),
    )

    segment_button = widgets.ToggleButton(
        description="Segment",
        value=False,
        button_style="primary",
        layout=widgets.Layout(padding=padding),
    )

    save_button = widgets.ToggleButton(
        description="Save", value=False, button_style="primary"
    )

    reset_button = widgets.ToggleButton(
        description="Reset", value=False, button_style="primary"
    )
    segment_button.layout.width = button_width
    save_button.layout.width = button_width
    reset_button.layout.width = button_width

    opacity_slider = widgets.FloatSlider(
        description="Mask opacity:",
        min=0,
        max=1,
        value=0.5,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    rectangular = widgets.Checkbox(
        value=False,
        description="Regularize",
        layout=widgets.Layout(width="130px", padding=padding),
        style=style,
    )

    colorpicker = widgets.ColorPicker(
        concise=False,
        description="Color",
        value="#ffff00",
        layout=widgets.Layout(width="140px", padding=padding),
        style=style,
    )

    buttons = widgets.VBox(
        [
            radio_buttons,
            widgets.HBox([fg_count, bg_count]),
            opacity_slider,
            widgets.HBox([rectangular, colorpicker]),
            widgets.HBox(
                [segment_button, save_button, reset_button],
                layout=widgets.Layout(padding="0px 4px 0px 4px"),
            ),
        ]
    )

    def opacity_changed(change):
        if change["new"]:
            mask_layer = m.find_layer("Masks")
            if mask_layer is not None:
                mask_layer.interact(opacity=opacity_slider.value)

    opacity_slider.observe(opacity_changed, "value")

    output = widgets.Output(
        layout=widgets.Layout(
            width=widget_width, padding=padding, max_width=widget_width
        )
    )

    toolbar_header = widgets.HBox()
    toolbar_header.children = [close_button, plus_button, minus_button, toolbar_button]
    toolbar_footer = widgets.VBox()
    toolbar_footer.children = [
        buttons,
        output,
    ]
    toolbar_widget = widgets.VBox()
    toolbar_widget.children = [toolbar_header, toolbar_footer]

    toolbar_event = ipyevents.Event(
        source=toolbar_widget, watched_events=["mouseenter", "mouseleave"]
    )

    def marker_callback(chooser):
        with output:
            if chooser.selected is not None:
                try:
                    gdf = gpd.read_file(chooser.selected)
                    centroids = gdf.centroid
                    coords = [[point.x, point.y] for point in centroids]
                    for coord in coords:
                        if plus_button.value:
                            if is_colab():  # Colab does not support AwesomeIcon
                                marker = ipyleaflet.CircleMarker(
                                    location=(coord[1], coord[0]),
                                    radius=2,
                                    color="green",
                                    fill_color="green",
                                )
                            else:
                                marker = ipyleaflet.Marker(
                                    location=[coord[1], coord[0]],
                                    icon=ipyleaflet.AwesomeIcon(
                                        name="plus-circle",
                                        marker_color="green",
                                        icon_color="darkred",
                                    ),
                                )
                            m.fg_layer.add(marker)
                            m.fg_markers.append(marker)
                            fg_count.value = len(m.fg_markers)
                        elif minus_button.value:
                            if is_colab():
                                marker = ipyleaflet.CircleMarker(
                                    location=(coord[1], coord[0]),
                                    radius=2,
                                    color="red",
                                    fill_color="red",
                                )
                            else:
                                marker = ipyleaflet.Marker(
                                    location=[coord[1], coord[0]],
                                    icon=ipyleaflet.AwesomeIcon(
                                        name="minus-circle",
                                        marker_color="red",
                                        icon_color="darkred",
                                    ),
                                )
                            m.bg_layer.add(marker)
                            m.bg_markers.append(marker)
                            bg_count.value = len(m.bg_markers)

                except Exception as e:
                    print(e)

            if m.marker_control in m.controls:
                m.remove_control(m.marker_control)
                delattr(m, "marker_control")

            plus_button.value = False
            minus_button.value = False

    def marker_button_click(change):
        if change["new"]:
            sandbox_path = os.environ.get("SANDBOX_PATH")
            filechooser = FileChooser(
                path=os.getcwd(),
                sandbox_path=sandbox_path,
                layout=widgets.Layout(width="454px"),
            )
            filechooser.use_dir_icons = True
            filechooser.filter_pattern = ["*.shp", "*.geojson", "*.gpkg"]
            filechooser.register_callback(marker_callback)
            marker_control = ipyleaflet.WidgetControl(
                widget=filechooser, position="topright"
            )
            m.add_control(marker_control)
            m.marker_control = marker_control
        else:
            if hasattr(m, "marker_control") and m.marker_control in m.controls:
                m.remove_control(m.marker_control)
                m.marker_control.close()

    plus_button.observe(marker_button_click, "value")
    minus_button.observe(marker_button_click, "value")

    def handle_toolbar_event(event):
        if event["type"] == "mouseenter":
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        elif event["type"] == "mouseleave":
            if not toolbar_button.value:
                toolbar_widget.children = [toolbar_button]
                toolbar_button.value = False
                close_button.value = False

    toolbar_event.on_dom_event(handle_toolbar_event)

    def toolbar_btn_click(change):
        if change["new"]:
            close_button.value = False
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        else:
            if not close_button.value:
                toolbar_widget.children = [toolbar_button]

    toolbar_button.observe(toolbar_btn_click, "value")

    def close_btn_click(change):
        if change["new"]:
            toolbar_button.value = False
            if m.toolbar_control in m.controls:
                m.remove_control(m.toolbar_control)
            toolbar_widget.close()

    close_button.observe(close_btn_click, "value")

    def handle_map_interaction(**kwargs):
        try:
            if kwargs.get("type") == "click":
                latlon = kwargs.get("coordinates")
                if radio_buttons.value == "Foreground":
                    if is_colab():
                        marker = ipyleaflet.CircleMarker(
                            location=tuple(latlon),
                            radius=2,
                            color="green",
                            fill_color="green",
                        )
                    else:
                        marker = ipyleaflet.Marker(
                            location=latlon,
                            icon=ipyleaflet.AwesomeIcon(
                                name="plus-circle",
                                marker_color="green",
                                icon_color="darkred",
                            ),
                        )
                    fg_layer.add(marker)
                    m.fg_markers.append(marker)
                    fg_count.value = len(m.fg_markers)
                elif radio_buttons.value == "Background":
                    if is_colab():
                        marker = ipyleaflet.CircleMarker(
                            location=tuple(latlon),
                            radius=2,
                            color="red",
                            fill_color="red",
                        )
                    else:
                        marker = ipyleaflet.Marker(
                            location=latlon,
                            icon=ipyleaflet.AwesomeIcon(
                                name="minus-circle",
                                marker_color="red",
                                icon_color="darkred",
                            ),
                        )
                    bg_layer.add(marker)
                    m.bg_markers.append(marker)
                    bg_count.value = len(m.bg_markers)

        except (TypeError, KeyError) as e:
            print(f"Error handling map interaction: {e}")

    m.on_interaction(handle_map_interaction)

    def segment_button_click(change):
        if change["new"]:
            segment_button.value = False
            with output:
                output.clear_output()
                if len(m.fg_markers) == 0:
                    print("Please add some foreground markers.")
                    segment_button.value = False
                    return

                else:
                    try:
                        fg_points = [
                            [marker.location[1], marker.location[0]]
                            for marker in m.fg_markers
                        ]
                        bg_points = [
                            [marker.location[1], marker.location[0]]
                            for marker in m.bg_markers
                        ]
                        point_coords = fg_points + bg_points
                        point_labels = [1] * len(fg_points) + [0] * len(bg_points)

                        filename = f"masks_{random_string()}.tif"
                        filename = os.path.join(out_dir, filename)
                        sam.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            point_crs="EPSG:4326",
                            output=filename,
                        )
                        if m.find_layer("Masks") is not None:
                            m.remove_layer(m.find_layer("Masks"))
                        if m.find_layer("Regularized") is not None:
                            m.remove_layer(m.find_layer("Regularized"))

                        if hasattr(sam, "prediction_fp") and os.path.exists(
                            sam.prediction_fp
                        ):
                            try:
                                os.remove(sam.prediction_fp)
                            except:
                                pass

                        # Skip the image layer if localtileserver is not available
                        try:
                            m.add_raster(
                                filename,
                                nodata=0,
                                cmap="Blues",
                                opacity=opacity_slider.value,
                                layer_name="Masks",
                                zoom_to_layer=False,
                            )

                            if rectangular.value:
                                vector = filename.replace(".tif", ".gpkg")
                                vector_rec = filename.replace(".tif", "_rect.gpkg")
                                raster_to_vector(filename, vector)
                                regularize(vector, vector_rec)
                                vector_style = {"color": colorpicker.value}
                                m.add_vector(
                                    vector_rec,
                                    layer_name="Regularized",
                                    style=vector_style,
                                    info_mode=None,
                                    zoom_to_layer=False,
                                )

                        except:
                            pass
                        output.clear_output()
                        segment_button.value = False
                        sam.prediction_fp = filename
                    except Exception as e:
                        segment_button.value = False
                        print(e)

    segment_button.observe(segment_button_click, "value")

    def filechooser_callback(chooser):
        with output:
            if chooser.selected is not None:
                try:
                    filename = chooser.selected
                    shutil.copy(sam.prediction_fp, filename)
                    vector = filename.replace(".tif", ".gpkg")
                    raster_to_vector(filename, vector)
                    if rectangular.value:
                        vector_rec = filename.replace(".tif", "_rect.gpkg")
                        regularize(vector, vector_rec)

                    fg_points = [
                        [marker.location[1], marker.location[0]]
                        for marker in m.fg_markers
                    ]
                    bg_points = [
                        [marker.location[1], marker.location[0]]
                        for marker in m.bg_markers
                    ]

                    coords_to_geojson(
                        fg_points, filename.replace(".tif", "_fg_markers.geojson")
                    )
                    coords_to_geojson(
                        bg_points, filename.replace(".tif", "_bg_markers.geojson")
                    )

                except Exception as e:
                    print(e)

                if hasattr(m, "save_control") and m.save_control in m.controls:
                    m.remove_control(m.save_control)
                    delattr(m, "save_control")
                save_button.value = False

    def save_button_click(change):
        if change["new"]:
            with output:
                sandbox_path = os.environ.get("SANDBOX_PATH")
                filechooser = FileChooser(
                    path=os.getcwd(),
                    filename="masks.tif",
                    sandbox_path=sandbox_path,
                    layout=widgets.Layout(width="454px"),
                )
                filechooser.use_dir_icons = True
                filechooser.filter_pattern = ["*.tif"]
                filechooser.register_callback(filechooser_callback)
                save_control = ipyleaflet.WidgetControl(
                    widget=filechooser, position="topright"
                )
                m.add_control(save_control)
                m.save_control = save_control
        else:
            if hasattr(m, "save_control") and m.save_control in m.controls:
                m.remove_control(m.save_control)
                delattr(m, "save_control")

    save_button.observe(save_button_click, "value")

    def reset_button_click(change):
        if change["new"]:
            segment_button.value = False
            reset_button.value = False
            opacity_slider.value = 0.5
            rectangular.value = False
            colorpicker.value = "#ffff00"
            output.clear_output()
            try:
                m.remove_layer(m.find_layer("Masks"))
                if m.find_layer("Regularized") is not None:
                    m.remove_layer(m.find_layer("Regularized"))
                m.clear_drawings()
                if hasattr(m, "fg_markers"):
                    m.user_rois = None
                    m.fg_markers = []
                    m.bg_markers = []
                    m.fg_layer.clear_layers()
                    m.bg_layer.clear_layers()
                    fg_count.value = 0
                    bg_count.value = 0
                try:
                    os.remove(sam.prediction_fp)
                except:
                    pass
            except:
                pass

    reset_button.observe(reset_button_click, "value")

    toolbar_control = ipyleaflet.WidgetControl(
        widget=toolbar_widget, position="topright"
    )
    m.add_control(toolbar_control)
    m.toolbar_control = toolbar_control

    return m


def random_string(string_length=6):
    """Generates a random string of fixed length.

    Args:
        string_length (int, optional): Fixed length. Defaults to 3.

    Returns:
        str: A random string
    """
    import random
    import string

    # random.seed(1001)
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))


def coords_to_geojson(coords, output=None):
    """Convert a list of coordinates (lon, lat) to a GeoJSON string or file.

    Args:
        coords (list): A list of coordinates (lon, lat).
        output (str, optional): The output file path. Defaults to None.

    Returns:
        dict: A GeoJSON dictionary.
    """

    import json

    if len(coords) == 0:
        return
    # Create a GeoJSON FeatureCollection object
    feature_collection = {"type": "FeatureCollection", "features": []}

    # Iterate through the coordinates list and create a GeoJSON Feature object for each coordinate
    for coord in coords:
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": coord},
            "properties": {},
        }
        feature_collection["features"].append(feature)

    # Convert the FeatureCollection object to a JSON string
    geojson_str = json.dumps(feature_collection)

    if output is not None:
        with open(output, "w") as f:
            f.write(geojson_str)
    else:
        return geojson_str


def show_canvas(image, fg_color=(0, 255, 0), bg_color=(0, 0, 255), radius=5):
    """Show a canvas to collect foreground and background points.

    Args:
        image (str | np.ndarray): The input image.
        fg_color (tuple, optional): The color for the foreground points. Defaults to (0, 255, 0).
        bg_color (tuple, optional): The color for the background points. Defaults to (0, 0, 255).
        radius (int, optional): The radius of the points. Defaults to 5.

    Returns:
        tuple: A tuple of two lists of foreground and background points.
    """
    if isinstance(image, str):
        if image.startswith("http"):
            image = download_file(image)

        image = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("Input image must be a URL or a NumPy array.")

    # Create an empty list to store the mouse click coordinates
    left_clicks = []
    right_clicks = []

    # Create a mouse callback function
    def get_mouse_coordinates(event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append the coordinates to the mouse_clicks list
            left_clicks.append((x, y))

            # Draw a green circle at the mouse click coordinates
            cv2.circle(image, (x, y), radius, fg_color, -1)

            # Show the updated image with the circle
            cv2.imshow("Image", image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Append the coordinates to the mouse_clicks list
            right_clicks.append((x, y))

            # Draw a red circle at the mouse click coordinates
            cv2.circle(image, (x, y), radius, bg_color, -1)

            # Show the updated image with the circle
            cv2.imshow("Image", image)

    # Create a window to display the image
    cv2.namedWindow("Image")

    # Set the mouse callback function for the window
    cv2.setMouseCallback("Image", get_mouse_coordinates)

    # Display the image in the window
    cv2.imshow("Image", image)

    # Wait for a key press to exit
    cv2.waitKey(0)

    # Destroy the window
    cv2.destroyAllWindows()

    return left_clicks, right_clicks


def install_package(package):
    """Install a Python package.

    Args:
        package (str | list): The package name or a GitHub URL or a list of package names or GitHub URLs.
    """
    import subprocess

    if isinstance(package, str):
        packages = [package]

    for package in packages:
        if package.startswith("https://github.com"):
            package = f"git+{package}"

        # Execute pip install command and show output in real-time
        command = f"pip install {package}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.decode("utf-8").strip())

        # Wait for process to complete
        process.wait()


def text_sam_gui(
    sam,
    basemap="SATELLITE",
    out_dir=None,
    box_threshold=0.25,
    text_threshold=0.25,
    cmap="viridis",
    opacity=0.5,
    **kwargs,
):
    """Display the SAM Map GUI.

    Args:
        sam (SamGeo):
        basemap (str, optional): The basemap to use. Defaults to "SATELLITE".
        out_dir (str, optional): The output directory. Defaults to None.

    """
    try:
        import shutil
        import tempfile
        import leafmap
        import ipyleaflet
        import ipyevents
        import ipywidgets as widgets
        import leafmap.colormaps as cm
        from ipyfilechooser import FileChooser
    except ImportError:
        raise ImportError(
            "The sam_map function requires the leafmap package. Please install it first."
        )

    if out_dir is None:
        out_dir = tempfile.gettempdir()

    m = leafmap.Map(**kwargs)
    m.default_style = {"cursor": "crosshair"}
    m.add_basemap(basemap, show=False)

    # Skip the image layer if localtileserver is not available
    try:
        m.add_raster(sam.source, layer_name="Image")
    except:
        pass

    widget_width = "280px"
    button_width = "90px"
    padding = "0px 4px 0px 4px"  # upper, right, bottom, left
    style = {"description_width": "initial"}

    toolbar_button = widgets.ToggleButton(
        value=True,
        tooltip="Toolbar",
        icon="gear",
        layout=widgets.Layout(width="28px", height="28px", padding="0px 0px 0px 4px"),
    )

    close_button = widgets.ToggleButton(
        value=False,
        tooltip="Close the tool",
        icon="times",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding="0px 0px 0px 4px"),
    )

    text_prompt = widgets.Text(
        description="Text prompt:",
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    box_slider = widgets.FloatSlider(
        description="Box threshold:",
        min=0,
        max=1,
        value=box_threshold,
        step=0.01,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    text_slider = widgets.FloatSlider(
        description="Text threshold:",
        min=0,
        max=1,
        step=0.01,
        value=text_threshold,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    cmap_dropdown = widgets.Dropdown(
        description="Palette:",
        options=cm.list_colormaps(),
        value=cmap,
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    opacity_slider = widgets.FloatSlider(
        description="Opacity:",
        min=0,
        max=1,
        value=opacity,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    def opacity_changed(change):
        if change["new"]:
            if hasattr(m, "layer_name"):
                mask_layer = m.find_layer(m.layer_name)
                if mask_layer is not None:
                    mask_layer.interact(opacity=opacity_slider.value)

    opacity_slider.observe(opacity_changed, "value")

    rectangular = widgets.Checkbox(
        value=False,
        description="Regularize",
        layout=widgets.Layout(width="130px", padding=padding),
        style=style,
    )

    colorpicker = widgets.ColorPicker(
        concise=False,
        description="Color",
        value="#ffff00",
        layout=widgets.Layout(width="140px", padding=padding),
        style=style,
    )

    segment_button = widgets.ToggleButton(
        description="Segment",
        value=False,
        button_style="primary",
        layout=widgets.Layout(padding=padding),
    )

    save_button = widgets.ToggleButton(
        description="Save", value=False, button_style="primary"
    )

    reset_button = widgets.ToggleButton(
        description="Reset", value=False, button_style="primary"
    )
    segment_button.layout.width = button_width
    save_button.layout.width = button_width
    reset_button.layout.width = button_width

    output = widgets.Output(
        layout=widgets.Layout(
            width=widget_width, padding=padding, max_width=widget_width
        )
    )

    toolbar_header = widgets.HBox()
    toolbar_header.children = [close_button, toolbar_button]
    toolbar_footer = widgets.VBox()
    toolbar_footer.children = [
        text_prompt,
        box_slider,
        text_slider,
        cmap_dropdown,
        opacity_slider,
        widgets.HBox([rectangular, colorpicker]),
        widgets.HBox(
            [segment_button, save_button, reset_button],
            layout=widgets.Layout(padding="0px 4px 0px 4px"),
        ),
        output,
    ]
    toolbar_widget = widgets.VBox()
    toolbar_widget.children = [toolbar_header, toolbar_footer]

    toolbar_event = ipyevents.Event(
        source=toolbar_widget, watched_events=["mouseenter", "mouseleave"]
    )

    def handle_toolbar_event(event):
        if event["type"] == "mouseenter":
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        elif event["type"] == "mouseleave":
            if not toolbar_button.value:
                toolbar_widget.children = [toolbar_button]
                toolbar_button.value = False
                close_button.value = False

    toolbar_event.on_dom_event(handle_toolbar_event)

    def toolbar_btn_click(change):
        if change["new"]:
            close_button.value = False
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        else:
            if not close_button.value:
                toolbar_widget.children = [toolbar_button]

    toolbar_button.observe(toolbar_btn_click, "value")

    def close_btn_click(change):
        if change["new"]:
            toolbar_button.value = False
            if m.toolbar_control in m.controls:
                m.remove_control(m.toolbar_control)
            toolbar_widget.close()

    close_button.observe(close_btn_click, "value")

    def segment_button_click(change):
        if change["new"]:
            segment_button.value = False
            with output:
                output.clear_output()
                if len(text_prompt.value) == 0:
                    print("Please enter a text prompt first.")
                elif sam.source is None:
                    print("Please run sam.set_image() first.")
                else:
                    print("Segmenting...")
                    layer_name = text_prompt.value.replace(" ", "_")
                    filename = os.path.join(
                        out_dir, f"{layer_name}_{random_string()}.tif"
                    )
                    try:
                        sam.predict(
                            sam.source,
                            text_prompt.value,
                            box_slider.value,
                            text_slider.value,
                            output=filename,
                        )
                        sam.output = filename
                        if m.find_layer(layer_name) is not None:
                            m.remove_layer(m.find_layer(layer_name))
                        if m.find_layer(f"{layer_name}_rect") is not None:
                            m.remove_layer(m.find_layer(f"{layer_name} Regularized"))
                    except Exception as e:
                        output.clear_output()
                        print(e)
                    if os.path.exists(filename):
                        try:
                            m.add_raster(
                                filename,
                                layer_name=layer_name,
                                palette=cmap_dropdown.value,
                                opacity=opacity_slider.value,
                                nodata=0,
                                zoom_to_layer=False,
                            )
                            m.layer_name = layer_name

                            if rectangular.value:
                                vector = filename.replace(".tif", ".gpkg")
                                vector_rec = filename.replace(".tif", "_rect.gpkg")
                                raster_to_vector(filename, vector)
                                regularize(vector, vector_rec)
                                vector_style = {"color": colorpicker.value}
                                m.add_vector(
                                    vector_rec,
                                    layer_name=f"{layer_name} Regularized",
                                    style=vector_style,
                                    info_mode=None,
                                    zoom_to_layer=False,
                                )

                            output.clear_output()
                        except Exception as e:
                            print(e)

    segment_button.observe(segment_button_click, "value")

    def filechooser_callback(chooser):
        with output:
            if chooser.selected is not None:
                try:
                    filename = chooser.selected
                    shutil.copy(sam.output, filename)
                    vector = filename.replace(".tif", ".gpkg")
                    raster_to_vector(filename, vector)
                    if rectangular.value:
                        vector_rec = filename.replace(".tif", "_rect.gpkg")
                        regularize(vector, vector_rec)
                except Exception as e:
                    print(e)

                if hasattr(m, "save_control") and m.save_control in m.controls:
                    m.remove_control(m.save_control)
                    delattr(m, "save_control")
                save_button.value = False

    def save_button_click(change):
        if change["new"]:
            with output:
                output.clear_output()
                if not hasattr(m, "layer_name"):
                    print("Please click the Segment button first.")
                else:
                    sandbox_path = os.environ.get("SANDBOX_PATH")
                    filechooser = FileChooser(
                        path=os.getcwd(),
                        filename=f"{m.layer_name}.tif",
                        sandbox_path=sandbox_path,
                        layout=widgets.Layout(width="454px"),
                    )
                    filechooser.use_dir_icons = True
                    filechooser.filter_pattern = ["*.tif"]
                    filechooser.register_callback(filechooser_callback)
                    save_control = ipyleaflet.WidgetControl(
                        widget=filechooser, position="topright"
                    )
                    m.add_control(save_control)
                    m.save_control = save_control

        else:
            if hasattr(m, "save_control") and m.save_control in m.controls:
                m.remove_control(m.save_control)
                delattr(m, "save_control")

    save_button.observe(save_button_click, "value")

    def reset_button_click(change):
        if change["new"]:
            segment_button.value = False
            save_button.value = False
            reset_button.value = False
            opacity_slider.value = 0.5
            box_slider.value = 0.25
            text_slider.value = 0.25
            cmap_dropdown.value = "viridis"
            text_prompt.value = ""
            output.clear_output()
            try:
                if hasattr(m, "layer_name") and m.find_layer(m.layer_name) is not None:
                    m.remove_layer(m.find_layer(m.layer_name))
                m.clear_drawings()
            except:
                pass

    reset_button.observe(reset_button_click, "value")

    toolbar_control = ipyleaflet.WidgetControl(
        widget=toolbar_widget, position="topright"
    )
    m.add_control(toolbar_control)
    m.toolbar_control = toolbar_control

    return m


def regularize(source, output=None, crs="EPSG:4326", **kwargs):
    """Regularize a polygon GeoDataFrame.

    Args:
        source (str | gpd.GeoDataFrame): The input file path or a GeoDataFrame.
        output (str, optional): The output file path. Defaults to None.


    Returns:
        gpd.GeoDataFrame: The output GeoDataFrame.
    """
    if isinstance(source, str):
        gdf = gpd.read_file(source)
    elif isinstance(source, gpd.GeoDataFrame):
        gdf = source
    else:
        raise ValueError("The input source must be a GeoDataFrame or a file path.")

    polygons = gdf.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    result = gpd.GeoDataFrame(geometry=polygons, data=gdf.drop("geometry", axis=1))

    if crs is not None:
        result.to_crs(crs, inplace=True)
    if output is not None:
        result.to_file(output, **kwargs)
    else:
        return result


def split_raster(filename, out_dir, tile_size=256, overlap=0):
    """Split a raster into tiles.

    Args:
        filename (str): The path or http URL to the raster file.
        out_dir (str): The path to the output directory.
        tile_size (int | tuple, optional): The size of the tiles. Can be an integer or a tuple of (width, height). Defaults to 256.
        overlap (int, optional): The number of pixels to overlap between tiles. Defaults to 0.

    Raises:
        ImportError: Raised if GDAL is not installed.
    """

    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError(
            "GDAL is required to use this function. Install it with `conda install gdal -c conda-forge`"
        )

    if isinstance(filename, str):
        if filename.startswith("http"):
            output = filename.split("/")[-1]
            download_file(filename, output)
            filename = output

    # Open the input GeoTIFF file
    ds = gdal.Open(filename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if isinstance(tile_size, int):
        tile_width = tile_size
        tile_height = tile_size
    elif isinstance(tile_size, tuple):
        tile_width = tile_size[0]
        tile_height = tile_size[1]

    # Get the size of the input raster
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Calculate the number of tiles needed in both directions, taking into account the overlap
    num_tiles_x = (width - overlap) // (tile_width - overlap) + int(
        (width - overlap) % (tile_width - overlap) > 0
    )
    num_tiles_y = (height - overlap) // (tile_height - overlap) + int(
        (height - overlap) % (tile_height - overlap) > 0
    )

    # Get the georeferencing information of the input raster
    geotransform = ds.GetGeoTransform()

    # Loop over all the tiles
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate the pixel coordinates of the tile, taking into account the overlap and clamping to the edge of the raster
            x_min = i * (tile_width - overlap)
            y_min = j * (tile_height - overlap)
            x_max = min(x_min + tile_width, width)
            y_max = min(y_min + tile_height, height)

            # Adjust the size of the last tile in each row and column to include any remaining pixels
            if i == num_tiles_x - 1:
                x_min = max(x_max - tile_width, 0)
            if j == num_tiles_y - 1:
                y_min = max(y_max - tile_height, 0)

            # Calculate the size of the tile, taking into account the overlap
            tile_width = x_max - x_min
            tile_height = y_max - y_min

            # Set the output file name
            output_file = f"{out_dir}/tile_{i}_{j}.tif"

            # Create a new dataset for the tile
            driver = gdal.GetDriverByName("GTiff")
            tile_ds = driver.Create(
                output_file,
                tile_width,
                tile_height,
                ds.RasterCount,
                ds.GetRasterBand(1).DataType,
            )

            # Calculate the georeferencing information for the output tile
            tile_geotransform = (
                geotransform[0] + x_min * geotransform[1],
                geotransform[1],
                0,
                geotransform[3] + y_min * geotransform[5],
                0,
                geotransform[5],
            )

            # Set the geotransform and projection of the tile
            tile_ds.SetGeoTransform(tile_geotransform)
            tile_ds.SetProjection(ds.GetProjection())

            # Read the data from the input raster band(s) and write it to the tile band(s)
            for k in range(ds.RasterCount):
                band = ds.GetRasterBand(k + 1)
                tile_band = tile_ds.GetRasterBand(k + 1)
                tile_data = band.ReadAsArray(x_min, y_min, tile_width, tile_height)
                tile_band.WriteArray(tile_data)

            # Close the tile dataset
            tile_ds = None

    # Close the input dataset
    ds = None


def merge_rasters(
    input_dir,
    output,
    input_pattern="*.tif",
    output_format="GTiff",
    output_nodata=None,
    output_options=["COMPRESS=DEFLATE"],
):
    """Merge a directory of rasters into a single raster.

    Args:
        input_dir (str): The path to the input directory.
        output (str): The path to the output raster.
        input_pattern (str, optional): The pattern to match the input files. Defaults to "*.tif".
        output_format (str, optional): The output format. Defaults to "GTiff".
        output_nodata (float, optional): The output nodata value. Defaults to None.
        output_options (list, optional): A list of output options. Defaults to ["COMPRESS=DEFLATE"].

    Raises:
        ImportError: Raised if GDAL is not installed.
    """

    import glob

    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError(
            "GDAL is required to use this function. Install it with `conda install gdal -c conda-forge`"
        )
    # Get a list of all the input files
    input_files = glob.glob(os.path.join(input_dir, input_pattern))

    # Merge the input files into a single output file
    gdal.Warp(
        output,
        input_files,
        format=output_format,
        dstNodata=output_nodata,
        options=output_options,
    )
