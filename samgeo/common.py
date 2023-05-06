"""
The source code is adapted from https://github.com/aliaksandr960/segment-anything-eo. Credit to the author Aliaksandr Hancharenka.
"""

import os
import tempfile
import cv2
import numpy as np
import rasterio
from tqdm import tqdm

import shapely
import geopandas as gpd
from pyproj import Transformer, transform
import rasterio
from rasterio import features


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


def download_checkpoint(url=None, output=None, overwrite=False, **kwargs):
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
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
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
                    print("Downloaded image %d/%d" % (k, totalnum))

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
    with rasterio.open(src_fp) as src:
        features = rasterio.features.dataset_features(
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


def transform_coords(x, y, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x, y)


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
            rigth_x_bound = max(bound, x + width - raster_w)
            bottom_y_bound = max(bound, y + height - raster_h)

            blocks.append(
                {
                    "x": x,
                    "y": y,
                    "height": height,
                    "width": width,
                    "bounds": [[bound, bottom_y_bound], [bound, rigth_x_bound]],
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
                uin8_out = func(uint8_rgb_in)

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


def tiff_to_vector(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a gpkg file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the vector file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    with rasterio.open(tiff_path) as src:
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
    gdf.to_file(output, **kwargs)


def tiff_to_gpkg(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a gpkg file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the gpkg file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".gpkg"):
        output += ".gpkg"

    tiff_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def tiff_to_shp(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a shapefile.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the shapefile.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".shp"):
        output += ".shp"

    tiff_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def tiff_to_geojson(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a GeoJSON file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the GeoJSON file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".geojson"):
        output += ".geojson"

    tiff_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


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
    source, figsize=(20, 20), cmap=None, axis="off", fig_args={}, show_args={}, **kwargs
):
    import matplotlib.pyplot as plt

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


def overlay_images(
    image1,
    image2,
    opacity=0.5,
    backend="TkAgg",
    height_ratios=[10, 1],
    show_args1={},
    show_args2={},
):
    import sys
    import matplotlib
    import matplotlib.widgets as mpwidgets
    import matplotlib.pyplot as plt

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
    img1 = ax0.imshow(y, alpha=opacity, **show_args2)

    # Define the update function
    def update(value):
        img1.set_alpha(value)
        fig.canvas.draw_idle()

    # Create the slider
    slider0 = mpwidgets.Slider(
        ax=ax1, label="opacity", valmin=0, valmax=1, valinit=opacity
    )
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
    import matplotlib.pyplot as plt

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
            os.remove(filename)

        print("Package updated successfully.")

    except Exception as e:
        raise Exception(e)
