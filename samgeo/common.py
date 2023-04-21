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
        'sam_vit_h_4b8939.pth': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'sam_vit_l_0b3195.pth': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'sam_vit_b_01ec64.pth': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    if isinstance(url, str) and url in checkpoints:
        url = checkpoints[url]

    if url is None:
        url = checkpoints['sam_vit_h_4b8939.pth']

    if output is None:
        output = os.path.basename(url)

    return download_file(url, output,overwrite=overwrite, **kwargs)


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


def tms_to_geotiff(
    output,
    bbox,
    zoom=None,
    resolution=None,
    source="OpenStreetMap",
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
        print(f"The output file {output} already exists. Use `overwrite=True` to overwrite it.")
        return

    xyz_tiles = {
        "OPENSTREETMAP": {
            "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "OpenStreetMap",
            "name": "OpenStreetMap",
        },
        "ROADMAP": {
            "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Maps",
        },
        "SATELLITE": {
            "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        },
        "TERRAIN": {
            "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Terrain",
        },
        "HYBRID": {
            "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        },
    }

    if isinstance(source, str) and source.upper() in xyz_tiles:
        source = xyz_tiles[source.upper()]["url"]
    elif isinstance(source, str) and source.startswith("http"):
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
        if to_cog:
            image_to_cog(output, output)
    except Exception as e:
        raise Exception(e)


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
                    'x': x,
                    'y': y,
                    'height': height,
                    'width': width,
                    'bounds': [[bound, bottom_y_bound], [bound, rigth_x_bound]],
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
        rh, rw = profile['height'], profile['width']
        sh, sw = sample_size
        bound = bound

        resize_hw = sample_resize

        sample_grid = calculate_sample_grid(
            raster_h=rh, raster_w=rw, sample_h=sh, sample_w=sw, bound=bound
        )
        # set 1 channel uint8 output
        profile['count'] = 1
        profile['dtype'] = 'uint8'

        with rasterio.open(dst_fp, 'w', **profile) as dst:
            for b in tqdm(sample_grid):
                r = read_block(src, **b)

                uint8_rgb_in = data_to_rgb(r)
                orig_size = uint8_rgb_in.shape[:2]
                if resize_hw is not None:
                    uint8_rgb_in = cv2.resize(
                        uint8_rgb_in, resize_hw, interpolation=cv2.INTER_LINEAR
                    )

                # Do something
                uin8_out = func(uint8_rgb_in)

                if resize_hw is not None:
                    uin8_out = cv2.resize(
                        uin8_out, orig_size, interpolation=cv2.INTER_NEAREST
                    )
                # Zero channel, because
                write_block(dst, uin8_out, **b)


def image_to_image(image, func, sample_size=(384, 384), sample_resize=None, bound=128):
    with tempfile.NamedTemporaryFile() as src_tmpfile:
        s, b = cv2.imencode('.tif', image)
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
