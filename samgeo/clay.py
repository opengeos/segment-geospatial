"""
Clay foundation model wrapper for geospatial embeddings.

This module provides a wrapper around the Clay foundation model for generating
rich spectral embeddings from geospatial imagery. It integrates with the 
segment-geospatial library's raster I/O infrastructure.
"""

import os
import math
import datetime
import numpy as np
import torch
import cv2
import rasterio
import warnings
from typing import Optional, Union, Tuple, Dict, List, Any
from pathlib import Path

try:
    from claymodel.model import ClayMAEModule
    from claymodel.utils import posemb_sincos_2d_with_gsd
    from torchvision.transforms import v2
    import yaml
    from box import Box
    CLAY_AVAILABLE = True
except ImportError:
    CLAY_AVAILABLE = False

from .common import (
    check_file_path,
    download_file,
    transform_coords,
    reproject,
)


# Default metadata for common sensors
DEFAULT_METADATA = {
    'sentinel-2-l2a': {
        'band_order': ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'swir16', 'swir22'],
        'rgb_indices': [2, 1, 0],
        'gsd': 10,
        'bands': {
            'mean': {'blue': 1105., 'green': 1355., 'red': 1552., 'rededge1': 1887., 'rededge2': 2422., 'rededge3': 2630., 'nir': 2743., 'nir08': 2785., 'swir16': 2388., 'swir22': 1835.},
            'std': {'blue': 1809., 'green': 1757., 'red': 1888., 'rededge1': 1870., 'rededge2': 1732., 'rededge3': 1697., 'nir': 1742., 'nir08': 1648., 'swir16': 1470., 'swir22': 1379.},
            'wavelength': {'blue': 0.493, 'green': 0.56, 'red': 0.665, 'rededge1': 0.704, 'rededge2': 0.74, 'rededge3': 0.783, 'nir': 0.842, 'nir08': 0.865, 'swir16': 1.61, 'swir22': 2.19}
        }
    },
    'landsat-c2l2-sr': {
        'band_order': ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22'],
        'rgb_indices': [0, 1, 2],
        'gsd': 30,
        'bands': {
            'mean': {'red': 13705., 'green': 13310., 'blue': 12474., 'nir08': 17801., 'swir16': 14615., 'swir22': 12701.},
            'std': {'red': 9578., 'green': 9408., 'blue': 10144., 'nir08': 8277., 'swir16': 5300., 'swir22': 4522.},
            'wavelength': {'red': 0.65, 'green': 0.56, 'blue': 0.48, 'nir08': 0.86, 'swir16': 1.6, 'swir22': 2.2}
        }
    },
    'naip': {
        'band_order': ['red', 'green', 'blue', 'nir'],
        'rgb_indices': [0, 1, 2],
        'gsd': 1.0,
        'bands': {
            'mean': {'red': 110.16, 'green': 115.41, 'blue': 98.15, 'nir': 139.04},
            'std': {'red': 47.23, 'green': 39.82, 'blue': 35.43, 'nir': 49.86},
            'wavelength': {'red': 0.65, 'green': 0.56, 'blue': 0.48, 'nir': 0.842}
        }
    }
}


def normalize_timestamp(date):
    """Normalize timestamp to week and hour components for Clay model."""
    if isinstance(date, str):
        date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00'))
    elif not isinstance(date, datetime.datetime):
        date = datetime.datetime.now()
    
    # Get day of year and hour
    day_of_year = date.timetuple().tm_yday
    hour = date.hour
    
    # Normalize to [-1, 1] range
    week_norm = 2 * (day_of_year - 1) / 365 - 1
    hour_norm = 2 * hour / 24 - 1
    
    return [week_norm, hour_norm]


def normalize_latlon(lat: float, lon: float) -> Tuple[List[float], List[float]]:
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180
    
    lat_norm = [math.sin(lat_rad), math.cos(lat_rad)]
    lon_norm = [math.sin(lon_rad), math.cos(lon_rad)]
    
    return lat_norm, lon_norm


class Clay:
    """
    Clay foundation model wrapper for generating geospatial embeddings.
    
    This class provides an interface to generate rich spectral embeddings from
    geospatial imagery using the Clay foundation model.
    """
    
    
    
    def __init__(
        self,
        checkpoint_path: str,
        metadata_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize Clay embeddings model.
        
        Args:
            checkpoint_path: Path to Clay model checkpoint
            metadata_path: Path to Clay metadata YAML file (optional)
            device: Device to run model on ('auto', 'cuda', 'cpu')
            mask_ratio: Masking ratio for model (0.0 for inference)
            shuffle: Whether to shuffle patches
        """
        if not CLAY_AVAILABLE:
            raise ImportError(
                "Clay model dependencies not available. "
                "Please install: pip install claymodel torch torchvision pyyaml python-box"
            )
        
        self.checkpoint_path = check_file_path(checkpoint_path, make_dirs=False)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load metadata
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = Box(yaml.safe_load(f))
        else:
            self.metadata = Box(self.DEFAULT_METADATA)
            if metadata_path:
                warnings.warn(f"Metadata file not found: {metadata_path}. Using defaults.")
        
        # Load model
        self._load_model()
        
        # Image processing attributes
        self.image = None
        self.source = None
        self.sensor_type = None
        self.raster_profile = None
        
    def _load_model(self):
        """Load the Clay model from checkpoint."""
        try:
            torch.set_default_device(self.device)
            self.model = ClayMAEModule.load_from_checkpoint(
                self.checkpoint_path,
                shuffle=False,
                mask_ratio=0.0
            )
            self.model.eval()
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Clay model: {e}")
    
    def _detect_sensor_type(
        self, 
        src: rasterio.DatasetReader, 
        source_path: Optional[str] = None
    ) -> str:
        """
        Detect sensor type from raster metadata and characteristics.
        
        Args:
            src: Rasterio dataset reader
            source_path: Optional source file path for filename-based detection
            
        Returns:
            Detected sensor type string
        """
        band_count = src.count
        resolution = abs(src.transform[0])  # Pixel size
        
        # Try filename-based detection first
        if source_path:
            filename = os.path.basename(source_path).lower()
            if 'sentinel' in filename or 's2' in filename:
                return 'sentinel-2-l2a'
            elif 'landsat' in filename or 'l8' in filename or 'l9' in filename:
                return 'landsat-c2l2-sr'
            elif 'naip' in filename:
                return 'naip'
        
        # Fallback to resolution and band count heuristics
        if band_count == 4 and resolution <= 5:
            return 'naip'  # High-res 4-band imagery
        elif band_count >= 6 and 25 <= resolution <= 35:
            return 'landsat-c2l2-sr'  # Landsat resolution
        elif band_count >= 10 and 8 <= resolution <= 12:
            return 'sentinel-2-l2a'  # Sentinel-2 resolution
        elif band_count == 4:
            return 'naip'  # Default 4-band to NAIP
        else:
            # Default fallback
            warnings.warn(
                f"Could not detect sensor type (bands: {band_count}, "
                f"resolution: {resolution:.1f}m). Defaulting to NAIP."
            )
            return 'naip'
    
    def _get_raster_center_latlon(self, src: rasterio.DatasetReader) -> Tuple[float, float]:
        """Get the center lat/lon of the raster."""
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        
        # Transform to WGS84 if needed
        if src.crs != 'EPSG:4326':
            lon, lat = transform_coords(
                [(center_x, center_y)], 
                src.crs, 
                'EPSG:4326'
            )[0]
        else:
            lon, lat = center_x, center_y
            
        return lat, lon
    
    def _prepare_datacube(
        self, 
        image: np.ndarray, 
        sensor_type: str,
        lat: float, 
        lon: float, 
        date: Optional[datetime.datetime] = None,
        gsd_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare datacube for Clay model input.
        
        Args:
            image: Input image array [H, W, C]
            sensor_type: Detected sensor type
            lat: Latitude of image center
            lon: Longitude of image center  
            date: Image acquisition date
            gsd_override: Override GSD value
            
        Returns:
            Datacube dictionary for Clay model
        """
        if date is None:
            date = datetime.datetime.now()
        
        # Get sensor metadata
        sensor_meta = self.metadata[sensor_type]
        band_order = sensor_meta.band_order
        gsd = gsd_override if gsd_override is not None else sensor_meta.gsd
        
        # Extract normalization parameters
        means = [sensor_meta.bands.mean[band] for band in band_order]
        stds = [sensor_meta.bands.std[band] for band in band_order]
        wavelengths = [sensor_meta.bands.wavelength[band] for band in band_order]
        
        # Convert image to torch tensor and normalize
        # Ensure we have the right number of bands
        if image.shape[2] != len(band_order):
            warnings.warn(
                f"Image has {image.shape[2]} bands but sensor {sensor_type} "
                f"expects {len(band_order)} bands. Using available bands."
            )
            # Take only the available bands
            num_bands = min(image.shape[2], len(band_order))
            image = image[:, :, :num_bands]
            means = means[:num_bands]
            stds = stds[:num_bands]
            wavelengths = wavelengths[:num_bands]
        
        # Convert to tensor and transpose to [C, H, W]
        pixels = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        
        # Normalize
        transform = v2.Compose([v2.Normalize(mean=means, std=stds)])
        pixels = transform(pixels).unsqueeze(0)  # Add batch dimension
        
        # Prepare temporal encoding
        time_norm = normalize_timestamp(date)
        
        # Prepare spatial encoding
        lat_norm, lon_norm = normalize_latlon(lat, lon)
        
        # Create datacube
        datacube = {
            'pixels': pixels.to(self.device),
            'time': torch.tensor(
                time_norm + time_norm,  # Clay expects 4 elements: [week, hour, week, hour]
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0),
            'latlon': torch.tensor(
                lat_norm + lon_norm,  # Clay expects 4 elements: [sin_lat, cos_lat, sin_lon, cos_lon]
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0),
            'gsd': torch.tensor(gsd, device=self.device),
            'waves': torch.tensor(wavelengths, device=self.device)
        }
        
        return datacube
    
    def set_image(
        self, 
        source: Union[str, np.ndarray],
        sensor_type: Optional[str] = None,
        date: Optional[Union[str, datetime.datetime]] = None,
        gsd_override: Optional[float] = None
    ):
        """
        Set the input image for embedding generation.
        
        Args:
            source: Path to image file or numpy array
            sensor_type: Optional sensor type override
            date: Image acquisition date
            gsd_override: Override GSD value
        """
        if isinstance(source, str):
            if source.startswith("http"):
                source = download_file(source)
            
            if not os.path.exists(source):
                raise ValueError(f"Input path {source} does not exist.")
            
            # Read with rasterio for geospatial images
            try:
                with rasterio.open(source) as src:
                    # Read all bands
                    image = src.read()  # Shape: [C, H, W]
                    image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]
                    
                    # Store raster metadata
                    self.raster_profile = src.profile
                    
                    # Detect sensor type
                    if sensor_type is None:
                        sensor_type = self._detect_sensor_type(src, source)
                    
                    # Get image center coordinates
                    lat, lon = self._get_raster_center_latlon(src)
                    
            except Exception:
                # Fallback to OpenCV for regular images
                image = cv2.imread(source)
                if image is None:
                    raise ValueError(f"Could not read image: {source}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Use defaults for non-geospatial images
                sensor_type = sensor_type or 'naip'
                lat, lon = 0.0, 0.0  # Default coordinates
                self.raster_profile = None
                
        elif isinstance(source, np.ndarray):
            image = source
            sensor_type = sensor_type or 'naip'
            lat, lon = 0.0, 0.0
            self.raster_profile = None
            
        else:
            raise ValueError("Source must be a file path or numpy array")
        
        # Parse date if string
        if isinstance(date, str):
            try:
                date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00'))
            except ValueError:
                date = datetime.datetime.now()
                warnings.warn(f"Could not parse date: {date}. Using current time.")
        elif date is None:
            date = datetime.datetime.now()
        
        # Store image and metadata
        self.source = source if isinstance(source, str) else None
        self.image = image
        self.sensor_type = sensor_type
        self.lat = lat
        self.lon = lon
        self.date = date
        self.gsd_override = gsd_override
        
        print(f"Set image: shape={image.shape}, sensor={sensor_type}, "
              f"lat={lat:.4f}, lon={lon:.4f}")
    
    def generate_embeddings(
        self, 
        tile_size: int = 256,
        overlap: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate embeddings for the loaded image.
        
        Args:
            tile_size: Size of tiles for processing large images
            overlap: Overlap fraction between tiles (0.0 to 1.0)
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        if self.image is None:
            raise ValueError("No image loaded. Call set_image() first.")
        
        image = self.image
        h, w = image.shape[:2]
        
        # If image is smaller than tile_size, process as single tile
        if h <= tile_size and w <= tile_size:
            # Pad image to tile_size if needed
            if h < tile_size or w < tile_size:
                pad_h = max(0, tile_size - h)
                pad_w = max(0, tile_size - w)
                image = np.pad(
                    image, 
                    ((0, pad_h), (0, pad_w), (0, 0)), 
                    mode='reflect'
                )
            
            # Generate single embedding
            datacube = self._prepare_datacube(
                image, self.sensor_type, self.lat, self.lon, 
                self.date, self.gsd_override
            )
            
            with torch.no_grad():
                encoded_patches, _, _, _ = self.model.model.encoder(datacube)
                # Extract class token (global embedding)
                embedding = encoded_patches[:, 0, :].cpu().numpy()
            
            return {
                'embeddings': embedding,
                'tile_coords': [(0, 0, h, w)],
                'image_shape': (h, w),
                'sensor_type': self.sensor_type,
                'lat': self.lat,
                'lon': self.lon,
                'date': self.date.isoformat() if self.date else None,
                'num_tiles': 1
            }
        
        else:
            # Process as overlapping tiles
            step_size = int(tile_size * (1 - overlap))
            embeddings = []
            tile_coords = []
            
            for y in range(0, h - tile_size + 1, step_size):
                for x in range(0, w - tile_size + 1, step_size):
                    # Extract tile
                    tile = image[y:y+tile_size, x:x+tile_size]
                    
                    # Prepare datacube for this tile
                    datacube = self._prepare_datacube(
                        tile, self.sensor_type, self.lat, self.lon,
                        self.date, self.gsd_override
                    )
                    
                    # Generate embedding
                    with torch.no_grad():
                        encoded_patches, _, _, _ = self.model.model.encoder(datacube)
                        embedding = encoded_patches[:, 0, :].cpu().numpy()
                    
                    embeddings.append(embedding)
                    tile_coords.append((x, y, x+tile_size, y+tile_size))
            
            return {
                'embeddings': np.vstack(embeddings),
                'tile_coords': tile_coords,
                'image_shape': (h, w),
                'sensor_type': self.sensor_type,
                'lat': self.lat,
                'lon': self.lon,
                'date': self.date.isoformat() if self.date else None,
                'num_tiles': len(embeddings)
            }
    
    def save_embeddings(
        self, 
        embeddings_result: Dict[str, Any], 
        output_path: str,
        format: str = 'npz'
    ):
        """
        Save embeddings to file.
        
        Args:
            embeddings_result: Result from generate_embeddings()
            output_path: Output file path
            format: Output format ('npz', 'pt')
        """
        output_path = check_file_path(output_path)
        
        if format == 'npz':
            np.savez_compressed(
                output_path,
                embeddings=embeddings_result['embeddings'],
                tile_coords=np.array(embeddings_result['tile_coords']),
                image_shape=np.array(embeddings_result['image_shape']),
                sensor_type=embeddings_result['sensor_type'],
                lat=embeddings_result['lat'],
                lon=embeddings_result['lon'],
                date=embeddings_result['date'],
                num_tiles=embeddings_result['num_tiles']
            )
        elif format == 'pt':
            torch.save(embeddings_result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved embeddings to {output_path}")


def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Load embeddings from file.
    
    Args:
        file_path: Path to embeddings file
        
    Returns:
        Embeddings dictionary
    """
    if file_path.endswith('.npz'):
        data = np.load(file_path, allow_pickle=True)
        return {
            'embeddings': data['embeddings'],
            'tile_coords': data['tile_coords'].tolist(),
            'image_shape': tuple(data['image_shape']),
            'sensor_type': str(data['sensor_type']),
            'lat': float(data['lat']),
            'lon': float(data['lon']),
            'date': str(data['date']) if data['date'] != 'None' else None,
            'num_tiles': int(data['num_tiles'])
        }
    elif file_path.endswith('.pt'):
        return torch.load(file_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")