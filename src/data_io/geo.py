"""
Geospatial utilities for coordinate transformations and mapping.
Handles pixel <-> lat/lon conversions and spatial operations.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds, rowcol, xy
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.crs import CRS
import pyproj
from typing import Tuple, List, Optional, Union, Dict
import logging
import cv2
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class GeoTransformer:
    """Handle coordinate transformations for satellite images."""
    
    def __init__(self, bounds: Tuple[float, float, float, float], 
                 image_shape: Tuple[int, int], 
                 crs: str = "EPSG:4326"):
        """
        Initialize geo transformer.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in geographic coordinates
            image_shape: (height, width) in pixels
            crs: Coordinate reference system
        """
        self.bounds = bounds  # (west, south, east, north)
        self.height, self.width = image_shape
        self.crs = CRS.from_string(crs)
        
        # Create affine transform
        self.transform = from_bounds(
            west=bounds[0], south=bounds[1], 
            east=bounds[2], north=bounds[3],
            width=self.width, height=self.height
        )
        
        # Calculate pixel size
        self.pixel_size_x = (bounds[2] - bounds[0]) / self.width
        self.pixel_size_y = (bounds[3] - bounds[1]) / self.height
        
    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to pixel coordinates (row, col)."""
        row, col = rowcol(self.transform, lon, lat)
        return int(row), int(col)
    
    def pixel_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert pixel coordinates to lat/lon."""
        lon, lat = xy(self.transform, row, col)
        return float(lat), float(lon)
    
    def latlon_to_pixel_batch(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert arrays of lat/lon to pixel coordinates."""
        rows, cols = rowcol(self.transform, lons, lats)
        return rows.astype(int), cols.astype(int)
    
    def pixel_to_latlon_batch(self, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert arrays of pixel coordinates to lat/lon."""
        lons, lats = xy(self.transform, rows, cols)
        return lats, lons
    
    def is_point_in_bounds(self, lat: float, lon: float) -> bool:
        """Check if a lat/lon point is within image bounds."""
        return (self.bounds[0] <= lon <= self.bounds[2] and 
                self.bounds[1] <= lat <= self.bounds[3])
    
    def get_pixel_area_km2(self) -> float:
        """Get area of one pixel in square kilometers."""
        # Use approximate conversion for small areas
        lat_center = (self.bounds[1] + self.bounds[3]) / 2
        
        # Convert degrees to meters (approximate)
        meters_per_deg_lat = 111320  # meters per degree latitude
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(lat_center))
        
        pixel_area_m2 = (abs(self.pixel_size_y) * meters_per_deg_lat * 
                        abs(self.pixel_size_x) * meters_per_deg_lon)
        
        return pixel_area_m2 / 1e6  # convert to km²


class FirePointMapper:
    """Map fire points to pixel coordinates and create masks."""
    
    def __init__(self, geo_transformer: GeoTransformer):
        self.geo_transformer = geo_transformer
    
    def points_to_pixels(self, fire_points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Convert fire points (lat, lon) to pixel coordinates.
        
        Args:
            fire_points: List of (lat, lon) tuples
            
        Returns:
            List of (row, col) pixel coordinates
        """
        pixel_coords = []
        
        for lat, lon in fire_points:
            if self.geo_transformer.is_point_in_bounds(lat, lon):
                row, col = self.geo_transformer.latlon_to_pixel(lat, lon)
                # Check if pixel is within image bounds
                if (0 <= row < self.geo_transformer.height and 
                    0 <= col < self.geo_transformer.width):
                    pixel_coords.append((row, col))
                else:
                    logger.warning(f"Fire point ({lat}, {lon}) maps to out-of-bounds pixel ({row}, {col})")
            else:
                logger.warning(f"Fire point ({lat}, {lon}) is outside image bounds")
        
        return pixel_coords
    
    def create_binary_mask(self, fire_points: List[Tuple[float, float]], 
                          radius: int = 2) -> np.ndarray:
        """
        Create binary fire mask from fire points.
        
        Args:
            fire_points: List of (lat, lon) fire locations
            radius: Radius in pixels around each fire point
            
        Returns:
            Binary mask (0=no fire, 1=fire)
        """
        mask = np.zeros((self.geo_transformer.height, self.geo_transformer.width), dtype=np.uint8)
        
        pixel_coords = self.points_to_pixels(fire_points)
        
        for row, col in pixel_coords:
            # Create circular mask around fire point
            y_min = max(0, row - radius)
            y_max = min(self.geo_transformer.height, row + radius + 1)
            x_min = max(0, col - radius)
            x_max = min(self.geo_transformer.width, col + radius + 1)
            
            # Create circular kernel
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if np.sqrt((y - row)**2 + (x - col)**2) <= radius:
                        mask[y, x] = 1
        
        return mask
    
    def create_distance_mask(self, fire_points: List[Tuple[float, float]], 
                           max_distance: float = 5.0) -> np.ndarray:
        """
        Create distance-based fire mask (values decrease with distance).
        
        Args:
            fire_points: List of (lat, lon) fire locations
            max_distance: Maximum distance in pixels for non-zero values
            
        Returns:
            Distance mask (0-1, where 1=fire center, 0=far from fire)
        """
        mask = np.zeros((self.geo_transformer.height, self.geo_transformer.width), dtype=np.float32)
        
        pixel_coords = self.points_to_pixels(fire_points)
        
        if not pixel_coords:
            return mask
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:self.geo_transformer.height, :self.geo_transformer.width]
        
        for row, col in pixel_coords:
            # Calculate distance from fire point
            distances = np.sqrt((y_grid - row)**2 + (x_grid - col)**2)
            
            # Create distance-based mask
            fire_mask = np.maximum(0, 1 - distances / max_distance)
            mask = np.maximum(mask, fire_mask)  # Take maximum if multiple fires overlap
        
        return mask


class FireDetectionConfig:
    """Configuration class for fire detection parameters."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize fire detection configuration."""
        if config_dict is None:
            config_dict = {}
        
        # Get fire detection parameters with defaults based on debug results
        fire_config = config_dict.get('data', {}).get('fire_detection', {})
        thresholds = fire_config.get('rgb_thresholds', {})
        noise_config = fire_config.get('noise_removal', {})
        
        # RGB threshold parameters - optimized based on debug results
        self.red_threshold = thresholds.get('red_threshold', 180)      # From debug: 170-200 works well
        self.orange_ratio = thresholds.get('orange_ratio', 1.0)        # From debug: 0.7-1.2 works well
        self.brightness_threshold = thresholds.get('brightness_threshold', 400)  # Moderate brightness
        self.green_max = thresholds.get('green_max', 160)              # Allow some green
        self.blue_max = thresholds.get('blue_max', 140)                # Allow some blue
        self.red_green_contrast = thresholds.get('red_green_contrast', 40)  # Moderate contrast
        self.red_blue_contrast = thresholds.get('red_blue_contrast', 30)    # Moderate contrast
        self.min_fire_area = thresholds.get('min_fire_area', 2)        # Allow smaller fires
        
        # Noise removal parameters
        self.noise_removal_enabled = noise_config.get('enabled', True)
        self.kernel_size = noise_config.get('kernel_size', 3)
        self.min_component_area = noise_config.get('min_component_area', 2)
        
        logger.info(f"Fire Detection Config loaded:")
        logger.info(f"  Red threshold: {self.red_threshold}")
        logger.info(f"  Orange ratio: {self.orange_ratio}")
        logger.info(f"  Brightness threshold: {self.brightness_threshold}")
        logger.info(f"  Green max: {self.green_max}")
        logger.info(f"  Blue max: {self.blue_max}")
        logger.info(f"  Min fire area: {self.min_fire_area}")
        logger.info(f"  Noise removal: {self.noise_removal_enabled}")
    
    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> 'FireDetectionConfig':
        """Create configuration from YAML file."""
        if not Path(yaml_path).exists():
            logger.warning(f"Config file not found: {yaml_path}. Using optimized defaults.")
            return cls()
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(config_dict)
        except Exception as e:
            logger.error(f"Error loading config from {yaml_path}: {e}")
            return cls()


def load_fire_detection_config(data_config_path: str = None) -> FireDetectionConfig:
    """Load fire detection configuration from YAML file."""
    if data_config_path is None:
        # Try common config paths
        possible_paths = [
            "configs/data.yaml",
            "configs/data.yml", 
            "data.yaml",
            "data.yml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                data_config_path = path
                break
    
    if data_config_path and Path(data_config_path).exists():
        return FireDetectionConfig.from_yaml_file(data_config_path)
    else:
        logger.warning("No config file found. Using optimized fire detection parameters.")
        return FireDetectionConfig()


def detect_fire_pixels_auto(image: np.ndarray, 
                           config: Optional[FireDetectionConfig] = None) -> List[Tuple[int, int]]:
    """
    Automatically detect fire pixels using optimized configuration parameters.
    
    Args:
        image: RGB or RGBA image array (H, W, 3) or (H, W, 4)
        config: Fire detection configuration. If None, loads from config file.
        
    Returns:
        List of (row, col) pixel coordinates of detected fires
    """
    if config is None:
        config = load_fire_detection_config()
    
    if image.shape[2] not in [3, 4]:
        raise ValueError("Image must be RGB (3 channels) or RGBA (4 channels)")
    
    # Extract RGB channels (ignore alpha if present)
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    
    # Apply fire detection criteria using optimized parameters
    conditions = [
        red > config.red_threshold,                           # Primary red requirement
        red > green * config.orange_ratio,                    # Red dominates green
        red > blue * 1.2,                                     # Red higher than blue (relaxed)
        (red + green + blue) > config.brightness_threshold,   # Brightness requirement
        (red - green) > config.red_green_contrast,            # Red-green contrast
        (red - blue) > config.red_blue_contrast,              # Red-blue contrast
        green < config.green_max,                             # Limit vegetation
        blue < config.blue_max                                # Limit water/sky
    ]
    
    # ALL conditions must be satisfied
    fire_mask = np.all(conditions, axis=0)
    
    # Apply morphological operations if enabled
    if config.noise_removal_enabled and np.any(fire_mask):
        # Remove salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel_size, config.kernel_size))
        fire_mask = cv2.morphologyEx(fire_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Remove very small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fire_mask)
        
        cleaned_mask = np.zeros_like(fire_mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= config.min_fire_area:
                cleaned_mask[labels == i] = 1
        
        fire_mask = cleaned_mask
    
    # Get pixel coordinates
    fire_pixels = np.where(fire_mask > 0)
    detected_pixels = list(zip(fire_pixels[0], fire_pixels[1]))
    
    return detected_pixels


def calculate_fire_area(mask: np.ndarray, geo_transformer: GeoTransformer) -> float:
    """Calculate total fire area in square kilometers."""
    fire_pixel_count = np.sum(mask > 0)
    pixel_area_km2 = geo_transformer.get_pixel_area_km2()
    return fire_pixel_count * pixel_area_km2


def get_fire_centroid(mask: np.ndarray, geo_transformer: GeoTransformer) -> Optional[Tuple[float, float]]:
    """Get centroid of fire mask in lat/lon coordinates."""
    if not np.any(mask):
        return None
    
    # Find centroid in pixel coordinates
    fire_pixels = np.where(mask > 0)
    centroid_row = np.mean(fire_pixels[0])
    centroid_col = np.mean(fire_pixels[1])
    
    # Convert to lat/lon
    return geo_transformer.pixel_to_latlon(int(centroid_row), int(centroid_col))


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    print("Testing Fire Detection Configuration System")
    print("=" * 50)
    
    # Load configuration
    config = load_fire_detection_config("configs/data.yaml")
    
    print(f"Configuration loaded successfully!")
    print(f"Red threshold: {config.red_threshold}")
    print(f"Orange ratio: {config.orange_ratio}")
    print(f"Brightness threshold: {config.brightness_threshold}")
    print(f"Noise removal enabled: {config.noise_removal_enabled}")
    
    # Example bounds for testing
    bounds = (-81.0, 42.0, -80.0, 43.0)  # (west, south, east, north)
    image_shape = (3800, 3400)  # Corrected order: height, width
    
    geo_transformer = GeoTransformer(bounds, image_shape)
    fire_mapper = FirePointMapper(geo_transformer)
    
    # Test coordinate conversion
    test_lat, test_lon = 42.5, -80.5
    row, col = geo_transformer.latlon_to_pixel(test_lat, test_lon)
    back_lat, back_lon = geo_transformer.pixel_to_latlon(row, col)
    
    print(f"\nCoordinate conversion test:")
    print(f"Original: ({test_lat}, {test_lon})")
    print(f"Pixel: ({row}, {col})")
    print(f"Back to lat/lon: ({back_lat:.6f}, {back_lon:.6f})")
    
    # Test fire mask creation
    fire_points = [(42.3, -80.7), (42.7, -80.3)]
    mask = fire_mapper.create_binary_mask(fire_points, radius=3)
    print(f"Created fire mask with {np.sum(mask)} fire pixels")
    
    # Calculate area
    area = calculate_fire_area(mask, geo_transformer)
    print(f"Total fire area: {area:.2f} km²")
    
    print(f"\nFire Detection Parameters (from config):")
    print(f"- Red threshold: {config.red_threshold}")
    print(f"- Orange ratio: {config.orange_ratio}")
    print(f"- Brightness: {config.brightness_threshold}")
    print(f"- Green max: {config.green_max}")
    print(f"- Blue max: {config.blue_max}")
    print(f"- Min fire area: {config.min_fire_area}")
    print(f"- Noise removal: {config.noise_removal_enabled}")
    
    print("\n✅ Configuration system ready!")