"""
Data reading utilities for forest fire detection.
Handles GeoTIFF images and CSV metadata.
"""

import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Dict, List, Tuple, Optional, Union
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageMetadata:
    """Container for image metadata from CSV."""
    
    def __init__(self, row: pd.Series):
        self.filename = row['File Name']
        self.state = row['State']
        self.date = pd.to_datetime(row['Date'])
        self.top_right_lat = float(row['Top Right Latitude'])
        self.top_right_lon = float(row['Top Right Longitude'])
        self.bottom_left_lat = float(row['Bottom Left Latitude'])
        self.bottom_left_lon = float(row['Bottom Left Longitude'])
        
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (min_x, min_y, max_x, max_y)."""
        return (
            self.bottom_left_lon,  # min_x (west)
            self.bottom_left_lat,  # min_y (south)
            self.top_right_lon,    # max_x (east)
            self.top_right_lat     # max_y (north)
        )
    
    def __repr__(self):
        return f"ImageMetadata({self.filename}, {self.state}, {self.date.date()})"


class MetadataReader:
    """Read and parse CSV metadata file."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from CSV file."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            logger.info(f"Loaded metadata for {len(self.df)} images")
        except Exception as e:
            logger.error(f"Error loading metadata from {self.csv_path}: {e}")
            raise
    
    def get_metadata(self, filename: str) -> Optional[ImageMetadata]:
        """Get metadata for a specific filename."""
        row = self.df[self.df['File Name'] == filename]
        if row.empty:
            logger.warning(f"No metadata found for {filename}")
            return None
        return ImageMetadata(row.iloc[0])
    
    def get_state_files(self, state: str) -> List[str]:
        """Get all filenames for a specific state."""
        state_df = self.df[self.df['State'].str.lower() == state.lower()]
        return state_df['File Name'].tolist()
    
    def get_date_range_files(self, start_date: str, end_date: str, 
                           state: Optional[str] = None) -> List[str]:
        """Get filenames within date range, optionally filtered by state."""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        mask = (self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)
        if state:
            mask &= (self.df['State'].str.lower() == state.lower())
        
        return self.df[mask]['File Name'].tolist()
    
    def get_all_metadata(self) -> List[ImageMetadata]:
        """Get metadata for all images."""
        return [ImageMetadata(row) for _, row in self.df.iterrows()]


class GeoTIFFReader:
    """Read GeoTIFF satellite images."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def read_image(self, filename: str, state: str, 
                   window: Optional[Window] = None) -> Tuple[np.ndarray, Dict]:
        """
        Read a GeoTIFF image.
        
        Args:
            filename: Image filename
            state: State name (subdirectory)
            window: Optional window to read subset
            
        Returns:
            tuple: (image_array, metadata_dict)
        """
        # Construct full path
        image_path = self._get_image_path(filename, state)
        
        try:
            with rasterio.open(image_path) as src:
                # Read image data
                if window:
                    image = src.read(window=window)
                else:
                    image = src.read()
                
                # Get metadata
                metadata = {
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'crs': src.crs,
                    'height': src.height,
                    'width': src.width,
                    'count': src.count,
                    'dtype': src.dtypes[0] if src.dtypes else None,  # Fixed: use dtypes[0] instead of dtype
                    'nodata': src.nodata
                }
                
                # Convert from (C, H, W) to (H, W, C) and handle channel conversion
                if image.shape[0] >= 3:
                    # Take only first 3 channels (RGB) if more than 3 channels
                    if image.shape[0] > 3:
                        image = image[:3]  # Keep only RGB, drop alpha
                    image = np.transpose(image, (1, 2, 0))
                
                return image, metadata
                
        except Exception as e:
            logger.error(f"Error reading {image_path}: {e}")
            raise
    
    def _get_image_path(self, filename: str, state: str) -> Path:
        """Construct full path to image file."""
        # Handle different filename patterns
        if not filename.endswith(('.png', '.tif', '.tiff')):
            filename += '.png'
        
        # Try different path patterns
        possible_paths = [
            self.base_path / state / filename,
            self.base_path / state / f"{state}-{filename}",
            self.base_path / state / filename.replace('.png', '') / filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If not found, raise error with helpful message
        raise FileNotFoundError(
            f"Could not find image {filename} for state {state}. "
            f"Checked paths: {[str(p) for p in possible_paths]}"
        )
    
    def get_image_info(self, filename: str, state: str) -> Dict:
        """Get basic info about an image without loading it."""
        image_path = self._get_image_path(filename, state)
        
        with rasterio.open(image_path) as src:
            return {
                'path': str(image_path),
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'dtype': src.dtypes[0] if src.dtypes else None,  # Fixed: use dtypes[0] instead of dtype
                'crs': src.crs,
                'bounds': src.bounds,
                'transform': src.transform
            }


def load_image_opencv(image_path: str) -> np.ndarray:
    """Load image using OpenCV as fallback."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def discover_images(base_path: str, state: str, pattern: str = "*.png") -> List[str]:
    """Discover all image files in a directory."""
    search_path = Path(base_path) / state
    if not search_path.exists():
        logger.warning(f"Directory does not exist: {search_path}")
        return []
    
    images = list(search_path.glob(pattern))
    images.extend(list(search_path.glob("**/" + pattern)))  # Search subdirectories
    
    return [img.name for img in images]


# Example usage
if __name__ == "__main__":
    # Example usage
    metadata_reader = MetadataReader("data/raw/WorldView Metadata - Sheet1.csv")
    geotiff_reader = GeoTIFFReader("data/raw/forest_fire_dataset")
    
    # Get Ontario files
    ontario_files = metadata_reader.get_state_files("Ontario")
    print(f"Found {len(ontario_files)} Ontario images")
    
    if ontario_files:
        # Read first image
        filename = ontario_files[0]
        metadata = metadata_reader.get_metadata(filename)
        image, img_metadata = geotiff_reader.read_image(filename, "Ontario")
        
        print(f"Loaded {filename}: {image.shape}, bounds: {metadata.get_bounds()}")