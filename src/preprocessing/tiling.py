"""
Image tiling utilities for splitting large satellite images into smaller patches.
Handles overlapping tiles, fire content filtering, and metadata preservation.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Iterator, NamedTuple
from pathlib import Path
import json
from dataclasses import dataclass
import logging
from itertools import product

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass
class TileInfo:
    """Information about a single tile."""
    tile_id: str
    source_image: str
    tile_row: int  # Row index in tiling grid
    tile_col: int  # Column index in tiling grid
    pixel_bounds: Tuple[int, int, int, int]  # (row_start, row_end, col_start, col_end)
    geographic_bounds: Optional[Tuple[float, float, float, float]] = None  # (west, south, east, north)
    has_fire: bool = False
    fire_pixel_count: int = 0
    total_pixels: int = 0
    
    @property
    def fire_ratio(self) -> float:
        """Ratio of fire pixels to total pixels."""
        return self.fire_pixel_count / self.total_pixels if self.total_pixels > 0 else 0.0


class ImageTiler:
    """Split large images into smaller tiles for training."""
    
    def __init__(self, tile_size: int = 512, overlap: int = 64, 
                 min_fire_pixels: int = 10, geo_transformer=None):
        """
        Initialize tiler.
        
        Args:
            tile_size: Size of each tile (assumed square)
            overlap: Overlap between adjacent tiles in pixels
            min_fire_pixels: Minimum fire pixels required to include tile
            geo_transformer: Optional transformer for geographic coordinates
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_fire_pixels = min_fire_pixels
        self.geo_transformer = geo_transformer
    
    def calculate_tile_grid(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate number of tiles in each dimension."""
        height, width = image_shape[:2]
        
        # Calculate number of tiles needed
        n_rows = max(1, (height - self.overlap) // self.stride)
        n_cols = max(1, (width - self.overlap) // self.stride)
        
        # Add extra tiles if needed to cover the entire image
        if (n_rows - 1) * self.stride + self.tile_size < height:
            n_rows += 1
        if (n_cols - 1) * self.stride + self.tile_size < width:
            n_cols += 1
            
        return n_rows, n_cols
    
    def get_tile_bounds(self, tile_row: int, tile_col: int, 
                       image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get pixel bounds for a specific tile."""
        height, width = image_shape[:2]
        
        # Calculate starting positions
        row_start = tile_row * self.stride
        col_start = tile_col * self.stride
        
        # Calculate ending positions (ensure we don't exceed image bounds)
        row_end = min(row_start + self.tile_size, height)
        col_end = min(col_start + self.tile_size, width)
        
        # Adjust start positions if tile would be too small
        if row_end - row_start < self.tile_size and row_end == height:
            row_start = max(0, height - self.tile_size)
        if col_end - col_start < self.tile_size and col_end == width:
            col_start = max(0, width - self.tile_size)
            
        return row_start, row_end, col_start, col_end
    
    def extract_tile(self, image: np.ndarray, tile_bounds: Tuple[int, int, int, int],
                    target_size: Optional[int] = None) -> np.ndarray:
        """Extract a single tile from image."""
        row_start, row_end, col_start, col_end = tile_bounds
        
        # Extract tile
        if len(image.shape) == 3:
            tile = image[row_start:row_end, col_start:col_end, :]
        else:
            tile = image[row_start:row_end, col_start:col_end]
        
        # Resize if necessary (for edge tiles that might be smaller)
        if target_size and tile.shape[:2] != (target_size, target_size):
            if len(tile.shape) == 3:
                tile = cv2.resize(tile, (target_size, target_size))
            else:
                tile = cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        return tile
    
    def get_geographic_bounds(self, tile_bounds: Tuple[int, int, int, int]) -> Optional[Tuple[float, float, float, float]]:
        """Get geographic bounds for a tile."""
        if not self.geo_transformer:
            return None
        
        row_start, row_end, col_start, col_end = tile_bounds
        
        # Get corner coordinates
        top_left_lat, top_left_lon = self.geo_transformer.pixel_to_latlon(row_start, col_start)
        bottom_right_lat, bottom_right_lon = self.geo_transformer.pixel_to_latlon(row_end-1, col_end-1)
        
        # Return as (west, south, east, north)
        return (
            min(top_left_lon, bottom_right_lon),  # west
            min(top_left_lat, bottom_right_lat),  # south
            max(top_left_lon, bottom_right_lon),  # east
            max(top_left_lat, bottom_right_lat)   # north
        )
    
    def analyze_fire_content(self, mask: np.ndarray, tile_bounds: Tuple[int, int, int, int]) -> Tuple[int, bool]:
        """Analyze fire content in a tile."""
        row_start, row_end, col_start, col_end = tile_bounds
        tile_mask = mask[row_start:row_end, col_start:col_end]
        
        fire_pixel_count = int(np.sum(tile_mask > 0))  # Convert to Python int
        has_fire = fire_pixel_count >= self.min_fire_pixels
        
        return fire_pixel_count, bool(has_fire)  # Ensure Python bool
    
    def generate_tiles(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                      source_filename: str = "unknown") -> List[TileInfo]:
        """
        Generate all tiles for an image.
        
        Args:
            image: Input image array
            mask: Optional fire mask
            source_filename: Name of source image file
            
        Returns:
            List of TileInfo objects
        """
        image_shape = image.shape[:2]
        n_rows, n_cols = self.calculate_tile_grid(image_shape)
        
        tiles = []
        
        for tile_row, tile_col in product(range(n_rows), range(n_cols)):
            # Get tile bounds
            tile_bounds = self.get_tile_bounds(tile_row, tile_col, image_shape)
            
            # Analyze fire content if mask provided
            fire_pixel_count = 0
            has_fire = True  # Include all tiles if no mask provided
            
            if mask is not None:
                fire_pixel_count, has_fire = self.analyze_fire_content(mask, tile_bounds)
            
            # Create tile info
            tile_id = f"{Path(source_filename).stem}_r{tile_row:03d}_c{tile_col:03d}"
            
            tile_info = TileInfo(
                tile_id=tile_id,
                source_image=source_filename,
                tile_row=int(tile_row),  # Ensure Python int
                tile_col=int(tile_col),  # Ensure Python int
                pixel_bounds=tuple(int(x) for x in tile_bounds),  # Ensure Python ints
                geographic_bounds=self.get_geographic_bounds(tile_bounds),
                has_fire=bool(has_fire),  # Ensure Python bool
                fire_pixel_count=int(fire_pixel_count),  # Ensure Python int
                total_pixels=int(self.tile_size * self.tile_size)  # Ensure Python int
            )
            
            tiles.append(tile_info)
        
        logger.info(f"Generated {len(tiles)} tiles for {source_filename}")
        if mask is not None:
            fire_tiles = [t for t in tiles if t.has_fire]
            logger.info(f"  {len(fire_tiles)} tiles contain fire")
        
        return tiles
    
    def save_tiles_to_disk(self, image: np.ndarray, mask: Optional[np.ndarray],
                          tiles: List[TileInfo], output_dir: str,
                          save_all: bool = False) -> Dict[str, List[str]]:
        """
        Save tile images and masks to disk.
        
        Args:
            image: Source image
            mask: Source mask (optional)
            tiles: List of tile info
            output_dir: Output directory
            save_all: If True, save all tiles; if False, only save tiles with fire
            
        Returns:
            Dictionary with lists of saved image and mask paths
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        if mask is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
        
        saved_images = []
        saved_masks = []
        
        for tile_info in tiles:
            # Skip tiles without fire unless save_all is True
            if not save_all and not tile_info.has_fire:
                continue
            
            # Extract and save image tile
            tile_image = self.extract_tile(image, tile_info.pixel_bounds, self.tile_size)
            image_path = images_dir / f"{tile_info.tile_id}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(tile_image, cv2.COLOR_RGB2BGR))
            saved_images.append(str(image_path))
            
            # Extract and save mask tile if available
            if mask is not None:
                tile_mask = self.extract_tile(mask, tile_info.pixel_bounds, self.tile_size)
                mask_path = masks_dir / f"{tile_info.tile_id}.png"
                cv2.imwrite(str(mask_path), tile_mask * 255)  # Convert 0/1 to 0/255
                saved_masks.append(str(mask_path))
        
        logger.info(f"Saved {len(saved_images)} image tiles and {len(saved_masks)} mask tiles")
        
        return {
            'images': saved_images,
            'masks': saved_masks
        }


class TileDataset:
    """Manage collections of tiles with metadata."""
    
    def __init__(self, tiles_dir: str):
        self.tiles_dir = Path(tiles_dir)
        self.metadata_file = self.tiles_dir / "tiles_metadata.json"
        self.tiles = []
        self._load_metadata()
    
    def _load_metadata(self):
        """Load tile metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                self.tiles = []
                for tile_data in data['tiles']:
                    tile_info = TileInfo(
                        tile_id=tile_data['tile_id'],
                        source_image=tile_data['source_image'],
                        tile_row=tile_data['tile_row'],
                        tile_col=tile_data['tile_col'],
                        pixel_bounds=tuple(tile_data['pixel_bounds']),
                        geographic_bounds=tuple(tile_data['geographic_bounds']) if tile_data.get('geographic_bounds') else None,
                        has_fire=tile_data['has_fire'],
                        fire_pixel_count=tile_data['fire_pixel_count'],
                        total_pixels=tile_data['total_pixels']
                    )
                    self.tiles.append(tile_info)
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not load metadata from {self.metadata_file}: {e}")
                logger.info("Starting with empty tile dataset")
                self.tiles = []
                # Optionally backup the corrupted file
                if self.metadata_file.exists():
                    backup_path = self.metadata_file.with_suffix('.json.backup')
                    self.metadata_file.rename(backup_path)
                    logger.info(f"Backed up corrupted metadata to {backup_path}")
        else:
            self.tiles = []
    
    def save_metadata(self):
        """Save tile metadata to disk."""
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            'tiles': [],
            'total_tiles': len(self.tiles),
            'fire_tiles': sum(1 for t in self.tiles if t.has_fire)
        }
        
        for tile in self.tiles:
            tile_data = {
                'tile_id': tile.tile_id,
                'source_image': tile.source_image,
                'tile_row': tile.tile_row,
                'tile_col': tile.tile_col,
                'pixel_bounds': tile.pixel_bounds,
                'geographic_bounds': tile.geographic_bounds,
                'has_fire': tile.has_fire,
                'fire_pixel_count': tile.fire_pixel_count,
                'total_pixels': tile.total_pixels
            }
            data['tiles'].append(tile_data)
        
        # Use custom encoder to handle any remaining NumPy types
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    def add_tiles(self, tiles: List[TileInfo]):
        """Add tiles to the dataset."""
        self.tiles.extend(tiles)
        self.save_metadata()
    
    def get_fire_tiles(self) -> List[TileInfo]:
        """Get tiles that contain fire."""
        return [t for t in self.tiles if t.has_fire]
    
    def get_no_fire_tiles(self) -> List[TileInfo]:
        """Get tiles that don't contain fire."""
        return [t for t in self.tiles if not t.has_fire]
    
    def get_balanced_sample(self, n_fire: int, n_no_fire: int) -> List[TileInfo]:
        """Get balanced sample of fire and no-fire tiles."""
        fire_tiles = self.get_fire_tiles()
        no_fire_tiles = self.get_no_fire_tiles()
        
        # Sample tiles
        import random
        sampled_fire = random.sample(fire_tiles, min(n_fire, len(fire_tiles)))
        sampled_no_fire = random.sample(no_fire_tiles, min(n_no_fire, len(no_fire_tiles)))
        
        return sampled_fire + sampled_no_fire
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        fire_tiles = self.get_fire_tiles()
        
        return {
            'total_tiles': len(self.tiles),
            'fire_tiles': len(fire_tiles),
            'no_fire_tiles': len(self.tiles) - len(fire_tiles),
            'fire_ratio': len(fire_tiles) / len(self.tiles) if self.tiles else 0,
            'avg_fire_pixels': np.mean([t.fire_pixel_count for t in fire_tiles]) if fire_tiles else 0,
            'total_fire_pixels': sum(t.fire_pixel_count for t in fire_tiles)
        }


def reconstruct_from_tiles(tiles: List[np.ndarray], tile_infos: List[TileInfo],
                          original_shape: Tuple[int, int], overlap: int) -> np.ndarray:
    """
    Reconstruct full image from overlapping tiles using blending.
    
    Args:
        tiles: List of tile arrays
        tile_infos: Corresponding tile information
        original_shape: Shape of original image
        overlap: Overlap between tiles
        
    Returns:
        Reconstructed image
    """
    if len(tiles) != len(tile_infos):
        raise ValueError("Number of tiles must match number of tile infos")
    
    # Initialize output arrays
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    weight_map = np.zeros(original_shape[:2], dtype=np.float32)
    
    for tile, tile_info in zip(tiles, tile_infos):
        row_start, row_end, col_start, col_end = tile_info.pixel_bounds
        
        # Resize tile if needed
        expected_h = row_end - row_start
        expected_w = col_end - col_start
        
        if tile.shape[:2] != (expected_h, expected_w):
            if len(tile.shape) == 3:
                tile = cv2.resize(tile, (expected_w, expected_h))
            else:
                tile = cv2.resize(tile, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
        
        # Create weight mask for blending (higher weights in center)
        tile_h, tile_w = tile.shape[:2]
        weight_mask = np.ones((tile_h, tile_w), dtype=np.float32)
        
        if overlap > 0:
            # Create distance-based weights
            center_h, center_w = tile_h // 2, tile_w // 2
            y, x = np.ogrid[:tile_h, :tile_w]
            distance = np.sqrt((y - center_h)**2 + (x - center_w)**2)
            max_distance = min(center_h, center_w)
            weight_mask = np.maximum(0.1, 1.0 - distance / max_distance)
        
        # Add to reconstruction
        if len(tile.shape) == 3:
            for c in range(tile.shape[2]):
                reconstructed[row_start:row_end, col_start:col_end, c] += tile[:, :, c] * weight_mask
        else:
            reconstructed[row_start:row_end, col_start:col_end] += tile * weight_mask
        
        weight_map[row_start:row_end, col_start:col_end] += weight_mask
    
    # Normalize by weights
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    if len(reconstructed.shape) == 3:
        for c in range(reconstructed.shape[2]):
            reconstructed[:, :, c] /= weight_map
    else:
        reconstructed /= weight_map
    
    return reconstructed.astype(np.uint8)


# Example usage
if __name__ == "__main__":
    # Example usage of tiling system
    from data_io.readers import MetadataReader, GeoTIFFReader
    from src.data_io.geo import GeoTransformer
    from preprocessing.rasterize import MaskGenerator, AnnotationManager
    
    # Load sample data
    metadata_reader = MetadataReader("data/raw/WorldView Metadata - Sheet1.csv")
    geotiff_reader = GeoTIFFReader("data/raw/forest_fire_dataset")
    
    ontario_files = metadata_reader.get_state_files("Ontario")
    if ontario_files:
        filename = ontario_files[0]
        img_metadata = metadata_reader.get_metadata(filename)
        image, _ = geotiff_reader.read_image(filename, "Ontario")
        
        # Set up geo transformer and generate mask
        bounds = img_metadata.get_bounds()
        geo_transformer = GeoTransformer(bounds, image.shape[:2])
        
        # Create tiler
        tiler = ImageTiler(
            tile_size=512,
            overlap=64,
            min_fire_pixels=10,
            geo_transformer=geo_transformer
        )
        
        # For demo, create a dummy mask
        dummy_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        dummy_mask[1000:1100, 1500:1600] = 1  # Add some fake fire pixels
        
        # Generate tiles
        tiles = tiler.generate_tiles(image, dummy_mask, filename)
        
        # Save tiles
        saved_paths = tiler.save_tiles_to_disk(
            image, dummy_mask, tiles,
            "data/interim/tiles",
            save_all=False
        )
        
        # Create tile dataset
        tile_dataset = TileDataset("data/interim/tiles")
        tile_dataset.add_tiles(tiles)
        
        print(f"Generated {len(tiles)} tiles")
        print(f"Statistics: {tile_dataset.get_statistics()}")
        print(f"Saved {len(saved_paths['images'])} image tiles")