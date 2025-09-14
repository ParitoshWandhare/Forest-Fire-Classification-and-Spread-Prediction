"""
Rasterization utilities for creating fire masks from point data or manual annotations.
Supports both automatic fire detection and manual annotation workflows.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
import json
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

from data_io.geo import (GeoTransformer, FirePointMapper, detect_fire_pixels_auto, 
                        FireDetectionConfig, load_fire_detection_config)

logger = logging.getLogger(__name__)


@dataclass
class FireAnnotation:
    """Container for manual fire annotations."""
    image_filename: str
    state: str
    date: datetime
    fire_points: List[Tuple[float, float]]  # (lat, lon) coordinates
    fire_polygons: List[List[Tuple[float, float]]]  # List of polygon vertices
    annotation_method: str  # 'manual', 'auto', 'hybrid'
    annotator: str
    created_at: datetime
    confidence: float = 1.0  # 0-1 confidence score


class FireRasterizer:
    """Convert fire annotations to binary masks."""
    
    def __init__(self, geo_transformer: GeoTransformer):
        self.geo_transformer = geo_transformer
        self.fire_mapper = FirePointMapper(geo_transformer)
    
    def points_to_mask(self, fire_points: List[Tuple[float, float]], 
                      radius: int = 3) -> np.ndarray:
        """Convert fire points to binary mask."""
        return self.fire_mapper.create_binary_mask(fire_points, radius=radius)
    
    def polygons_to_mask(self, fire_polygons: List[List[Tuple[float, float]]]) -> np.ndarray:
        """Convert fire polygons to binary mask."""
        mask = np.zeros((self.geo_transformer.height, self.geo_transformer.width), dtype=np.uint8)
        
        for polygon in fire_polygons:
            if len(polygon) < 3:
                logger.warning("Polygon has fewer than 3 vertices, skipping")
                continue
            
            # Convert lat/lon polygon to pixel coordinates
            pixel_polygon = []
            for lat, lon in polygon:
                if self.geo_transformer.is_point_in_bounds(lat, lon):
                    row, col = self.geo_transformer.latlon_to_pixel(lat, lon)
                    pixel_polygon.append([col, row])  # OpenCV expects (x, y)
            
            if len(pixel_polygon) >= 3:
                pixel_polygon = np.array(pixel_polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pixel_polygon], 1)
        
        return mask
    
    def annotation_to_mask(self, annotation: FireAnnotation) -> np.ndarray:
        """Convert complete annotation to binary mask."""
        # Start with empty mask
        mask = np.zeros((self.geo_transformer.height, self.geo_transformer.width), dtype=np.uint8)
        
        # Add point-based fires
        if annotation.fire_points:
            point_mask = self.points_to_mask(annotation.fire_points)
            mask = np.logical_or(mask, point_mask).astype(np.uint8)
        
        # Add polygon-based fires
        if annotation.fire_polygons:
            polygon_mask = self.polygons_to_mask(annotation.fire_polygons)
            mask = np.logical_or(mask, polygon_mask).astype(np.uint8)
        
        return mask


class AutoFireDetector:
    """Automatic fire detection from satellite imagery using configuration."""
    
    def __init__(self, config: Optional[FireDetectionConfig] = None, 
                 data_config_path: Optional[str] = None):
        """
        Initialize fire detector with configuration.
        
        Args:
            config: Fire detection configuration object
            data_config_path: Path to YAML configuration file
        """
        if config is not None:
            self.config = config
        else:
            self.config = load_fire_detection_config(data_config_path)
        
        logger.info(f"AutoFireDetector initialized with parameters:")
        logger.info(f"  Red threshold: {self.config.red_threshold}")
        logger.info(f"  Orange ratio: {self.config.orange_ratio}")
        logger.info(f"  Brightness threshold: {self.config.brightness_threshold}")
        logger.info(f"  Green max: {self.config.green_max}")
        logger.info(f"  Blue max: {self.config.blue_max}")
        logger.info(f"  Min fire area: {self.config.min_fire_area}")
    
    def detect_fires(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect fire pixels using configured parameters.
        
        Args:
            image: RGB or RGBA image array
            
        Returns:
            List of (row, col) pixel coordinates of detected fires
        """
        return detect_fire_pixels_auto(image, self.config)
    
    def cluster_fire_pixels(self, fire_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Group nearby fire pixels into clusters."""
        if not fire_pixels:
            return []
        
        try:
            from sklearn.cluster import DBSCAN
            
            # Convert to numpy array
            pixels = np.array(fire_pixels)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=10,  # max distance between points
                min_samples=self.config.min_fire_area
            ).fit(pixels)
            
            # Group pixels by cluster
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(tuple(pixels[i]))
            
            return list(clusters.values())
        
        except ImportError:
            logger.warning("sklearn not available, using simple distance-based clustering")
            return self._simple_clustering(fire_pixels)
    
    def _simple_clustering(self, fire_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Simple distance-based clustering when sklearn is not available."""
        if not fire_pixels:
            return []
        
        clusters = []
        remaining_pixels = fire_pixels.copy()
        max_distance = 10
        
        while remaining_pixels:
            # Start a new cluster with the first remaining pixel
            current_cluster = [remaining_pixels.pop(0)]
            
            # Find all pixels within distance threshold
            i = 0
            while i < len(remaining_pixels):
                pixel = remaining_pixels[i]
                
                # Check distance to any pixel in current cluster
                min_dist = float('inf')
                for cluster_pixel in current_cluster:
                    dist = np.sqrt((pixel[0] - cluster_pixel[0])**2 + 
                                 (pixel[1] - cluster_pixel[1])**2)
                    min_dist = min(min_dist, dist)
                
                # If close enough, add to cluster
                if min_dist <= max_distance:
                    current_cluster.append(remaining_pixels.pop(i))
                else:
                    i += 1
            
            # Only keep clusters above minimum size
            if len(current_cluster) >= self.config.min_fire_area:
                clusters.append(current_cluster)
        
        return clusters
    
    def pixels_to_annotation(self, image_filename: str, state: str, date: datetime,
                           fire_pixels: List[Tuple[int, int]], 
                           geo_transformer: GeoTransformer) -> FireAnnotation:
        """Convert detected fire pixels to annotation format."""
        # Convert pixel coordinates to lat/lon
        fire_points = []
        for row, col in fire_pixels:
            try:
                lat, lon = geo_transformer.pixel_to_latlon(row, col)
                fire_points.append((lat, lon))
            except Exception as e:
                logger.warning(f"Could not convert pixel ({row}, {col}) to lat/lon: {e}")
        
        return FireAnnotation(
            image_filename=image_filename,
            state=state,
            date=date,
            fire_points=fire_points,
            fire_polygons=[],
            annotation_method='auto_optimized',
            annotator=f'auto_detector_r{self.config.red_threshold}_or{self.config.orange_ratio}',
            created_at=datetime.now(),
            confidence=0.85  # Higher confidence with optimized parameters
        )


class AnnotationManager:
    """Manage fire annotations with persistence."""
    
    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self._annotations = {}  # filename -> FireAnnotation
        self._load_all_annotations()
    
    def _get_annotation_path(self, filename: str) -> Path:
        """Get path for annotation file."""
        base_name = Path(filename).stem
        return self.annotations_dir / f"{base_name}_annotation.json"
    
    def _load_all_annotations(self):
        """Load all existing annotations."""
        for annotation_file in self.annotations_dir.glob("*_annotation.json"):
            try:
                self._load_annotation(annotation_file)
            except Exception as e:
                logger.warning(f"Could not load annotation {annotation_file}: {e}")
    
    def _load_annotation(self, annotation_path: Path):
        """Load single annotation from file."""
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Handle date parsing
        try:
            date = datetime.fromisoformat(data['date'])
        except (ValueError, KeyError):
            date = datetime.now()
            logger.warning(f"Could not parse date for {data.get('image_filename', 'unknown')}")
        
        try:
            created_at = datetime.fromisoformat(data['created_at'])
        except (ValueError, KeyError):
            created_at = datetime.now()
        
        annotation = FireAnnotation(
            image_filename=data['image_filename'],
            state=data['state'],
            date=date,
            fire_points=data.get('fire_points', []),
            fire_polygons=data.get('fire_polygons', []),
            annotation_method=data.get('annotation_method', 'manual'),
            annotator=data.get('annotator', 'unknown'),
            created_at=created_at,
            confidence=data.get('confidence', 1.0)
        )
        
        self._annotations[data['image_filename']] = annotation
    
    def save_annotation(self, annotation: FireAnnotation):
        """Save annotation to disk."""
        annotation_path = self._get_annotation_path(annotation.image_filename)
        
        data = {
            'image_filename': annotation.image_filename,
            'state': annotation.state,
            'date': annotation.date.isoformat(),
            'fire_points': annotation.fire_points,
            'fire_polygons': annotation.fire_polygons,
            'annotation_method': annotation.annotation_method,
            'annotator': annotation.annotator,
            'created_at': annotation.created_at.isoformat(),
            'confidence': annotation.confidence
        }
        
        with open(annotation_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._annotations[annotation.image_filename] = annotation
        logger.info(f"Saved annotation for {annotation.image_filename}")
    
    def get_annotation(self, filename: str) -> Optional[FireAnnotation]:
        """Get annotation for a specific image."""
        return self._annotations.get(filename)
    
    def has_annotation(self, filename: str) -> bool:
        """Check if annotation exists for image."""
        return filename in self._annotations
    
    def list_annotated_files(self) -> List[str]:
        """Get list of all annotated filenames."""
        return list(self._annotations.keys())
    
    def get_annotations_by_method(self, method: str) -> List[FireAnnotation]:
        """Get annotations filtered by method (manual, auto, hybrid)."""
        return [ann for ann in self._annotations.values() if ann.annotation_method == method]
    
    def get_statistics(self) -> Dict:
        """Get statistics about the annotations."""
        total_annotations = len(self._annotations)
        methods = {}
        states = {}
        total_fire_points = 0
        total_fire_polygons = 0
        
        for ann in self._annotations.values():
            methods[ann.annotation_method] = methods.get(ann.annotation_method, 0) + 1
            states[ann.state] = states.get(ann.state, 0) + 1
            total_fire_points += len(ann.fire_points)
            total_fire_polygons += len(ann.fire_polygons)
        
        return {
            'total_annotations': total_annotations,
            'methods': methods,
            'states': states,
            'total_fire_points': total_fire_points,
            'total_fire_polygons': total_fire_polygons,
            'avg_fire_points_per_image': total_fire_points / max(total_annotations, 1),
            'avg_fire_polygons_per_image': total_fire_polygons / max(total_annotations, 1)
        }


class MaskGenerator:
    """Generate training masks from annotations."""
    
    def __init__(self, annotation_manager: AnnotationManager):
        self.annotation_manager = annotation_manager
    
    def generate_mask_for_image(self, filename: str, geo_transformer: GeoTransformer) -> Optional[np.ndarray]:
        """Generate binary mask for a specific image."""
        annotation = self.annotation_manager.get_annotation(filename)
        if not annotation:
            logger.warning(f"No annotation found for {filename}")
            return None
        
        rasterizer = FireRasterizer(geo_transformer)
        return rasterizer.annotation_to_mask(annotation)
    
    def generate_all_masks(self, geo_transformer_factory, output_dir: str) -> Dict[str, str]:
        """Generate masks for all annotated images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_mapping = {}
        
        for filename in self.annotation_manager.list_annotated_files():
            try:
                geo_transformer = geo_transformer_factory(filename)
                mask = self.generate_mask_for_image(filename, geo_transformer)
                if mask is not None:
                    mask_filename = Path(filename).stem + "_mask.png"
                    mask_path = output_dir / mask_filename
                    cv2.imwrite(str(mask_path), mask * 255)
                    mask_mapping[filename] = str(mask_path)
                    logger.info(f"Generated mask: {mask_path}")
                
            except Exception as e:
                logger.error(f"Error generating mask for {filename}: {e}")
        
        return mask_mapping


def create_manual_annotation_template(filename: str, state: str, date: datetime) -> FireAnnotation:
    """Create empty annotation template for manual annotation."""
    return FireAnnotation(
        image_filename=filename,
        state=state,
        date=date,
        fire_points=[],
        fire_polygons=[],
        annotation_method='manual',
        annotator='human',
        created_at=datetime.now(),
        confidence=1.0
    )