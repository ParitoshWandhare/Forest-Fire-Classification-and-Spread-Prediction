#!/usr/bin/env python3
"""
Data preparation script for forest fire detection.
This script:
1. Loads raw satellite images and metadata
2. Performs automatic fire detection (or loads manual annotations)
3. Creates fire masks
4. Generates training tiles
5. Splits data into train/val/test sets
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_io.readers import MetadataReader, GeoTIFFReader
from data_io.geo import GeoTransformer
from preprocessing.rasterize import (
    AutoFireDetector, AnnotationManager, MaskGenerator,
    FireRasterizer, create_manual_annotation_template
)
from preprocessing.tiling import ImageTiler, TileDataset


def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_preparation.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_single_image(filename: str, state: str, metadata_reader: MetadataReader,
                        geotiff_reader: GeoTIFFReader, auto_detector: AutoFireDetector,
                        annotation_manager: AnnotationManager, config: dict) -> bool:
    """
    Process a single image: load, detect fires, create annotation.
    
    Returns:
        bool: True if processing was successful
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check if annotation already exists
        if annotation_manager.has_annotation(filename):
            logger.info(f"Annotation already exists for {filename}, skipping")
            return True
        
        # Get metadata
        img_metadata = metadata_reader.get_metadata(filename)
        if not img_metadata:
            logger.warning(f"No metadata found for {filename}")
            return False
        
        # Load image
        logger.info(f"Processing {filename}...")
        image, raster_metadata = geotiff_reader.read_image(filename, state)
        logger.info(f"Loaded image: {image.shape}")
        
        # Set up geo transformer
        bounds = img_metadata.get_bounds()
        geo_transformer = GeoTransformer(bounds, image.shape[:2])
        
        # Detect fires automatically
        fire_pixels = auto_detector.detect_fires(image)
        logger.info(f"Detected {len(fire_pixels)} fire pixels")
        
        # Create annotation
        if len(fire_pixels) > 0:
            annotation = auto_detector.pixels_to_annotation(
                filename, state, img_metadata.date, fire_pixels, geo_transformer
            )
            
            # Save annotation
            annotation_manager.save_annotation(annotation)
            logger.info(f"Saved annotation with {len(annotation.fire_points)} fire points")
        else:
            # Create empty annotation for images with no fire
            annotation = create_manual_annotation_template(filename, state, img_metadata.date)
            annotation.annotation_method = 'auto'
            annotation.annotator = 'auto_detector'
            annotation.created_at = datetime.now()
            annotation.confidence = 0.9
            
            annotation_manager.save_annotation(annotation)
            logger.info("Saved empty annotation (no fires detected)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return False


def generate_tiles_for_image(filename: str, state: str, metadata_reader: MetadataReader,
                           geotiff_reader: GeoTIFFReader, annotation_manager: AnnotationManager,
                           mask_generator: MaskGenerator, tiler: ImageTiler,
                           output_dir: str) -> list:
    """Generate tiles for a single image."""
    logger = logging.getLogger(__name__)
    
    try:
        # Get metadata and load image
        img_metadata = metadata_reader.get_metadata(filename)
        image, _ = geotiff_reader.read_image(filename, state)
        
        # Set up geo transformer
        bounds = img_metadata.get_bounds()
        geo_transformer = GeoTransformer(bounds, image.shape[:2])
        
        # Generate mask
        mask = mask_generator.generate_mask_for_image(filename, geo_transformer)
        if mask is None:
            logger.warning(f"No mask generated for {filename}")
            return []
        
        # Generate tiles
        tiles = tiler.generate_tiles(image, mask, filename)
        
        # Save tiles to disk
        tiler.save_tiles_to_disk(image, mask, tiles, output_dir, save_all=False)
        
        return tiles
        
    except Exception as e:
        logger.error(f"Error generating tiles for {filename}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare forest fire detection data")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--data-config", required=True, help="Path to data configuration")
    parser.add_argument("--state", default="Ontario", help="State to process")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--skip-annotations", action="store_true", 
                       help="Skip annotation generation (use existing)")
    parser.add_argument("--skip-tiling", action="store_true", 
                       help="Skip tile generation (use existing)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data preparation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data config: {args.data_config}")
    
    # Load configurations
    try:
        config = load_config(args.config)
        data_config = load_config(args.data_config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Initialize readers
    try:
        metadata_reader = MetadataReader(data_config['data']['metadata_csv'])
        geotiff_reader = GeoTIFFReader(data_config['data']['images_base_dir'])
        logger.info("Initialized data readers")
    except Exception as e:
        logger.error(f"Error initializing readers: {e}")
        return 1
    
    # Get list of images to process
    if args.start_date and args.end_date:
        image_files = metadata_reader.get_date_range_files(
            args.start_date, args.end_date, args.state
        )
    else:
        image_files = metadata_reader.get_state_files(args.state)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    if not image_files:
        logger.warning("No images found to process")
        return 0
    
    # Initialize annotation components
    if not args.skip_annotations:
        annotation_manager = AnnotationManager(data_config['data']['interim_dir'] + "/annotations")
        
        # auto_detector_config = data_config['data'].get('fire_detection', {})
        # auto_detector = AutoFireDetector({
        #     'red_threshold': auto_detector_config.get('rgb_thresholds', {}).get('red_min', 200),
        #     'orange_ratio': auto_detector_config.get('rgb_thresholds', {}).get('orange_threshold', 0.7),
        #     'min_cluster_size': 3,
        #     'max_cluster_distance': 10
        # })
        auto_detector = AutoFireDetector(data_config_path=args.data_config)
        
        # Process images for annotation
        logger.info("Starting fire detection and annotation...")
        successful_annotations = 0
        
        for filename in tqdm(image_files, desc="Processing images"):
            success = process_single_image(
                filename, args.state, metadata_reader, geotiff_reader,
                auto_detector, annotation_manager, data_config
            )
            if success:
                successful_annotations += 1
        
        logger.info(f"Successfully processed {successful_annotations}/{len(image_files)} images")
        
        # Print annotation statistics
        all_annotations = annotation_manager.list_annotated_files()
        auto_annotations = annotation_manager.get_annotations_by_method('auto')
        logger.info(f"Total annotations: {len(all_annotations)}")
        logger.info(f"Auto annotations: {len(auto_annotations)}")
    
    else:
        annotation_manager = AnnotationManager(data_config['data']['interim_dir'] + "/annotations")
        logger.info("Skipping annotation generation, using existing annotations")
    
    # Generate tiles
    if not args.skip_tiling:
        logger.info("Starting tile generation...")
        
        # Initialize tiling components
        mask_generator = MaskGenerator(annotation_manager)
        
        tiler = ImageTiler(
            tile_size=data_config['data']['tile_size'],
            overlap=data_config['data']['tile_overlap'],
            min_fire_pixels=data_config['data']['min_fire_pixels']
        )
        
        tiles_output_dir = data_config['data']['tiles_dir']
        tile_dataset = TileDataset(tiles_output_dir)
        
        # Process annotated images for tiling
        annotated_files = annotation_manager.list_annotated_files()
        if args.max_images:
            annotated_files = annotated_files[:args.max_images]
        
        all_tiles = []
        
        for filename in tqdm(annotated_files, desc="Generating tiles"):
            if filename in [f for f in image_files]:  # Only process files in our target list
                tiles = generate_tiles_for_image(
                    filename, args.state, metadata_reader, geotiff_reader,
                    annotation_manager, mask_generator, tiler, tiles_output_dir
                )
                all_tiles.extend(tiles)
        
        # Add tiles to dataset
        if all_tiles:
            tile_dataset.add_tiles(all_tiles)
            
            # Print tile statistics
            stats = tile_dataset.get_statistics()
            logger.info("Tile generation completed")
            logger.info(f"Total tiles: {stats['total_tiles']}")
            logger.info(f"Fire tiles: {stats['fire_tiles']}")
            logger.info(f"No-fire tiles: {stats['no_fire_tiles']}")
            logger.info(f"Fire ratio: {stats['fire_ratio']:.3f}")
            
    else:
        logger.info("Skipping tile generation")
    
    logger.info("Data preparation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())