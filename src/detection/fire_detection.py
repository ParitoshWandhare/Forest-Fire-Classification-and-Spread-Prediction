"""
Fire detection functions for satellite imagery.
Calibrated for specific satellite data with lower RGB values.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_fire_pixels_calibrated(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Fire detection calibrated for your specific satellite imagery.
    Based on analysis showing fires have lower RGB values than expected.
    """
    if image.shape[2] not in [3, 4]:
        raise ValueError("Image must be RGB (3 channels) or RGBA (4 channels)")
    
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    
    # Calibrated for your satellite data (much more permissive)
    red_threshold = 160        # Lower than 180 since only 180 worked
    orange_ratio = 0.7         # Lower than 0.8 for more sensitivity
    brightness_min = 350       # Lower brightness threshold
    
    # Basic fire criteria
    red_mask = red > red_threshold
    orange_mask = (green > 10) & ((red / (green + 1e-8)) > orange_ratio)  # Avoid division by tiny values
    bright_mask = (red + green + blue) > brightness_min
    
    # Additional helpful criteria
    red_dominance = red >= green  # Red should be at least as bright as green
    not_too_blue = blue < red * 1.2  # Fires shouldn't be predominantly blue
    
    # Combine criteria
    fire_mask = red_mask & orange_mask & bright_mask & red_dominance & not_too_blue
    
    # Optional: Remove isolated pixels (noise reduction)
    kernel = np.ones((2,2), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    fire_pixels = np.where(fire_mask)
    return list(zip(fire_pixels[0], fire_pixels[1]))


def detect_fire_pixels_very_sensitive(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Very sensitive fire detection - will catch more fires but also more false positives.
    Use this if the calibrated version misses obvious fires.
    """
    if image.shape[2] not in [3, 4]:
        raise ValueError("Image must be RGB (3 channels) or RGBA (4 channels)")
    
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    
    # Very permissive thresholds
    conditions = [
        red > 140,                           # Low red threshold
        red > green * 0.6,                   # Red should be somewhat dominant
        (red + green + blue) > 300,          # Low brightness threshold
        red > blue * 0.8,                    # Red should be brighter than blue
        (red - green) > -20,                 # Allow red to be slightly less than green
    ]
    
    # At least 4 out of 5 conditions must be true (allows some flexibility)
    condition_sum = sum(conditions)
    fire_mask = condition_sum >= 4
    
    fire_pixels = np.where(fire_mask)
    return list(zip(fire_pixels[0], fire_pixels[1]))


def analyze_specific_fire_pixels(image: np.ndarray, fire_coords: List[Tuple[int, int]]) -> None:
    """
    Analyze the actual RGB values of the orange spots you can see.
    """
    if not fire_coords:
        print("No fire coordinates provided")
        return
    
    print("Analyzing visible fire pixels:")
    for i, (row, col) in enumerate(fire_coords):
        if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
            r, g, b = image[row, col]
            brightness = int(r) + int(g) + int(b)
            rg_ratio = float(r) / (float(g) + 1e-8)
            rb_ratio = float(r) / (float(b) + 1e-8)
            
            print(f"Fire pixel {i+1} at ({row}, {col}): R={r}, G={g}, B={b}")
            print(f"  Brightness: {brightness}, R/G ratio: {rg_ratio:.2f}, R/B ratio: {rb_ratio:.2f}")
    
    # Get average values
    if fire_coords:
        fire_pixels = np.array([image[row, col] for row, col in fire_coords if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]])
        if len(fire_pixels) > 0:
            print(f"\nAverage fire pixel values:")
            print(f"R: {fire_pixels[:, 0].mean():.1f} (min: {fire_pixels[:, 0].min()}, max: {fire_pixels[:, 0].max()})")
            print(f"G: {fire_pixels[:, 1].mean():.1f} (min: {fire_pixels[:, 1].min()}, max: {fire_pixels[:, 1].max()})")
            print(f"B: {fire_pixels[:, 2].mean():.1f} (min: {fire_pixels[:, 2].min()}, max: {fire_pixels[:, 2].max()})")


def find_optimal_thresholds(image: np.ndarray, known_fire_coords: List[Tuple[int, int]]):
    """
    Test different threshold combinations and see which ones capture your known fire pixels.
    """
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    
    # Test ranges
    red_thresholds = [140, 150, 160, 170, 180]
    orange_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    brightness_thresholds = [250, 300, 350, 400, 450]
    
    best_score = 0
    best_params = None
    
    for red_thresh in red_thresholds:
        for orange_ratio in orange_ratios:
            for bright_thresh in brightness_thresholds:
                # Test this combination
                red_mask = red > red_thresh
                orange_mask = (green > 0) & ((red / (green + 1e-8)) > orange_ratio)
                bright_mask = (red + green + blue) > bright_thresh
                
                fire_mask = red_mask & orange_mask & bright_mask
                
                # Count how many known fire pixels this detects
                detected_fires = sum(1 for row, col in known_fire_coords 
                                   if fire_mask[row, col])
                total_detected = np.sum(fire_mask)
                
                # Score: prioritize finding known fires, penalize too many detections
                score = detected_fires - (total_detected - detected_fires) * 0.1
                
                if score > best_score:
                    best_score = score
                    best_params = (red_thresh, orange_ratio, bright_thresh)
                    
                if detected_fires == len(known_fire_coords):
                    print(f"Perfect match found: R>{red_thresh}, RG>{orange_ratio}, B>{bright_thresh}")
                    print(f"Detected {detected_fires}/{len(known_fire_coords)} known fires, {total_detected} total pixels")
    
    if best_params:
        print(f"\nBest parameters: R>{best_params[0]}, RG>{best_params[1]:.1f}, B>{best_params[2]}")
        print(f"Score: {best_score}")
    
    return best_params


if __name__ == "__main__":
    # Test with your specific image
    image_path = "data/interim/tiles/images/Ontario-2023-06-02_r004_c006.png"
    
    if Path(image_path).exists():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Manually specify coordinates of visible fire pixels (you'll need to identify these)
        # Look at your image and estimate coordinates of the orange spots
        known_fire_coords = [
            (50, 100),   # Replace with actual coordinates of visible fires
            (150, 200),  # Replace with actual coordinates of visible fires
            (300, 250),  # Replace with actual coordinates of visible fires
        ]
        
        print("Step 1: Analyzing visible fire pixels")
        analyze_specific_fire_pixels(image, known_fire_coords)
        
        print("\nStep 2: Finding optimal thresholds")
        best_params = find_optimal_thresholds(image, known_fire_coords)
        
        print("\nStep 3: Testing calibrated detection")
        fire_pixels = detect_fire_pixels_calibrated(image)
        print(f"Calibrated detection found {len(fire_pixels)} fire pixels")
        
        print("\nStep 4: Testing very sensitive detection")
        sensitive_fire_pixels = detect_fire_pixels_very_sensitive(image)
        print(f"Sensitive detection found {len(sensitive_fire_pixels)} fire pixels")
        
    else:
        print(f"Image not found at {image_path}")
        print("Please update the image_path variable")