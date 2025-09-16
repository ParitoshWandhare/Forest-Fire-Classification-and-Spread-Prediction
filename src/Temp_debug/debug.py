"""
Debug script to test and visualize fire detection thresholds.
Use this to find optimal parameters for your specific satellite imagery.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_fire_regions(image: np.ndarray, 
                        sample_coords: list = None,
                        window_size: int = 50):
    """
    Analyze color values in known fire regions to determine optimal thresholds.
    
    Args:
        image: RGB image array
        sample_coords: List of (row, col) coordinates of known fire pixels
        window_size: Size of window around each coordinate to analyze
    """
    
    if sample_coords is None:
        # If no coordinates provided, let user click on fire spots
        print("Click on fire pixels in the displayed image, then close the window")
        
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                col, row = int(event.xdata), int(event.ydata)
                sample_coords.append((row, col))
                print(f"Added fire pixel at ({row}, {col})")
        
        sample_coords = []
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title("Click on fire pixels (orange/red spots)")
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    if not sample_coords:
        print("No fire coordinates provided")
        return
    
    fire_pixels = []
    for row, col in sample_coords:
        # Extract window around fire pixel
        r_start = max(0, row - window_size//2)
        r_end = min(image.shape[0], row + window_size//2)
        c_start = max(0, col - window_size//2)  
        c_end = min(image.shape[1], col + window_size//2)
        
        window = image[r_start:r_end, c_start:c_end]
        # Get pixels with high red values (likely fire)
        red_pixels = window[window[:,:,0] > 150]
        fire_pixels.extend(red_pixels)
    
    fire_pixels = np.array(fire_pixels)
    
    if len(fire_pixels) == 0:
        print("No fire pixels found")
        return
    
    print(f"Analyzed {len(fire_pixels)} fire pixels")
    print("\nFire Pixel Statistics:")
    print(f"Red channel   - Mean: {fire_pixels[:,0].mean():.1f}, Std: {fire_pixels[:,0].std():.1f}, Min: {fire_pixels[:,0].min()}, Max: {fire_pixels[:,0].max()}")
    print(f"Green channel - Mean: {fire_pixels[:,1].mean():.1f}, Std: {fire_pixels[:,1].std():.1f}, Min: {fire_pixels[:,1].min()}, Max: {fire_pixels[:,1].max()}")
    print(f"Blue channel  - Mean: {fire_pixels[:,2].mean():.1f}, Std: {fire_pixels[:,2].std():.1f}, Min: {fire_pixels[:,2].min()}, Max: {fire_pixels[:,2].max()}")
    
    # Calculate ratios
    red_green_ratio = fire_pixels[:,0] / (fire_pixels[:,1] + 1e-8)
    red_blue_ratio = fire_pixels[:,0] / (fire_pixels[:,2] + 1e-8)
    brightness = np.sum(fire_pixels, axis=1)
    
    print(f"\nRed/Green ratio - Mean: {red_green_ratio.mean():.2f}, Min: {red_green_ratio.min():.2f}")
    print(f"Red/Blue ratio  - Mean: {red_blue_ratio.mean():.2f}, Min: {red_blue_ratio.min():.2f}")
    print(f"Brightness      - Mean: {brightness.mean():.1f}, Min: {brightness.min()}")
    
    print(f"\nRecommended thresholds:")
    print(f"red_threshold: {int(fire_pixels[:,0].mean() - fire_pixels[:,0].std())}")
    print(f"red_green_ratio: {red_green_ratio.mean() * 0.8:.2f}")
    print(f"red_blue_ratio: {red_blue_ratio.mean() * 0.8:.2f}")
    print(f"brightness_threshold: {int(brightness.mean() - brightness.std())}")


def test_fire_detection_thresholds(image: np.ndarray,
                                 red_thresholds: list = [170, 200, 220],
                                 orange_ratios: list = [0.5, 1.2, 1.5, 2.0],
                                 brightness_thresholds: list = [250, 500, 600]):
    """
    Test different threshold combinations and show results.
    """
    
    fig, axes = plt.subplots(len(red_thresholds), len(orange_ratios), 
                            figsize=(16, 12))
    
    if len(red_thresholds) == 1:
        axes = axes.reshape(1, -1)
    if len(orange_ratios) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, red_thresh in enumerate(red_thresholds):
        for j, orange_ratio in enumerate(orange_ratios):
            # Test detection
            red = image[:, :, 0].astype(float)
            green = image[:, :, 1].astype(float)
            blue = image[:, :, 2].astype(float)
            
            red_mask = red > red_thresh
            orange_mask = (green > 0) & ((red / (green + 1e-8)) > orange_ratio)
            bright_mask = (red + green + blue) > brightness_thresholds[0]
            
            fire_mask = red_mask & orange_mask & bright_mask
            
            # Create visualization
            overlay = image.copy()
            overlay[fire_mask] = [255, 0, 255]  # Magenta for detected fire
            
            ax = axes[i, j] if len(red_thresholds) > 1 and len(orange_ratios) > 1 else axes[max(i,j)]
            ax.imshow(overlay)
            ax.set_title(f'R>{red_thresh}, RG>{orange_ratio:.1f}\nFire pixels: {np.sum(fire_mask)}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_conservative_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a very conservative fire mask that only detects obvious fires.
    """
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float) 
    blue = image[:, :, 2].astype(float)
    
    # Very strict criteria
    conditions = [
        red > 200,                          # High red
        red > green * 1.8,                  # Red dominates green  
        red > blue * 1.5,                   # Red dominates blue
        (red + green + blue) > 550,         # Very bright
        (red - green) > 40,                 # Strong red-green contrast
        red > 220                           # Even higher red threshold
    ]
    
    fire_mask = np.all(conditions, axis=0)
    
    # Remove small isolated pixels (noise)
    kernel = np.ones((3,3), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    return fire_mask.astype(bool)


if __name__ == "__main__":
    # Example usage with your satellite image
    
    # Load your image (replace with actual path)
    image_path = "data/interim/tiles/images/Ontario-2023-06-02_r004_c006.png"
    
    if Path(image_path).exists():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("Testing conservative fire detection...")
        conservative_mask = create_conservative_mask(image)
        print(f"Conservative detection found {np.sum(conservative_mask)} fire pixels")
        
        # Visualize result
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(conservative_mask, cmap='hot')
        ax2.set_title(f"Conservative Mask\n{np.sum(conservative_mask)} fire pixels")
        ax2.axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[conservative_mask] = [255, 0, 255]
        ax3.imshow(overlay)
        ax3.set_title("Fire Detection Overlay")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Test different thresholds
        print("\nTesting different threshold combinations...")
        test_fire_detection_thresholds(image)
        
    else:
        print(f"Image not found at {image_path}")
        print("Please update the image_path variable with your actual image location")