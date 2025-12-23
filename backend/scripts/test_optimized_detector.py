#!/usr/bin/env python3
"""Test script for the optimized wall detector.

Run this script to test the wall detection on a sample floor plan image.

Usage:
    python test_optimized_detector.py <image_path>
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from app.services.optimized_wall_detector import OptimizedWallDetector


def test_detector(image_path: str, scale: float = 0.05):
    """Test the optimized wall detector on an image."""
    print(f"\n{'='*60}")
    print(f"Testing Optimized Wall Detector")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Scale: {scale} m/px")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h} pixels")
    
    # Initialize detector
    detector = OptimizedWallDetector(scale=scale)
    
    # Detect walls
    print(f"\nDetecting walls...")
    walls, rooms = detector.detect_walls(image, scale)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Walls detected: {len(walls)}")
    print(f"Rooms detected: {len(rooms)}")
    
    if walls:
        # Thickness statistics
        thicknesses = [w.thickness for w in walls]
        print(f"\nWall Thickness Statistics:")
        print(f"  Min: {min(thicknesses):.3f}m ({min(thicknesses)/scale:.1f}px)")
        print(f"  Max: {max(thicknesses):.3f}m ({max(thicknesses)/scale:.1f}px)")
        print(f"  Avg: {np.mean(thicknesses):.3f}m ({np.mean(thicknesses)/scale:.1f}px)")
        print(f"  Median: {np.median(thicknesses):.3f}m ({np.median(thicknesses)/scale:.1f}px)")
        
        # Material distribution
        materials = {}
        for wall in walls:
            mat = wall.material
            materials[mat] = materials.get(mat, 0) + 1
        
        print(f"\nMaterial Distribution:")
        for mat, count in sorted(materials.items(), key=lambda x: -x[1]):
            print(f"  {mat}: {count} walls ({100*count/len(walls):.1f}%)")
        
        # Wall length statistics
        lengths = []
        for wall in walls:
            dx = wall.end.x - wall.start.x
            dy = wall.end.y - wall.start.y
            length_px = np.sqrt(dx*dx + dy*dy)
            length_m = length_px * scale
            lengths.append(length_m)
        
        print(f"\nWall Length Statistics:")
        print(f"  Min: {min(lengths):.2f}m")
        print(f"  Max: {max(lengths):.2f}m")
        print(f"  Total: {sum(lengths):.2f}m")
        
        # Sample walls
        print(f"\nSample Walls (first 5):")
        for i, wall in enumerate(walls[:5]):
            print(f"  [{i}] ({wall.start.x:.0f},{wall.start.y:.0f}) -> ({wall.end.x:.0f},{wall.end.y:.0f})")
            print(f"       thickness={wall.thickness:.3f}m, material={wall.material}, attenuation={wall.attenuation_db}dB")
    
    # Visualize results
    output_path = image_path.replace('.', '_detected.')
    visualize_results(image, walls, rooms, output_path)
    print(f"\nVisualization saved to: {output_path}")
    
    return walls, rooms


def visualize_results(image: np.ndarray, walls, rooms, output_path: str):
    """Visualize detected walls on the image."""
    result = image.copy()
    
    # Draw walls with thickness-based colors
    for wall in walls:
        # Color based on material
        colors = {
            'concrete': (0, 0, 255),    # Red
            'brick': (0, 128, 255),      # Orange
            'drywall': (0, 255, 0),      # Green
            'glass': (255, 255, 0),      # Cyan
            'wood': (0, 165, 255),       # Orange
            'metal': (255, 0, 255),      # Magenta
        }
        color = colors.get(wall.material, (128, 128, 128))
        
        # Line thickness based on wall thickness (scaled)
        line_thickness = max(1, int(wall.thickness / 0.05))  # 1px per 5cm
        
        pt1 = (int(wall.start.x), int(wall.start.y))
        pt2 = (int(wall.end.x), int(wall.end.y))
        cv2.line(result, pt1, pt2, color, line_thickness)
    
    # Add legend
    y = 30
    for material, color in [('concrete', (0,0,255)), ('brick', (0,128,255)), 
                            ('drywall', (0,255,0)), ('glass', (255,255,0))]:
        cv2.line(result, (10, y), (40, y), color, 3)
        cv2.putText(result, material, (50, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 25
    
    cv2.imwrite(output_path, result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_optimized_detector.py <image_path> [scale]")
        print("\nExample:")
        print("  python test_optimized_detector.py floor_plan.png 0.05")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    
    test_detector(image_path, scale)
