"""Map processing service for extracting walls from floor plan images."""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import logging

from app.schemas.project import MapData, WallSegment, Room, Point, MapDimensions, ForbiddenZone

logger = logging.getLogger(__name__)

# Material attenuation database (IEEE 802.11 standards)
MATERIAL_ATTENUATION = {
    'concrete': 15.0,  # dB
    'brick': 12.0,
    'wood': 6.0,
    'glass': 5.0,
    'drywall': 3.0,
    'metal': 25.0,
    'unknown': 10.0
}


@dataclass
class Opening:
    """Door or window opening in a wall."""
    start: Point
    end: Point
    opening_type: str  # 'door' or 'window'
    width: float


class EnhancedWallDetector:
    """Multi-strategy wall detection with improved accuracy."""

    def __init__(self, scale: float = 0.05):
        self.scale = scale

    def detect_walls(self, image: np.ndarray) -> List[WallSegment]:
        """
        Detect walls using improved single-strategy approach for accuracy.

        Uses precise Hough Transform with axis snapping and duplicate removal.
        Multiple strategies were causing alignment issues due to averaging.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of detected wall segments
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Preprocess WITHOUT deskew to maintain coordinate alignment
        processed = preprocess_image(gray, apply_deskew=False)

        # Use the improved extract_walls function for better accuracy
        walls = extract_walls(processed, self.scale)

        return walls

    def _detect_walls_hough(self, gray: np.ndarray) -> List[WallSegment]:
        """Detect walls using improved Hough transform."""
        # Multi-scale edge detection
        edges_low = cv2.Canny(gray, 30, 100)
        edges_high = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges_low, edges_high)

        # Morphological closing to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=25,
            maxLineGap=15
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length < 15:
                continue

            thickness = estimate_wall_thickness(gray, x1, y1, x2, y2, self.scale)

            walls.append(WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=thickness,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            ))

        return walls

    def _detect_walls_contours(self, gray: np.ndarray) -> List[WallSegment]:
        """Detect walls from thick lines using contour analysis."""
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        walls = []
        for contour in contours:
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            width, height = rect[1]

            # Check if elongated (wall-like)
            if min(width, height) < 3:
                continue

            aspect_ratio = max(width, height) / (min(width, height) + 0.001)

            if aspect_ratio > 3:  # Wall-like shape
                # Get centerline
                center = rect[0]
                angle = rect[2]

                if width < height:
                    length = height
                    angle += 90
                else:
                    length = width

                # Convert angle to radians
                angle_rad = math.radians(angle)

                # Calculate endpoints
                dx = (length / 2) * math.cos(angle_rad)
                dy = (length / 2) * math.sin(angle_rad)

                x1, y1 = center[0] - dx, center[1] - dy
                x2, y2 = center[0] + dx, center[1] + dy

                walls.append(WallSegment(
                    start=Point(x=float(x1), y=float(y1)),
                    end=Point(x=float(x2), y=float(y2)),
                    thickness=float(min(width, height)) * self.scale,
                    material='concrete',
                    attenuation_db=MATERIAL_ATTENUATION['concrete']
                ))

        return walls

    def _detect_walls_adaptive(self, gray: np.ndarray) -> List[WallSegment]:
        """Detect walls using adaptive thresholding."""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 5
        )

        # Morphological operations
        kernel_h = np.ones((1, 15), np.uint8)
        kernel_v = np.ones((15, 1), np.uint8)

        # Detect horizontal lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

        # Detect vertical lines
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        # Combine
        combined = cv2.bitwise_or(horizontal, vertical)

        # Detect lines
        lines = cv2.HoughLinesP(
            combined,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length < 15:
                continue

            walls.append(WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            ))

        return walls

    def _merge_wall_detections(self, wall_lists: List[List[WallSegment]]) -> List[WallSegment]:
        """Merge walls from multiple detection methods."""
        all_walls = []
        for walls in wall_lists:
            all_walls.extend(walls)

        if len(all_walls) <= 1:
            return all_walls

        # Remove duplicates based on proximity
        unique_walls = []
        used = set()

        for i, wall1 in enumerate(all_walls):
            if i in used:
                continue

            # Find similar walls
            similar = [wall1]
            used.add(i)

            for j, wall2 in enumerate(all_walls):
                if j in used:
                    continue

                if self._walls_similar(wall1, wall2):
                    similar.append(wall2)
                    used.add(j)

            # Average similar walls
            if len(similar) > 1:
                unique_walls.append(self._average_walls(similar))
            else:
                unique_walls.append(wall1)

        return unique_walls

    def _walls_similar(self, wall1: WallSegment, wall2: WallSegment, threshold: float = 10.0) -> bool:
        """Check if two walls are essentially the same."""
        # Check if endpoints are close
        d1 = math.sqrt((wall1.start.x - wall2.start.x)**2 + (wall1.start.y - wall2.start.y)**2)
        d2 = math.sqrt((wall1.end.x - wall2.end.x)**2 + (wall1.end.y - wall2.end.y)**2)
        d3 = math.sqrt((wall1.start.x - wall2.end.x)**2 + (wall1.start.y - wall2.end.y)**2)
        d4 = math.sqrt((wall1.end.x - wall2.start.x)**2 + (wall1.end.y - wall2.start.y)**2)

        return (d1 < threshold and d2 < threshold) or (d3 < threshold and d4 < threshold)

    def _average_walls(self, walls: List[WallSegment]) -> WallSegment:
        """Average multiple similar walls."""
        avg_start_x = sum(w.start.x for w in walls) / len(walls)
        avg_start_y = sum(w.start.y for w in walls) / len(walls)
        avg_end_x = sum(w.end.x for w in walls) / len(walls)
        avg_end_y = sum(w.end.y for w in walls) / len(walls)
        avg_thickness = sum(w.thickness for w in walls) / len(walls)

        return WallSegment(
            start=Point(x=avg_start_x, y=avg_start_y),
            end=Point(x=avg_end_x, y=avg_end_y),
            thickness=avg_thickness,
            material=walls[0].material,
            attenuation_db=walls[0].attenuation_db
        )

    def detect_openings(self, image: np.ndarray, walls: List[WallSegment]) -> List[Opening]:
        """
        Detect door and window openings in walls.

        Looks for:
        1. Gaps in walls
        2. Door swing arcs
        3. Window symbols

        Args:
            image: Input image
            walls: Detected walls

        Returns:
            List of detected openings
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        openings = []

        # Detect door swing arcs
        door_openings = self._detect_door_arcs(gray)
        openings.extend(door_openings)

        # Detect gaps in walls
        gap_openings = self._detect_wall_gaps(gray, walls)
        openings.extend(gap_openings)

        return openings

    def _detect_door_arcs(self, gray: np.ndarray) -> List[Opening]:
        """Detect door swing arcs (quarter circles)."""
        openings = []

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=80
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # Door swing has radius approximately equal to door width
                width = float(r) * self.scale

                openings.append(Opening(
                    start=Point(x=float(x), y=float(y)),
                    end=Point(x=float(x + r), y=float(y)),
                    opening_type='door',
                    width=width
                ))

        return openings

    def _detect_wall_gaps(self, gray: np.ndarray, walls: List[WallSegment]) -> List[Opening]:
        """Detect gaps in detected walls that could be openings."""
        openings = []

        # Find wall endpoints that are close but not connected
        endpoints = []
        for wall in walls:
            endpoints.append((wall.start.x, wall.start.y, wall))
            endpoints.append((wall.end.x, wall.end.y, wall))

        # Check for nearby endpoints on same line
        for i, (x1, y1, wall1) in enumerate(endpoints):
            for j, (x2, y2, wall2) in enumerate(endpoints):
                if i >= j or wall1 == wall2:
                    continue

                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Check if walls are collinear
                if are_collinear(wall1, wall2, tolerance=10.0):
                    if 20 < dist < 150:  # Reasonable door/window size
                        openings.append(Opening(
                            start=Point(x=x1, y=y1),
                            end=Point(x=x2, y=y2),
                            opening_type='door' if dist > 60 else 'window',
                            width=dist * self.scale
                        ))

        return openings


async def process_map_image(image_path: str, scale: float) -> MapData:
    """
    Process uploaded floor plan image to extract walls and rooms.

    Args:
        image_path: Path to the uploaded image
        scale: Meters per pixel

    Returns:
        MapData object with extracted features
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get dimensions
    height, width = gray.shape

    # Preprocess
    processed = preprocess_image(gray)

    # Extract walls
    walls = extract_walls(processed, scale)

    # Segment rooms
    rooms = segment_rooms(processed, walls, scale)

    return MapData(
        dimensions=MapDimensions(width=width, height=height),
        walls=walls,
        rooms=rooms,
        forbidden_zones=[]
    )


def preprocess_image(gray: np.ndarray, apply_deskew: bool = False) -> np.ndarray:
    """
    Preprocess grayscale image for wall detection.

    Steps:
    1. Denoise using bilateral filter
    2. Enhance contrast with CLAHE
    3. Optionally deskew (disabled by default to preserve alignment)
    """
    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Deskew only if explicitly requested (can cause coordinate misalignment)
    if apply_deskew:
        enhanced = deskew_image(enhanced)

    return enhanced


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Detect and correct rotation of floor plan."""
    # Detect edges
    edges = cv2.Canny(img, 50, 150)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None or len(lines) < 5:
        return img

    # Find dominant angle
    angles = []
    for line in lines[:50]:  # Limit to first 50 lines
        rho, theta = line[0]
        angle = np.degrees(theta)
        # Normalize to [-45, 45]
        if angle > 45:
            angle = angle - 90
        if angle < -45:
            angle = angle + 90
        angles.append(angle)

    # Get median angle
    median_angle = np.median(angles)

    # Rotate image if needed (only for significant rotation)
    if abs(median_angle) > 1:
        center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        img = cv2.warpAffine(
            img, rot_mat, img.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    return img


def extract_walls(gray: np.ndarray, scale: float) -> List[WallSegment]:
    """
    Extract wall segments from preprocessed image with improved accuracy.

    Uses multiple thresholding + Hough line transform with strict parameters.
    """
    height, width = gray.shape

    # Determine if image is predominantly white (typical floor plan) or dark
    mean_intensity = np.mean(gray)
    is_light_background = mean_intensity > 127

    if is_light_background:
        # Floor plan with black lines on white background
        # Use Otsu's thresholding to find optimal threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Inverted floor plan or dark background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up noise with morphological operations
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # Thin the lines to get skeleton (better line detection)
    # This helps get the centerline of thick walls
    thinned = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary

    # Edge detection on both binary and thinned
    edges1 = cv2.Canny(binary, 50, 150, apertureSize=3)
    edges2 = cv2.Canny(thinned, 30, 100, apertureSize=3) if thinned is not binary else np.zeros_like(binary)
    edges = cv2.bitwise_or(edges1, edges2)

    # Calculate adaptive parameters based on image size
    min_line_length = max(20, min(width, height) // 40)  # At least 2.5% of smaller dimension
    max_line_gap = max(5, min_line_length // 4)
    threshold = max(30, min_line_length)

    # Detect lines using probabilistic Hough transform with stricter parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    # Convert to wall segments with snapping
    walls = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate line length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Filter out very short lines (likely noise)
        if length < min_line_length:
            continue

        # Snap nearly horizontal/vertical lines to exact H/V (within 3 degrees)
        x1, y1, x2, y2 = snap_to_axis(x1, y1, x2, y2, angle_threshold=3.0)

        # Calculate wall thickness based on line intensity
        thickness = estimate_wall_thickness(gray, x1, y1, x2, y2, scale)

        # Default material (user can edit later)
        material = "concrete"

        wall = WallSegment(
            start=Point(x=float(x1), y=float(y1)),
            end=Point(x=float(x2), y=float(y2)),
            thickness=thickness,
            material=material,
            attenuation_db=MATERIAL_ATTENUATION[material]
        )
        walls.append(wall)

    # Merge collinear walls (stricter tolerance)
    walls = merge_collinear_walls(walls, tolerance=3.0)

    # Remove duplicate walls that are very close
    walls = remove_duplicate_walls(walls, distance_threshold=5.0)

    return walls


def snap_to_axis(x1: int, y1: int, x2: int, y2: int, angle_threshold: float = 3.0) -> Tuple[int, int, int, int]:
    """
    Snap nearly horizontal or vertical lines to exact horizontal/vertical.

    Args:
        x1, y1, x2, y2: Line endpoints
        angle_threshold: Maximum angle deviation (degrees) to snap

    Returns:
        Snapped line endpoints
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return x1, y1, x2, y2

    angle = math.degrees(math.atan2(abs(dy), abs(dx)))

    # Near horizontal (angle close to 0)
    if angle < angle_threshold:
        # Snap to horizontal - use average y
        avg_y = (y1 + y2) // 2
        return x1, avg_y, x2, avg_y

    # Near vertical (angle close to 90)
    if angle > (90 - angle_threshold):
        # Snap to vertical - use average x
        avg_x = (x1 + x2) // 2
        return avg_x, y1, avg_x, y2

    return x1, y1, x2, y2


def remove_duplicate_walls(walls: List[WallSegment], distance_threshold: float = 5.0) -> List[WallSegment]:
    """
    Remove walls that are essentially duplicates (very close to each other).
    """
    if len(walls) < 2:
        return walls

    unique_walls = []
    used = set()

    for i, wall1 in enumerate(walls):
        if i in used:
            continue

        # Check against all other walls
        is_duplicate = False
        for j, wall2 in enumerate(walls):
            if j <= i or j in used:
                continue

            # Calculate distance between wall midpoints
            mid1_x = (wall1.start.x + wall1.end.x) / 2
            mid1_y = (wall1.start.y + wall1.end.y) / 2
            mid2_x = (wall2.start.x + wall2.end.x) / 2
            mid2_y = (wall2.start.y + wall2.end.y) / 2

            midpoint_dist = math.sqrt((mid2_x - mid1_x)**2 + (mid2_y - mid1_y)**2)

            # Check if walls have similar lengths
            len1 = math.sqrt((wall1.end.x - wall1.start.x)**2 + (wall1.end.y - wall1.start.y)**2)
            len2 = math.sqrt((wall2.end.x - wall2.start.x)**2 + (wall2.end.y - wall2.start.y)**2)

            if midpoint_dist < distance_threshold and abs(len1 - len2) < distance_threshold * 2:
                # Keep the longer wall
                if len2 > len1:
                    is_duplicate = True
                    break
                else:
                    used.add(j)

        if not is_duplicate:
            unique_walls.append(wall1)
            used.add(i)

    return unique_walls


def estimate_wall_thickness(
    gray: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    scale: float
) -> float:
    """Estimate wall thickness from image intensity profile."""
    # Sample points along the wall
    num_samples = 10
    thicknesses = []

    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))

        # Get perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1:
            continue

        # Perpendicular unit vector
        nx = -dy / length
        ny = dx / length

        # Sample intensity along perpendicular
        wall_width = 0
        for d in range(-20, 21):
            sx = int(px + d * nx)
            sy = int(py + d * ny)

            if 0 <= sx < gray.shape[1] and 0 <= sy < gray.shape[0]:
                if gray[sy, sx] < 100:  # Dark pixel (wall)
                    wall_width += 1

        if wall_width > 0:
            thicknesses.append(wall_width * scale)

    if thicknesses:
        return np.median(thicknesses)

    return 0.2  # Default 20cm


def merge_collinear_walls(walls: List[WallSegment], tolerance: float = 5.0) -> List[WallSegment]:
    """Merge walls that are approximately collinear and close together."""
    if len(walls) < 2:
        return walls

    merged = []
    used = set()

    for i, wall1 in enumerate(walls):
        if i in used:
            continue

        # Start with current wall
        current = wall1
        used.add(i)

        for j, wall2 in enumerate(walls):
            if j in used or j <= i:
                continue

            # Check if walls are collinear
            if are_collinear(current, wall2, tolerance):
                # Merge walls
                current = merge_two_walls(current, wall2)
                used.add(j)

        merged.append(current)

    return merged


def are_collinear(wall1: WallSegment, wall2: WallSegment, tolerance: float) -> bool:
    """Check if two walls are approximately collinear."""
    # Get direction vectors
    d1x = wall1.end.x - wall1.start.x
    d1y = wall1.end.y - wall1.start.y
    d2x = wall2.end.x - wall2.start.x
    d2y = wall2.end.y - wall2.start.y

    # Normalize
    len1 = np.sqrt(d1x * d1x + d1y * d1y)
    len2 = np.sqrt(d2x * d2x + d2y * d2y)

    if len1 < 1 or len2 < 1:
        return False

    d1x, d1y = d1x / len1, d1y / len1
    d2x, d2y = d2x / len2, d2y / len2

    # Check angle (dot product)
    dot = abs(d1x * d2x + d1y * d2y)
    if dot < 0.95:  # Less than ~18 degrees apart
        return False

    # Check distance between closest endpoints
    endpoints1 = [(wall1.start.x, wall1.start.y), (wall1.end.x, wall1.end.y)]
    endpoints2 = [(wall2.start.x, wall2.start.y), (wall2.end.x, wall2.end.y)]

    min_dist = float('inf')
    for p1 in endpoints1:
        for p2 in endpoints2:
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            min_dist = min(min_dist, dist)

    return min_dist < tolerance * 5  # Within 5x tolerance pixels


def merge_two_walls(wall1: WallSegment, wall2: WallSegment) -> WallSegment:
    """Merge two collinear walls into one."""
    # Get all endpoints
    points = [
        (wall1.start.x, wall1.start.y),
        (wall1.end.x, wall1.end.y),
        (wall2.start.x, wall2.start.y),
        (wall2.end.x, wall2.end.y)
    ]

    # Find the two points that are farthest apart
    max_dist = 0
    best_pair = (points[0], points[1])

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.sqrt(
                (points[i][0] - points[j][0]) ** 2 +
                (points[i][1] - points[j][1]) ** 2
            )
            if dist > max_dist:
                max_dist = dist
                best_pair = (points[i], points[j])

    return WallSegment(
        start=Point(x=best_pair[0][0], y=best_pair[0][1]),
        end=Point(x=best_pair[1][0], y=best_pair[1][1]),
        thickness=(wall1.thickness + wall2.thickness) / 2,
        material=wall1.material,
        attenuation_db=wall1.attenuation_db
    )


def segment_rooms(
    gray: np.ndarray,
    walls: List[WallSegment],
    scale: float
) -> List[Room]:
    """
    Identify distinct rooms using flood fill.

    Rooms are enclosed spaces between walls.
    """
    height, width = gray.shape

    # Create binary mask of walls
    wall_mask = np.ones((height, width), dtype=np.uint8) * 255

    for wall in walls:
        x1, y1 = int(wall.start.x), int(wall.start.y)
        x2, y2 = int(wall.end.x), int(wall.end.y)
        thickness = max(3, int(wall.thickness / scale)) if scale > 0 else 3
        cv2.line(wall_mask, (x1, y1), (x2, y2), 0, thickness=thickness)

    # Find connected components (rooms)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        wall_mask, connectivity=8
    )

    rooms = []
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get room area in pixels
        area_pixels = stats[i, cv2.CC_STAT_AREA]

        # Convert to square meters
        area_sqm = area_pixels * (scale ** 2) if scale > 0 else area_pixels

        # Filter out tiny rooms (< 1 sqm) and very large (probably background)
        if area_sqm < 1.0 or area_sqm > 1000:
            continue

        # Extract room polygon
        room_pixels = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            room_pixels,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)

            # Simplify polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert to list of points
            polygon = [[float(p[0][0]), float(p[0][1])] for p in approx]

            if len(polygon) >= 3:  # Valid polygon
                room = Room(
                    name=f"Room {len(rooms) + 1}",
                    area=area_sqm,
                    polygon=polygon
                )
                rooms.append(room)

    return rooms


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image dimensions without loading full image into memory."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height
