"""Improved wall detection for floor plans.

This version focuses on detecting actual structural walls by:
1. Only detecting horizontal and vertical lines (orthogonal walls)
2. Filtering out text, annotations, and furniture
3. Using adaptive thresholds based on image characteristics
4. Better wall merging and deduplication
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from app.schemas.project import WallSegment, Point

logger = logging.getLogger(__name__)

# Material attenuation database (IEEE 802.11 standards)
MATERIAL_ATTENUATION = {
    'concrete': 15.0,
    'brick': 12.0,
    'wood': 6.0,
    'glass': 5.0,
    'drywall': 3.0,
    'metal': 25.0,
    'unknown': 10.0
}


@dataclass
class DetectedLine:
    """Intermediate representation of a detected line."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float  # degrees from horizontal
    length: float
    thickness: float
    confidence: float


class ImprovedWallDetector:
    """
    Improved wall detection focused on accuracy over quantity.

    Key improvements:
    - Only detects orthogonal (horizontal/vertical) walls
    - Filters text and annotation regions
    - Adaptive thresholds based on image
    - Better wall merging
    """

    def __init__(self, scale: float = 0.05):
        self.scale = scale
        self.angle_tolerance = 5.0  # degrees from H/V to consider orthogonal
        self.min_wall_length_pixels = 50  # minimum wall length in pixels
        self.min_wall_thickness_pixels = 2  # minimum wall thickness
        self.max_wall_thickness_pixels = 30  # maximum wall thickness (filter furniture)

    def detect_walls(self, image: np.ndarray) -> List[WallSegment]:
        """
        Detect walls with improved accuracy.

        Args:
            image: Input BGR or grayscale image

        Returns:
            List of detected wall segments
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape

        # Adjust parameters based on image size
        self._adjust_params_for_image_size(width, height)

        # Step 1: Preprocess - denoise and enhance
        processed = self._preprocess(gray)

        # Step 2: Create wall mask (binary image of potential walls)
        wall_mask = self._create_wall_mask(processed)

        # Step 3: Detect lines from wall mask
        lines = self._detect_lines(wall_mask)

        # Step 4: Filter to only orthogonal lines
        orthogonal_lines = self._filter_orthogonal(lines)

        # Step 5: Filter by thickness (remove text, thin lines)
        thickness_filtered = self._filter_by_thickness(orthogonal_lines, gray)

        # Step 6: Merge nearby parallel walls
        merged = self._merge_walls(thickness_filtered)

        # Step 7: Convert to WallSegment objects
        walls = self._to_wall_segments(merged)

        logger.info(f"Detected {len(walls)} walls (from {len(lines)} initial lines)")

        return walls

    def _adjust_params_for_image_size(self, width: int, height: int):
        """Adjust detection parameters based on image size."""
        # Larger images need longer minimum walls
        min_dimension = min(width, height)

        if min_dimension > 2000:
            self.min_wall_length_pixels = 80
            self.max_wall_thickness_pixels = 40
        elif min_dimension > 1000:
            self.min_wall_length_pixels = 50
            self.max_wall_thickness_pixels = 30
        else:
            self.min_wall_length_pixels = 30
            self.max_wall_thickness_pixels = 20

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for wall detection."""
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def _create_wall_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create binary mask of potential walls."""
        # Use Otsu's thresholding to find optimal threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        # Remove small noise
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

        # Close small gaps in walls
        kernel_close = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        return closed

    def _detect_lines(self, binary: np.ndarray) -> List[DetectedLine]:
        """Detect lines using Hough transform."""
        # Edge detection
        edges = cv2.Canny(binary, 50, 150)

        # Detect lines with stricter parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,  # Higher threshold = fewer false positives
            minLineLength=self.min_wall_length_pixels,
            maxLineGap=10
        )

        if lines is None:
            return []

        detected = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(dy, dx))

            # Normalize angle to [0, 180)
            if angle < 0:
                angle += 180

            length = math.sqrt(dx*dx + dy*dy)

            detected.append(DetectedLine(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                angle=angle,
                length=length,
                thickness=0.0,  # Will be estimated later
                confidence=1.0
            ))

        return detected

    def _filter_orthogonal(self, lines: List[DetectedLine]) -> List[DetectedLine]:
        """Filter to only horizontal and vertical lines."""
        orthogonal = []

        for line in lines:
            # Check if horizontal (angle near 0 or 180)
            is_horizontal = (
                line.angle < self.angle_tolerance or
                line.angle > (180 - self.angle_tolerance)
            )

            # Check if vertical (angle near 90)
            is_vertical = abs(line.angle - 90) < self.angle_tolerance

            if is_horizontal or is_vertical:
                # Snap to exact H/V
                if is_horizontal:
                    # Make perfectly horizontal
                    avg_y = (line.y1 + line.y2) / 2
                    line.y1 = avg_y
                    line.y2 = avg_y
                    line.angle = 0
                else:
                    # Make perfectly vertical
                    avg_x = (line.x1 + line.x2) / 2
                    line.x1 = avg_x
                    line.x2 = avg_x
                    line.angle = 90

                orthogonal.append(line)

        logger.info(f"Filtered to {len(orthogonal)} orthogonal lines from {len(lines)}")
        return orthogonal

    def _filter_by_thickness(
        self,
        lines: List[DetectedLine],
        gray: np.ndarray
    ) -> List[DetectedLine]:
        """Filter lines by their thickness in the image."""
        filtered = []

        for line in lines:
            thickness = self._estimate_line_thickness(gray, line)
            line.thickness = thickness

            # Filter: walls should have reasonable thickness
            if self.min_wall_thickness_pixels <= thickness <= self.max_wall_thickness_pixels:
                filtered.append(line)

        logger.info(f"Filtered to {len(filtered)} lines by thickness from {len(lines)}")
        return filtered

    def _estimate_line_thickness(self, gray: np.ndarray, line: DetectedLine) -> float:
        """Estimate the thickness of a line by sampling perpendicular to it."""
        # Sample at multiple points along the line
        num_samples = min(10, int(line.length / 20) + 1)
        thicknesses = []

        for i in range(num_samples):
            t = (i + 0.5) / num_samples
            px = int(line.x1 + t * (line.x2 - line.x1))
            py = int(line.y1 + t * (line.y2 - line.y1))

            # Sample perpendicular
            if line.angle < 45 or line.angle > 135:
                # Mostly horizontal - sample vertically
                thickness = self._measure_thickness_vertical(gray, px, py)
            else:
                # Mostly vertical - sample horizontally
                thickness = self._measure_thickness_horizontal(gray, px, py)

            if thickness > 0:
                thicknesses.append(thickness)

        return np.median(thicknesses) if thicknesses else 0.0

    def _measure_thickness_vertical(self, gray: np.ndarray, x: int, y: int) -> float:
        """Measure wall thickness vertically at a point."""
        height = gray.shape[0]
        threshold = 128  # Dark pixels are walls

        # Find top edge
        top = y
        while top > 0 and gray[top, x] < threshold:
            top -= 1

        # Find bottom edge
        bottom = y
        while bottom < height - 1 and gray[bottom, x] < threshold:
            bottom += 1

        return float(bottom - top)

    def _measure_thickness_horizontal(self, gray: np.ndarray, x: int, y: int) -> float:
        """Measure wall thickness horizontally at a point."""
        width = gray.shape[1]
        threshold = 128

        # Find left edge
        left = x
        while left > 0 and gray[y, left] < threshold:
            left -= 1

        # Find right edge
        right = x
        while right < width - 1 and gray[y, right] < threshold:
            right += 1

        return float(right - left)

    def _merge_walls(self, lines: List[DetectedLine]) -> List[DetectedLine]:
        """Merge nearby parallel walls."""
        if len(lines) < 2:
            return lines

        # Separate horizontal and vertical
        horizontal = [l for l in lines if l.angle < 45 or l.angle > 135]
        vertical = [l for l in lines if 45 <= l.angle <= 135]

        merged_h = self._merge_parallel_lines(horizontal, is_horizontal=True)
        merged_v = self._merge_parallel_lines(vertical, is_horizontal=False)

        return merged_h + merged_v

    def _merge_parallel_lines(
        self,
        lines: List[DetectedLine],
        is_horizontal: bool
    ) -> List[DetectedLine]:
        """Merge parallel lines that are close together."""
        if len(lines) < 2:
            return lines

        # Sort by position
        if is_horizontal:
            lines = sorted(lines, key=lambda l: (l.y1 + l.y2) / 2)
            pos_key = lambda l: (l.y1 + l.y2) / 2
            range_key = lambda l: (min(l.x1, l.x2), max(l.x1, l.x2))
        else:
            lines = sorted(lines, key=lambda l: (l.x1 + l.x2) / 2)
            pos_key = lambda l: (l.x1 + l.x2) / 2
            range_key = lambda l: (min(l.y1, l.y2), max(l.y1, l.y2))

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            # Find lines to merge with
            group = [line1]
            used.add(i)

            pos1 = pos_key(line1)
            range1 = range_key(line1)

            for j, line2 in enumerate(lines):
                if j in used:
                    continue

                pos2 = pos_key(line2)
                range2 = range_key(line2)

                # Check if close enough to merge
                pos_diff = abs(pos2 - pos1)

                # Check if ranges overlap
                overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])

                if pos_diff < 15 and overlap > -50:  # Allow small gaps
                    group.append(line2)
                    used.add(j)

            # Merge the group
            if len(group) > 1:
                merged.append(self._merge_line_group(group, is_horizontal))
            else:
                merged.append(line1)

        return merged

    def _merge_line_group(
        self,
        lines: List[DetectedLine],
        is_horizontal: bool
    ) -> DetectedLine:
        """Merge a group of parallel lines into one."""
        if is_horizontal:
            avg_y = sum((l.y1 + l.y2) / 2 for l in lines) / len(lines)
            min_x = min(min(l.x1, l.x2) for l in lines)
            max_x = max(max(l.x1, l.x2) for l in lines)

            return DetectedLine(
                x1=min_x, y1=avg_y,
                x2=max_x, y2=avg_y,
                angle=0,
                length=max_x - min_x,
                thickness=sum(l.thickness for l in lines) / len(lines),
                confidence=max(l.confidence for l in lines)
            )
        else:
            avg_x = sum((l.x1 + l.x2) / 2 for l in lines) / len(lines)
            min_y = min(min(l.y1, l.y2) for l in lines)
            max_y = max(max(l.y1, l.y2) for l in lines)

            return DetectedLine(
                x1=avg_x, y1=min_y,
                x2=avg_x, y2=max_y,
                angle=90,
                length=max_y - min_y,
                thickness=sum(l.thickness for l in lines) / len(lines),
                confidence=max(l.confidence for l in lines)
            )

    def _to_wall_segments(self, lines: List[DetectedLine]) -> List[WallSegment]:
        """Convert detected lines to WallSegment objects."""
        walls = []

        for line in lines:
            # Convert thickness from pixels to meters
            thickness_m = line.thickness * self.scale

            # Clamp to reasonable wall thickness
            thickness_m = max(0.1, min(0.5, thickness_m))

            walls.append(WallSegment(
                start=Point(x=line.x1, y=line.y1),
                end=Point(x=line.x2, y=line.y2),
                thickness=thickness_m,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            ))

        return walls

    def detect_openings(self, image: np.ndarray, walls: List[WallSegment]) -> List:
        """
        Detect door and window openings.

        Note: This is a placeholder - opening detection is done by AI detector
        when available. This returns an empty list for compatibility.
        """
        return []
