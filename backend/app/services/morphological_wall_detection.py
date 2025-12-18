"""Morphological Wall Detection - Pure CV approach for CAD-quality output.

This approach isolates walls based on their visual properties:
1. Walls are thick black/dark lines (typically 3-15 pixels wide)
2. Text and dimensions are thin (1-2 pixels)
3. Furniture has different patterns (not solid lines)

Process:
1. Threshold to binary
2. Morphological operations to isolate thick lines
3. Connected component filtering (keep elongated shapes)
4. Skeletonization for centerlines
5. RANSAC/Douglas-Peucker for robust line fitting
6. Graph-based cleanup for proper connections
"""

import logging
import math
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import cv2

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

# Material attenuation
MATERIAL_ATTENUATION = {
    'concrete': 15.0,
    'brick': 12.0,
    'wood': 6.0,
    'drywall': 3.0,
    'unknown': 10.0
}


class MorphologicalWallDetector:
    """
    Pure CV wall detector using morphological operations.

    Isolates walls by their thickness and shape properties,
    then vectorizes using robust line fitting.
    """

    def __init__(self, min_wall_thickness: int = 3, max_wall_thickness: int = 20):
        """
        Initialize detector.

        Args:
            min_wall_thickness: Minimum wall thickness in pixels
            max_wall_thickness: Maximum wall thickness in pixels
        """
        self.min_wall_thickness = min_wall_thickness
        self.max_wall_thickness = max_wall_thickness
        self.scale = 0.05

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls using morphological approach.

        Args:
            image: Input BGR or grayscale image
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale
        height, width = image.shape[:2]

        logger.info(f"Starting morphological wall detection on {width}x{height} image")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: Adaptive thresholding for varying lighting
        logger.info("Step 1: Thresholding...")
        binary = self._adaptive_threshold(gray)

        # Step 2: Isolate thick lines (walls) using morphological operations
        logger.info("Step 2: Isolating walls by thickness...")
        wall_mask = self._isolate_thick_lines(binary)

        # Step 3: Filter by shape (keep elongated structures)
        logger.info("Step 3: Filtering by shape...")
        wall_mask = self._filter_by_shape(wall_mask)

        # Step 4: Skeletonize to get centerlines
        logger.info("Step 4: Skeletonizing...")
        skeleton = self._skeletonize(wall_mask)

        # Step 5: Extract line segments using probabilistic Hough
        logger.info("Step 5: Extracting lines...")
        raw_lines = self._extract_lines_hough(skeleton, wall_mask)
        logger.info(f"  Raw lines from Hough: {len(raw_lines)}")

        # Step 6: Fit lines with RANSAC for robustness
        logger.info("Step 6: RANSAC line fitting...")
        fitted_lines = self._ransac_fit_lines(skeleton, raw_lines)
        logger.info(f"  After RANSAC: {len(fitted_lines)}")

        # Step 7: Snap to grid and align H/V
        logger.info("Step 7: Grid snapping and alignment...")
        aligned_lines = self._align_and_snap(fitted_lines)

        # Step 8: Merge collinear and connect endpoints
        logger.info("Step 8: Merging and connecting...")
        final_lines = self._merge_and_connect(aligned_lines)
        logger.info(f"  Final walls: {len(final_lines)}")

        # Convert to WallSegments
        walls = []
        for x1, y1, x2, y2 in final_lines:
            wall = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            walls.append(wall)

        logger.info(f"Morphological detection complete: {len(walls)} walls")
        return walls, []

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to handle varying lighting."""
        # Denoise first
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Check if image is light or dark background
        mean_val = np.mean(denoised)

        if mean_val > 127:
            # Light background (typical floor plan) - invert so walls are white
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Dark background
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _isolate_thick_lines(self, binary: np.ndarray) -> np.ndarray:
        """
        Isolate thick lines (walls) from thin lines (text, dimensions).

        Uses morphological opening with different kernel sizes to separate
        thick structures from thin ones.
        """
        # Opening removes structures smaller than the kernel
        # We want to KEEP thick lines and REMOVE thin ones

        # First, erode to remove thin lines (text, dimensions)
        thin_kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(binary, thin_kernel, iterations=1)

        # Dilate back to restore thick lines
        dilated = cv2.dilate(eroded, thin_kernel, iterations=1)

        # Additional pass: opening to clean up
        open_kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, open_kernel)

        # Close small gaps in walls
        close_kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

        return closed

    def _filter_by_shape(self, binary: np.ndarray) -> np.ndarray:
        """
        Filter components by shape - keep elongated structures (walls).

        Walls have high aspect ratio (length >> width).
        Furniture and other objects tend to be more square/circular.
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Create output mask
        result = np.zeros_like(binary)

        height, width = binary.shape
        min_area = (height * width) * 0.0001  # At least 0.01% of image
        max_area = (height * width) * 0.5     # At most 50% of image

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Calculate aspect ratio (elongation)
            aspect_ratio = max(w, h) / (min(w, h) + 1)

            # Calculate solidity (area / convex hull area)
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / (hull_area + 1)

                # Walls are elongated (high aspect ratio) or have low solidity (L-shapes, etc.)
                # Keep if: elongated OR solid but not too square
                if aspect_ratio > 2.0 or (solidity > 0.3 and aspect_ratio > 1.5):
                    result[labels == i] = 255

        return result

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Skeletonize to get 1-pixel centerlines."""
        if hasattr(cv2, 'ximgproc'):
            return cv2.ximgproc.thinning(binary)
        else:
            # Fallback: morphological skeleton
            skeleton = np.zeros_like(binary)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            temp = binary.copy()
            while True:
                eroded = cv2.erode(temp, element)
                opened = cv2.dilate(eroded, element)
                temp_skel = cv2.subtract(temp, opened)
                skeleton = cv2.bitwise_or(skeleton, temp_skel)
                temp = eroded.copy()

                if cv2.countNonZero(temp) == 0:
                    break

            return skeleton

    def _extract_lines_hough(
        self,
        skeleton: np.ndarray,
        wall_mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Extract line segments using probabilistic Hough transform."""
        height, width = skeleton.shape

        # Adaptive parameters based on image size
        min_length = max(15, min(width, height) // 50)
        max_gap = max(5, min_length // 3)

        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=min_length,
            maxLineGap=max_gap
        )

        if lines is None:
            return []

        result = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            result.append((int(x1), int(y1), int(x2), int(y2)))

        return result

    def _ransac_fit_lines(
        self,
        skeleton: np.ndarray,
        initial_lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Refine lines using RANSAC for robustness.

        For each initial line, collect nearby skeleton points and
        fit a more accurate line through them.
        """
        if not initial_lines:
            return []

        # Get all skeleton points
        skeleton_points = np.column_stack(np.where(skeleton > 0))  # (y, x) format

        if len(skeleton_points) < 10:
            return initial_lines

        refined = []

        for x1, y1, x2, y2 in initial_lines:
            # Find skeleton points near this line
            nearby_points = self._points_near_line(
                skeleton_points, x1, y1, x2, y2, threshold=10
            )

            if len(nearby_points) < 5:
                refined.append((x1, y1, x2, y2))
                continue

            # Fit line to nearby points using least squares
            try:
                # Points are in (y, x) format, convert to (x, y)
                xs = nearby_points[:, 1].astype(float)
                ys = nearby_points[:, 0].astype(float)

                # Check if mostly vertical or horizontal
                x_range = np.max(xs) - np.min(xs)
                y_range = np.max(ys) - np.min(ys)

                if x_range > y_range:
                    # Fit y = mx + b
                    A = np.vstack([xs, np.ones(len(xs))]).T
                    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]

                    new_x1, new_x2 = int(np.min(xs)), int(np.max(xs))
                    new_y1 = int(m * new_x1 + b)
                    new_y2 = int(m * new_x2 + b)
                else:
                    # Fit x = my + b
                    A = np.vstack([ys, np.ones(len(ys))]).T
                    m, b = np.linalg.lstsq(A, xs, rcond=None)[0]

                    new_y1, new_y2 = int(np.min(ys)), int(np.max(ys))
                    new_x1 = int(m * new_y1 + b)
                    new_x2 = int(m * new_y2 + b)

                refined.append((new_x1, new_y1, new_x2, new_y2))

            except Exception:
                refined.append((x1, y1, x2, y2))

        return refined

    def _points_near_line(
        self,
        points: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int,
        threshold: float
    ) -> np.ndarray:
        """Find points within threshold distance of a line segment."""
        # Line direction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)

        if length < 1:
            return np.array([])

        # Normalize
        dx, dy = dx / length, dy / length

        # Points are (y, x) format
        px = points[:, 1] - x1
        py = points[:, 0] - y1

        # Project onto line
        proj = px * dx + py * dy

        # Perpendicular distance
        perp_x = px - proj * dx
        perp_y = py - proj * dy
        dist = np.sqrt(perp_x**2 + perp_y**2)

        # Keep points within threshold and within line segment bounds
        mask = (dist < threshold) & (proj >= -threshold) & (proj <= length + threshold)

        return points[mask]

    def _align_and_snap(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Snap to grid and align nearly H/V lines."""
        grid = 5
        angle_threshold = 5.0  # degrees

        result = []
        for x1, y1, x2, y2 in lines:
            # Snap to grid
            x1 = round(x1 / grid) * grid
            y1 = round(y1 / grid) * grid
            x2 = round(x2 / grid) * grid
            y2 = round(y2 / grid) * grid

            # Align H/V
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx == 0 and dy == 0:
                continue

            angle = math.degrees(math.atan2(dy, dx))

            # Nearly horizontal
            if angle < angle_threshold:
                avg_y = round((y1 + y2) / 2 / grid) * grid
                y1 = y2 = avg_y
            # Nearly vertical
            elif angle > (90 - angle_threshold):
                avg_x = round((x1 + x2) / 2 / grid) * grid
                x1 = x2 = avg_x

            # Skip degenerate lines
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length >= 10:
                result.append((x1, y1, x2, y2))

        return result

    def _merge_and_connect(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge collinear lines and connect nearby endpoints."""
        if len(lines) < 2:
            return lines

        # Step 1: Merge collinear lines
        merged = self._merge_collinear(lines)

        # Step 2: Connect nearby endpoints
        connected = self._connect_endpoints(merged)

        # Step 3: Remove duplicates
        final = self._remove_duplicates(connected)

        return final

    def _merge_collinear(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge collinear line segments."""
        if len(lines) < 2:
            return lines

        used = set()
        result = []

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            x1, y1, x2, y2 = line1
            used.add(i)

            # Find collinear lines to merge
            points = [(x1, y1), (x2, y2)]

            for j, line2 in enumerate(lines):
                if j in used:
                    continue

                if self._are_collinear(line1, line2):
                    x3, y3, x4, y4 = line2
                    points.extend([(x3, y3), (x4, y4)])
                    used.add(j)

            # Find extremal points
            if len(points) > 2:
                # For H/V lines, sort by one coordinate
                if abs(y2 - y1) < abs(x2 - x1):  # Horizontal
                    points.sort(key=lambda p: p[0])
                else:  # Vertical
                    points.sort(key=lambda p: p[1])

                result.append((points[0][0], points[0][1], points[-1][0], points[-1][1]))
            else:
                result.append(line1)

        return result

    def _are_collinear(
        self,
        line1: Tuple[int, int, int, int],
        line2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two lines are collinear and close."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Direction vectors
        d1x, d1y = x2 - x1, y2 - y1
        d2x, d2y = x4 - x3, y4 - y3

        len1 = math.sqrt(d1x**2 + d1y**2)
        len2 = math.sqrt(d2x**2 + d2y**2)

        if len1 < 1 or len2 < 1:
            return False

        # Normalize
        d1x, d1y = d1x / len1, d1y / len1
        d2x, d2y = d2x / len2, d2y / len2

        # Check angle
        dot = abs(d1x * d2x + d1y * d2y)
        if dot < 0.98:  # Within ~11 degrees
            return False

        # Check distance
        endpoints1 = [(x1, y1), (x2, y2)]
        endpoints2 = [(x3, y3), (x4, y4)]

        min_dist = float('inf')
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_dist = min(min_dist, dist)

        return min_dist < 20

    def _connect_endpoints(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Snap nearby endpoints together."""
        if len(lines) < 2:
            return lines

        # Collect all endpoints
        endpoints = []
        for i, (x1, y1, x2, y2) in enumerate(lines):
            endpoints.append((i, 0, x1, y1))  # 0 = start
            endpoints.append((i, 1, x2, y2))  # 1 = end

        # Find clusters
        threshold = 15
        clusters = []
        used = set()

        for i, (idx1, end1, x1, y1) in enumerate(endpoints):
            if i in used:
                continue

            cluster = [(idx1, end1, x1, y1)]
            used.add(i)

            for j, (idx2, end2, x2, y2) in enumerate(endpoints):
                if j in used:
                    continue

                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < threshold:
                    cluster.append((idx2, end2, x2, y2))
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Snap clusters to centroid
        lines = list(lines)
        grid = 5

        for cluster in clusters:
            cx = round(sum(p[2] for p in cluster) / len(cluster) / grid) * grid
            cy = round(sum(p[3] for p in cluster) / len(cluster) / grid) * grid

            for idx, end, _, _ in cluster:
                x1, y1, x2, y2 = lines[idx]
                if end == 0:
                    lines[idx] = (cx, cy, x2, y2)
                else:
                    lines[idx] = (x1, y1, cx, cy)

        return lines

    def _remove_duplicates(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Remove duplicate lines."""
        unique = []

        for line in lines:
            x1, y1, x2, y2 = line

            # Skip zero-length
            if x1 == x2 and y1 == y2:
                continue

            # Normalize direction
            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Check for duplicates
            is_dup = False
            for ux1, uy1, ux2, uy2 in unique:
                d1 = math.sqrt((x1 - ux1)**2 + (y1 - uy1)**2)
                d2 = math.sqrt((x2 - ux2)**2 + (y2 - uy2)**2)

                if d1 < 10 and d2 < 10:
                    is_dup = True
                    break

            if not is_dup:
                unique.append((x1, y1, x2, y2))

        return unique

    def detect_walls_from_file(
        self,
        file_path: str,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """Detect walls from an image file."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return self.detect_walls(image, scale)


def detect_walls_morphological(
    image_path: str,
    scale: float = 0.05
) -> Tuple[List[WallSegment], List[Room]]:
    """Convenience function for morphological wall detection."""
    detector = MorphologicalWallDetector()
    return detector.detect_walls_from_file(image_path, scale)
