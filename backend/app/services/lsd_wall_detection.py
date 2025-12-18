"""LSD-based Wall Detection - Robust line detection without parameter tuning.

Uses Line Segment Detector (LSD) which is more robust than Hough Transform.
Also uses contour-based extraction to trace actual wall boundaries.

Key advantages:
1. LSD is parameter-free and more accurate than Hough
2. Contour tracing follows actual wall edges
3. Parallel line grouping finds wall centerlines from edges
4. Works on various floor plan styles
"""

import logging
import math
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

MATERIAL_ATTENUATION = {'concrete': 15.0}


class LSDWallDetector:
    """
    Wall detector using Line Segment Detector (LSD) algorithm.

    LSD is a linear-time line segment detector that gives accurate results
    without parameter tuning, unlike Hough Transform.
    """

    def __init__(self):
        self.scale = 0.05
        # Create LSD detector
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls using LSD + contour-based approach.
        """
        self.scale = scale
        height, width = image.shape[:2]

        logger.info(f"LSD wall detection on {width}x{height} image")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Method 1: LSD on edge-enhanced image
        logger.info("Running LSD detection...")
        lsd_lines = self._detect_with_lsd(gray)
        logger.info(f"  LSD raw lines: {len(lsd_lines)}")

        # Method 2: Contour-based extraction
        logger.info("Running contour extraction...")
        contour_lines = self._detect_from_contours(gray)
        logger.info(f"  Contour lines: {len(contour_lines)}")

        # Combine and process
        all_lines = lsd_lines + contour_lines
        logger.info(f"  Combined: {len(all_lines)} lines")

        # Filter by length
        min_length = max(15, min(width, height) // 40)
        filtered = [l for l in all_lines if self._line_length(l) >= min_length]
        logger.info(f"  After length filter: {len(filtered)}")

        # Group parallel lines (wall edges → centerlines)
        logger.info("Grouping parallel lines...")
        centerlines = self._group_parallel_lines(filtered)
        logger.info(f"  Centerlines: {len(centerlines)}")

        # Snap and align
        aligned = self._snap_and_align(centerlines)

        # Merge collinear
        merged = self._merge_collinear(aligned)
        logger.info(f"  After merge: {len(merged)}")

        # Connect endpoints
        connected = self._connect_endpoints(merged)

        # Remove duplicates
        final = self._remove_duplicates(connected)
        logger.info(f"  Final: {len(final)}")

        # Convert to WallSegments
        walls = []
        for x1, y1, x2, y2 in final:
            wall = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            walls.append(wall)

        logger.info(f"LSD detection complete: {len(walls)} walls")
        return walls, []

    def _detect_with_lsd(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect lines using LSD algorithm."""
        # Enhance edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Run LSD
        lines, widths, prec, nfa = self.lsd.detect(blurred)

        if lines is None:
            return []

        result = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # Filter by width (walls are thicker)
            if widths is not None and i < len(widths):
                width = widths[i][0]
                if width < 1.5:  # Too thin, probably not a wall
                    continue
            result.append((int(x1), int(y1), int(x2), int(y2)))

        return result

    def _detect_from_contours(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect wall lines from contours."""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            # Approximate contour with polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Extract line segments from polygon
            for i in range(len(approx)):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % len(approx)][0]

                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])

                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 10:
                    lines.append((x1, y1, x2, y2))

        return lines

    def _line_length(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate line length."""
        x1, y1, x2, y2 = line
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _group_parallel_lines(
        self,
        lines: List[Tuple[int, int, int, int]],
        distance_threshold: float = 15,
        angle_threshold: float = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Group parallel lines that are close together.

        Walls have two edges - this finds the centerline between them.
        """
        if len(lines) < 2:
            return lines

        # Calculate angle for each line
        line_angles = []
        for x1, y1, x2, y2 in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            line_angles.append(angle)

        # Group by angle
        groups = []
        used = set()

        for i, (line1, angle1) in enumerate(zip(lines, line_angles)):
            if i in used:
                continue

            group = [line1]
            used.add(i)

            for j, (line2, angle2) in enumerate(zip(lines, line_angles)):
                if j in used:
                    continue

                # Check angle similarity
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff > angle_threshold:
                    continue

                # Check distance between lines
                dist = self._line_to_line_distance(line1, line2)
                if dist < distance_threshold:
                    group.append(line2)
                    used.add(j)

            groups.append(group)

        # For each group, compute centerline
        centerlines = []
        for group in groups:
            if len(group) == 1:
                centerlines.append(group[0])
            else:
                centerline = self._compute_centerline(group)
                if centerline:
                    centerlines.append(centerline)

        return centerlines

    def _line_to_line_distance(
        self,
        line1: Tuple[int, int, int, int],
        line2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate approximate distance between two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Use midpoints
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)

        # Perpendicular distance from mid2 to line1
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)

        if length < 1:
            return math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)

        # Normalize
        dx, dy = dx / length, dy / length

        # Vector from line1 start to mid2
        vx = mid2[0] - x1
        vy = mid2[1] - y1

        # Perpendicular distance
        perp_dist = abs(vx * (-dy) + vy * dx)

        return perp_dist

    def _compute_centerline(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute centerline of a group of parallel lines."""
        if not lines:
            return None

        # Collect all endpoints
        all_points = []
        for x1, y1, x2, y2 in lines:
            all_points.extend([(x1, y1), (x2, y2)])

        if len(all_points) < 2:
            return None

        # Fit a line through all points
        points = np.array(all_points, dtype=np.float32)

        # Use PCA to find principal direction
        mean = np.mean(points, axis=0)
        centered = points - mean

        # SVD for principal component
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]

        # Project points onto this direction
        projections = np.dot(centered, direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # Compute endpoints
        p1 = mean + min_proj * direction
        p2 = mean + max_proj * direction

        return (int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    def _snap_and_align(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Snap to grid and align H/V lines."""
        grid = 5
        angle_threshold = 5.0

        result = []
        for x1, y1, x2, y2 in lines:
            # Snap to grid
            x1 = round(x1 / grid) * grid
            y1 = round(y1 / grid) * grid
            x2 = round(x2 / grid) * grid
            y2 = round(y2 / grid) * grid

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx == 0 and dy == 0:
                continue

            angle = math.degrees(math.atan2(dy, dx))

            # Snap nearly horizontal
            if angle < angle_threshold:
                avg_y = round((y1 + y2) / 2 / grid) * grid
                y1 = y2 = avg_y
            # Snap nearly vertical
            elif angle > (90 - angle_threshold):
                avg_x = round((x1 + x2) / 2 / grid) * grid
                x1 = x2 = avg_x

            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length >= 10:
                result.append((x1, y1, x2, y2))

        return result

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
            points = [(x1, y1), (x2, y2)]

            for j, line2 in enumerate(lines):
                if j in used:
                    continue

                if self._are_collinear(line1, line2):
                    x3, y3, x4, y4 = line2
                    points.extend([(x3, y3), (x4, y4)])
                    used.add(j)

            # Find extremes
            if abs(y2 - y1) < abs(x2 - x1):  # More horizontal
                points.sort(key=lambda p: p[0])
            else:  # More vertical
                points.sort(key=lambda p: p[1])

            result.append((points[0][0], points[0][1], points[-1][0], points[-1][1]))

        return result

    def _are_collinear(
        self,
        line1: Tuple[int, int, int, int],
        line2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two lines are collinear."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Direction vectors
        d1x, d1y = x2 - x1, y2 - y1
        d2x, d2y = x4 - x3, y4 - y3

        len1 = math.sqrt(d1x**2 + d1y**2)
        len2 = math.sqrt(d2x**2 + d2y**2)

        if len1 < 1 or len2 < 1:
            return False

        d1x, d1y = d1x / len1, d1y / len1
        d2x, d2y = d2x / len2, d2y / len2

        # Check angle
        dot = abs(d1x * d2x + d1y * d2y)
        if dot < 0.98:
            return False

        # Check if endpoints are close
        endpoints = [(x1, y1), (x2, y2)]
        for ex, ey in endpoints:
            for px, py in [(x3, y3), (x4, y4)]:
                if math.sqrt((ex - px)**2 + (ey - py)**2) < 25:
                    return True

        return False

    def _connect_endpoints(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Connect nearby endpoints."""
        if len(lines) < 2:
            return lines

        threshold = 20
        grid = 5

        endpoints = []
        for i, (x1, y1, x2, y2) in enumerate(lines):
            endpoints.append((i, 0, x1, y1))
            endpoints.append((i, 1, x2, y2))

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

                if math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold:
                    cluster.append((idx2, end2, x2, y2))
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        lines = list(lines)
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

        for x1, y1, x2, y2 in lines:
            if x1 == x2 and y1 == y2:
                continue

            # Normalize
            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            is_dup = False
            for ux1, uy1, ux2, uy2 in unique:
                d1 = math.sqrt((x1 - ux1)**2 + (y1 - uy1)**2)
                d2 = math.sqrt((x2 - ux2)**2 + (y2 - uy2)**2)

                if d1 < 15 and d2 < 15:
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
        """Detect walls from file."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return self.detect_walls(image, scale)
