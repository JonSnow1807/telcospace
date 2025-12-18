"""Boundary-First Wall Detection.

This approach mimics how floor plans are actually drawn:
1. Find the outer boundary (perimeter walls)
2. Find interior walls that connect to the boundary
3. Filter out furniture, text, and other non-wall elements

Key insight: Real walls form a connected network with the building perimeter.
Furniture and text are isolated or don't connect to walls.
"""

import logging
import math
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import cv2

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

MATERIAL_ATTENUATION = {'concrete': 15.0}


class BoundaryWallDetector:
    """
    Boundary-first wall detector.

    Strategy:
    1. Find outer boundary of the floor plan
    2. Identify interior walls connected to boundary
    3. Build a connected wall network
    """

    def __init__(self):
        self.scale = 0.05

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """Detect walls using boundary-first approach."""
        self.scale = scale
        height, width = image.shape[:2]

        logger.info(f"Boundary wall detection on {width}x{height} image")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: Find the floor plan boundary
        logger.info("Step 1: Finding boundary...")
        boundary_mask, boundary_contour = self._find_boundary(gray)

        if boundary_contour is None:
            logger.warning("Could not find boundary, using full image")
            boundary_mask = np.ones_like(gray) * 255

        # Step 2: Extract walls within boundary
        logger.info("Step 2: Extracting walls...")
        wall_mask = self._extract_wall_mask(gray, boundary_mask)

        # Step 3: Vectorize walls
        logger.info("Step 3: Vectorizing...")
        raw_walls = self._vectorize_walls(wall_mask)
        logger.info(f"  Raw walls: {len(raw_walls)}")

        # Step 4: Filter to connected wall network
        logger.info("Step 4: Building wall network...")
        if boundary_contour is not None:
            boundary_walls = self._extract_boundary_walls(boundary_contour)
            connected_walls = self._build_wall_network(raw_walls, boundary_walls)
        else:
            connected_walls = raw_walls

        logger.info(f"  Connected walls: {len(connected_walls)}")

        # Step 5: Clean up
        logger.info("Step 5: Cleanup...")
        final_walls = self._cleanup_walls(connected_walls)
        logger.info(f"  Final walls: {len(final_walls)}")

        # Convert to WallSegments
        walls = []
        for x1, y1, x2, y2 in final_walls:
            wall = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            walls.append(wall)

        logger.info(f"Boundary detection complete: {len(walls)} walls")
        return walls, []

    def _find_boundary(
        self,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Find the outer boundary of the floor plan."""
        height, width = gray.shape

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Close gaps
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Fill holes - the interior of the floor plan
        # Use flood fill from corners
        filled = closed.copy()
        h, w = filled.shape

        # Flood fill from each corner (assumes corners are outside the floor plan)
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        cv2.floodFill(filled, flood_mask, (w-1, 0), 255)
        cv2.floodFill(filled, flood_mask, (0, h-1), 255)
        cv2.floodFill(filled, flood_mask, (w-1, h-1), 255)

        # Invert - now interior is white
        interior = cv2.bitwise_not(filled)

        # Find largest contour (the floor plan boundary)
        contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.ones_like(gray) * 255, None

        # Find largest by area
        largest = max(contours, key=cv2.contourArea)

        # Check if it's large enough to be the floor plan
        area = cv2.contourArea(largest)
        if area < (width * height * 0.05):  # At least 5% of image
            return np.ones_like(gray) * 255, None

        # Create boundary mask
        boundary_mask = np.zeros_like(gray)
        cv2.drawContours(boundary_mask, [largest], -1, 255, -1)

        return boundary_mask, largest

    def _extract_boundary_walls(
        self,
        contour: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Extract wall segments from boundary contour."""
        # Approximate contour with polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        walls = []
        n = len(approx)
        for i in range(n):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % n][0]

            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])

            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 20:
                walls.append((x1, y1, x2, y2))

        return walls

    def _extract_wall_mask(
        self,
        gray: np.ndarray,
        boundary_mask: np.ndarray
    ) -> np.ndarray:
        """Extract wall pixels within the boundary."""
        # Threshold to find dark lines (walls)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Keep only within boundary
        masked = cv2.bitwise_and(binary, boundary_mask)

        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)

        # Keep thick lines (walls) by removing thin ones
        # Erode then dilate with larger kernel
        erode_kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(cleaned, erode_kernel, iterations=1)

        dilate_kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(eroded, dilate_kernel, iterations=1)

        return dilated

    def _vectorize_walls(
        self,
        wall_mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Convert wall mask to line segments."""
        # Skeletonize
        if hasattr(cv2, 'ximgproc'):
            skeleton = cv2.ximgproc.thinning(wall_mask)
        else:
            skeleton = self._simple_skeleton(wall_mask)

        # Detect lines with Hough
        height, width = wall_mask.shape
        min_length = max(15, min(width, height) // 50)

        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=min_length,
            maxLineGap=10
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            walls.append((int(x1), int(y1), int(x2), int(y2)))

        return walls

    def _simple_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """Simple morphological skeleton."""
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

    def _build_wall_network(
        self,
        raw_walls: List[Tuple[int, int, int, int]],
        boundary_walls: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Build connected wall network from boundary + interior walls."""
        if not boundary_walls:
            return raw_walls

        # Start with boundary walls
        network = list(boundary_walls)
        network_endpoints = set()

        for x1, y1, x2, y2 in network:
            network_endpoints.add((x1, y1))
            network_endpoints.add((x2, y2))

        # Add interior walls that connect to the network
        remaining = list(raw_walls)
        changed = True

        while changed:
            changed = False
            still_remaining = []

            for wall in remaining:
                x1, y1, x2, y2 = wall

                # Check if either endpoint is near network
                connects = False
                for nx, ny in network_endpoints:
                    d1 = math.sqrt((x1 - nx)**2 + (y1 - ny)**2)
                    d2 = math.sqrt((x2 - nx)**2 + (y2 - ny)**2)

                    if d1 < 25 or d2 < 25:
                        connects = True
                        break

                if connects:
                    network.append(wall)
                    network_endpoints.add((x1, y1))
                    network_endpoints.add((x2, y2))
                    changed = True
                else:
                    still_remaining.append(wall)

            remaining = still_remaining

        return network

    def _cleanup_walls(
        self,
        walls: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Clean up walls: snap, align, merge, deduplicate."""
        if not walls:
            return walls

        # Snap to grid and align H/V
        grid = 5
        aligned = []

        for x1, y1, x2, y2 in walls:
            x1 = round(x1 / grid) * grid
            y1 = round(y1 / grid) * grid
            x2 = round(x2 / grid) * grid
            y2 = round(y2 / grid) * grid

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx == 0 and dy == 0:
                continue

            angle = math.degrees(math.atan2(dy, dx))

            if angle < 5:
                avg_y = round((y1 + y2) / 2 / grid) * grid
                y1 = y2 = avg_y
            elif angle > 85:
                avg_x = round((x1 + x2) / 2 / grid) * grid
                x1 = x2 = avg_x

            if math.sqrt((x2-x1)**2 + (y2-y1)**2) >= 10:
                aligned.append((x1, y1, x2, y2))

        # Merge collinear
        merged = self._merge_collinear(aligned)

        # Connect endpoints
        connected = self._connect_endpoints(merged)

        # Remove duplicates
        unique = self._remove_duplicates(connected)

        return unique

    def _merge_collinear(
        self,
        walls: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge collinear walls."""
        if len(walls) < 2:
            return walls

        used = set()
        result = []

        for i, wall1 in enumerate(walls):
            if i in used:
                continue

            x1, y1, x2, y2 = wall1
            used.add(i)
            points = [(x1, y1), (x2, y2)]

            for j, wall2 in enumerate(walls):
                if j in used:
                    continue

                x3, y3, x4, y4 = wall2

                # Check collinearity
                d1x, d1y = x2 - x1, y2 - y1
                d2x, d2y = x4 - x3, y4 - y3

                len1 = math.sqrt(d1x**2 + d1y**2)
                len2 = math.sqrt(d2x**2 + d2y**2)

                if len1 < 1 or len2 < 1:
                    continue

                dot = abs((d1x * d2x + d1y * d2y) / (len1 * len2))

                if dot > 0.98:
                    # Check distance
                    min_dist = min(
                        math.sqrt((x1-x3)**2 + (y1-y3)**2),
                        math.sqrt((x1-x4)**2 + (y1-y4)**2),
                        math.sqrt((x2-x3)**2 + (y2-y3)**2),
                        math.sqrt((x2-x4)**2 + (y2-y4)**2)
                    )

                    if min_dist < 20:
                        points.extend([(x3, y3), (x4, y4)])
                        used.add(j)

            # Find extremes
            if abs(y2 - y1) < abs(x2 - x1):
                points.sort(key=lambda p: p[0])
            else:
                points.sort(key=lambda p: p[1])

            result.append((points[0][0], points[0][1], points[-1][0], points[-1][1]))

        return result

    def _connect_endpoints(
        self,
        walls: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Snap nearby endpoints together."""
        if len(walls) < 2:
            return walls

        threshold = 15
        grid = 5

        endpoints = []
        for i, (x1, y1, x2, y2) in enumerate(walls):
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

                if math.sqrt((x1-x2)**2 + (y1-y2)**2) < threshold:
                    cluster.append((idx2, end2, x2, y2))
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        walls = list(walls)
        for cluster in clusters:
            cx = round(sum(p[2] for p in cluster) / len(cluster) / grid) * grid
            cy = round(sum(p[3] for p in cluster) / len(cluster) / grid) * grid

            for idx, end, _, _ in cluster:
                x1, y1, x2, y2 = walls[idx]
                if end == 0:
                    walls[idx] = (cx, cy, x2, y2)
                else:
                    walls[idx] = (x1, y1, cx, cy)

        return walls

    def _remove_duplicates(
        self,
        walls: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Remove duplicate walls."""
        unique = []

        for x1, y1, x2, y2 in walls:
            if x1 == x2 and y1 == y2:
                continue

            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            is_dup = False
            for ux1, uy1, ux2, uy2 in unique:
                d1 = math.sqrt((x1-ux1)**2 + (y1-uy1)**2)
                d2 = math.sqrt((x2-ux2)**2 + (y2-uy2)**2)

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
        """Detect walls from file."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return self.detect_walls(image, scale)
