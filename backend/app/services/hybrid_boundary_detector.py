"""Hybrid wall detection: Contour-based boundary + Rasterscan interior walls.

This combines two approaches for best accuracy:
1. Contour detection for the outer building boundary (perimeter)
2. Rasterscan HuggingFace API for interior room walls
"""

import logging
import tempfile
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

MATERIAL_ATTENUATION = {'concrete': 15.0, 'drywall': 3.0, 'default': 8.0}
RASTERSCAN_SPACE = "RasterScan/Automated-Floor-Plan-Digitalization"


class HybridBoundaryDetector:
    """
    Hybrid wall detector combining:
    - Contour detection for outer boundary
    - Rasterscan API for interior walls
    """

    def __init__(self):
        if not GRADIO_AVAILABLE:
            raise RuntimeError("gradio_client not installed. Run: pip install gradio_client")

        self.rasterscan_client = None
        self.scale = 0.05
        self._floor_plan_bounds = None  # Will store (x1, y1, x2, y2) of floor plan area

    def _ensure_rasterscan_client(self):
        """Lazy initialization of Rasterscan client."""
        if self.rasterscan_client is None:
            logger.info(f"Connecting to Rasterscan: {RASTERSCAN_SPACE}")
            self.rasterscan_client = Client(RASTERSCAN_SPACE)

    def detect_walls(self, image: np.ndarray, scale: float = 0.05) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls using hybrid approach.

        Args:
            image: BGR image as numpy array
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect outer boundary using contours
        logger.info("Step 1: Detecting outer boundary...")
        boundary_walls = self._detect_boundary(gray, w, h)
        logger.info(f"  Boundary walls: {len(boundary_walls)}")

        # Step 2: Get interior walls from Rasterscan
        logger.info("Step 2: Getting interior walls from Rasterscan...")
        interior_walls, rooms = self._get_rasterscan_walls(image)
        logger.info(f"  Interior walls: {len(interior_walls)}")

        # Step 3: Merge, removing duplicates
        logger.info("Step 3: Merging walls...")
        merged = self._merge_walls(boundary_walls, interior_walls)
        logger.info(f"  Final merged: {len(merged)} walls")

        # Convert to WallSegment objects
        wall_segments = [
            WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.15,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )
            for x1, y1, x2, y2 in merged
        ]

        return wall_segments, rooms

    def _detect_boundary(self, gray: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Detect outer boundary using convex hull of wall edges.

        Uses Canny edge detection to find wall edges, which works for both
        black walls and gray/colored walls. Then creates a convex hull.
        """

        # Use edge detection - works for any wall color
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges slightly to connect nearby lines
        kernel = np.ones((3, 3), np.uint8)
        wall_mask = cv2.dilate(edges, kernel, iterations=2)

        # Get all wall pixel coordinates
        wall_points = cv2.findNonZero(wall_mask)

        if wall_points is None or len(wall_points) < 100:
            logger.warning("Not enough wall pixels found for boundary detection")
            return []

        # Get convex hull of all wall pixels - this gives outer boundary
        hull = cv2.convexHull(wall_points)

        # Simplify hull to get clean boundary polygon
        epsilon = 0.01 * cv2.arcLength(hull, True)
        boundary = cv2.approxPolyDP(hull, epsilon, True)

        # Store bounds for filtering
        x, y, bw, bh = cv2.boundingRect(boundary)
        self._floor_plan_bounds = (x, y, x + bw, y + bh)

        # Convert to wall segments
        walls = []
        points = boundary.reshape(-1, 2)

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if length < 20:  # Skip very short segments
                continue

            # Snap to horizontal/vertical if close
            x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
            dx, dy = abs(x2 - x1), abs(y2 - y1)

            if dx < dy * 0.1:  # Nearly vertical
                avg_x = (x1 + x2) // 2
                walls.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))
            elif dy < dx * 0.1:  # Nearly horizontal
                avg_y = (y1 + y2) // 2
                walls.append((min(x1, x2), avg_y, max(x1, x2), avg_y))
            else:
                walls.append((x1, y1, x2, y2))

        logger.info(f"Boundary detection: {len(wall_points)} wall pixels -> {len(points)} hull points -> {len(walls)} wall segments")
        return walls

    def _get_rasterscan_walls(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Room]]:
        """Get walls from Rasterscan API."""
        self._ensure_rasterscan_client()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, image)

        try:
            result = self.rasterscan_client.predict(
                file=handle_file(temp_path),
                api_name="/run"
            )

            _, output_json = result

            # Parse walls
            walls = []
            for wall in output_json.get('walls', []):
                pos = wall.get('position', [])
                if len(pos) >= 2:
                    walls.append((
                        int(pos[0][0]), int(pos[0][1]),
                        int(pos[1][0]), int(pos[1][1])
                    ))

            # Parse rooms
            rooms = []
            for i, room_points in enumerate(output_json.get('rooms', [])):
                if not room_points:
                    continue
                polygon = []
                for pt in room_points:
                    if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                        polygon.append([float(pt['x']), float(pt['y'])])

                if len(polygon) >= 3:
                    area = self._polygon_area(polygon) * (self.scale ** 2)
                    rooms.append(Room(name=f"Room {i+1}", area=area, polygon=polygon))

            return walls, rooms

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _merge_walls(
        self,
        boundary: List[Tuple[int, int, int, int]],
        interior: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge boundary and interior walls, removing duplicates and noise."""

        # Filter boundary walls - remove any that are too far outside the main area
        filtered_boundary = []
        if self._floor_plan_bounds:
            bx1, by1, bx2, by2 = self._floor_plan_bounds
            margin = 30  # Allow small margin outside bounds
            for x1, y1, x2, y2 in boundary:
                # Check if wall is within or near the floor plan bounds
                if (x1 >= bx1 - margin and x2 <= bx2 + margin and
                    y1 >= by1 - margin and y2 <= by2 + margin):
                    filtered_boundary.append((x1, y1, x2, y2))
        else:
            filtered_boundary = list(boundary)

        merged = filtered_boundary

        for wall in interior:
            if not self._is_duplicate(wall, merged):
                merged.append(wall)

        return merged

    def _is_duplicate(self, wall: Tuple, existing: List[Tuple], threshold: float = 20) -> bool:
        """Check if wall is a duplicate of any existing wall."""
        x1, y1, x2, y2 = wall

        for ex1, ey1, ex2, ey2 in existing:
            # Check both orientations
            d1 = np.sqrt((x1-ex1)**2 + (y1-ey1)**2) + np.sqrt((x2-ex2)**2 + (y2-ey2)**2)
            d2 = np.sqrt((x1-ex2)**2 + (y1-ey2)**2) + np.sqrt((x2-ex1)**2 + (y2-ey1)**2)

            if min(d1, d2) < threshold * 2:
                return True

        return False

    def _polygon_area(self, polygon: List[List[float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(polygon)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]

        return abs(area) / 2.0


HYBRID_BOUNDARY_AVAILABLE = GRADIO_AVAILABLE
