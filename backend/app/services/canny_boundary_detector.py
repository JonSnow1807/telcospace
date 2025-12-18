"""Wall detection for floor plans using skeletonization.

Detects actual wall lines by:
1. Thresholding to find wall pixels (dark gray/black)
2. Skeletonizing to get wall centerlines
3. Using Hough transform to extract line segments
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

MATERIAL_ATTENUATION = {'concrete': 15.0, 'drywall': 3.0, 'default': 8.0}


class CannyBoundaryDetector:
    """
    Wall detector using skeletonization + Hough transform.

    Detects actual wall lines by:
    1. Thresholding to find wall pixels (dark gray/black)
    2. Skeletonizing to get wall centerlines
    3. Using Hough transform to extract line segments
    """

    def __init__(self):
        self.scale = 0.05
        self._floor_plan_bounds = None

    def detect_walls(self, image: np.ndarray, scale: float = 0.05) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls using skeleton-based line detection.

        Args:
            image: BGR image as numpy array
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect actual wall lines using skeleton
        logger.info("Detecting wall lines using skeletonization...")
        wall_lines = self._detect_wall_lines(gray, w, h)
        logger.info(f"  Detected {len(wall_lines)} wall segments")

        # Convert to WallSegment objects
        wall_segments = [
            WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.15,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )
            for x1, y1, x2, y2 in wall_lines
        ]

        # Simple room detection (placeholder)
        rooms = []

        return wall_segments, rooms

    def _detect_wall_lines(self, gray: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Detect actual wall lines using skeletonization + Hough transform."""

        if not SKIMAGE_AVAILABLE:
            logger.warning("skimage not available, falling back to edge detection")
            return self._detect_walls_fallback(gray, w, h)

        # Find wall pixels (dark areas - walls are gray/black)
        # Use threshold to get wall mask
        _, wall_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)

        # Skeletonize to get wall centerlines
        skeleton = skeletonize(wall_mask > 0)
        skeleton_img = (skeleton * 255).astype(np.uint8)

        # Use HoughLinesP on skeleton to get line segments
        lines = cv2.HoughLinesP(
            skeleton_img,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=40,
            maxLineGap=15
        )

        if lines is None:
            logger.warning("No lines detected from skeleton")
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = np.sqrt(dx**2 + dy**2)

            if length < 30:
                continue

            angle = np.arctan2(dy, dx) * 180 / np.pi

            # Keep mostly horizontal or vertical lines
            if angle < 25:  # Horizontal
                avg_y = (y1 + y2) // 2
                walls.append((min(x1, x2), avg_y, max(x1, x2), avg_y))
            elif angle > 65:  # Vertical
                avg_x = (x1 + x2) // 2
                walls.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))

        # Store bounds
        if walls:
            all_x = [w[0] for w in walls] + [w[2] for w in walls]
            all_y = [w[1] for w in walls] + [w[3] for w in walls]
            self._floor_plan_bounds = (min(all_x), min(all_y), max(all_x), max(all_y))

        return walls

    def _detect_walls_fallback(self, gray: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Fallback wall detection using edge detection if skimage not available."""
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            angle = np.arctan2(dy, dx) * 180 / np.pi

            if angle < 20 or angle > 70:
                walls.append((x1, y1, x2, y2))

        return walls

    def _detect_boundary(self, gray: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Detect outer boundary using contour tracing.

        Uses threshold + contour detection to find the actual floor plan shape,
        which preserves non-convex shapes (L-shapes, indentations, etc).
        """
        # Threshold to find the white/light background vs dark elements
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Invert so floor plan interior is white
        inverted = cv2.bitwise_not(binary)

        # Find contours
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found for boundary detection")
            return []

        # Get the largest contour (should be the floor plan)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < (w * h * 0.05):  # Less than 5% of image
            logger.warning(f"Largest contour too small: {area} pixels")
            return []

        # Simplify the contour to get clean wall segments
        epsilon = 0.005 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)

        # Store bounds for filtering
        x, y, bw, bh = cv2.boundingRect(simplified)
        self._floor_plan_bounds = (x, y, x + bw, y + bh)

        # Convert to wall segments
        walls = []
        points = simplified.reshape(-1, 2)

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if length < 15:  # Skip very short segments
                continue

            # Snap to horizontal/vertical if close
            x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
            dx, dy = abs(x2 - x1), abs(y2 - y1)

            if dx < dy * 0.15:  # Nearly vertical
                avg_x = (x1 + x2) // 2
                walls.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))
            elif dy < dx * 0.15:  # Nearly horizontal
                avg_y = (y1 + y2) // 2
                walls.append((min(x1, x2), avg_y, max(x1, x2), avg_y))
            else:
                walls.append((x1, y1, x2, y2))

        logger.info(f"Boundary detection: {area:.0f} px area -> {len(points)} contour points -> {len(walls)} wall segments")
        return walls

    def _detect_interior_walls(self, gray: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Detect interior walls using Hough lines within the boundary."""
        if not self._floor_plan_bounds:
            return []

        bx1, by1, bx2, by2 = self._floor_plan_bounds

        # Create mask for interior region
        interior_mask = np.zeros(gray.shape, dtype=np.uint8)
        interior_mask[by1:by2, bx1:bx2] = 255

        # Apply adaptive threshold to find walls
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Mask to interior only
        binary = cv2.bitwise_and(binary, interior_mask)

        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Filter: must be inside boundary
            if not (bx1 <= x1 <= bx2 and bx1 <= x2 <= bx2 and
                    by1 <= y1 <= by2 and by1 <= y2 <= by2):
                continue

            # Filter: must be roughly horizontal or vertical
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            angle = np.arctan2(dy, dx) * 180 / np.pi

            if angle < 15 or angle > 75:  # Nearly H or V
                # Snap to H/V
                if dx < dy * 0.2:  # Vertical
                    avg_x = (x1 + x2) // 2
                    walls.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))
                elif dy < dx * 0.2:  # Horizontal
                    avg_y = (y1 + y2) // 2
                    walls.append((min(x1, x2), avg_y, max(x1, x2), avg_y))

        return walls

    def _merge_walls(
        self,
        boundary: List[Tuple[int, int, int, int]],
        interior: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge boundary and interior walls, removing duplicates."""
        merged = list(boundary)

        for wall in interior:
            if not self._is_duplicate(wall, merged) and not self._is_near_boundary(wall, boundary):
                merged.append(wall)

        return merged

    def _is_duplicate(self, wall: Tuple, existing: List[Tuple], threshold: float = 30) -> bool:
        """Check if wall is a duplicate of any existing wall."""
        x1, y1, x2, y2 = wall

        for ex1, ey1, ex2, ey2 in existing:
            d1 = np.sqrt((x1-ex1)**2 + (y1-ey1)**2) + np.sqrt((x2-ex2)**2 + (y2-ey2)**2)
            d2 = np.sqrt((x1-ex2)**2 + (y1-ey2)**2) + np.sqrt((x2-ex1)**2 + (y2-ey1)**2)

            if min(d1, d2) < threshold * 2:
                return True

        return False

    def _is_near_boundary(self, wall: Tuple, boundary: List[Tuple], threshold: float = 20) -> bool:
        """Check if wall is too close to boundary (likely part of it)."""
        x1, y1, x2, y2 = wall
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        for bx1, by1, bx2, by2 in boundary:
            # Check if midpoint is near boundary line
            if bx1 == bx2:  # Vertical boundary
                if abs(mid_x - bx1) < threshold and min(by1, by2) <= mid_y <= max(by1, by2):
                    return True
            elif by1 == by2:  # Horizontal boundary
                if abs(mid_y - by1) < threshold and min(bx1, bx2) <= mid_x <= max(bx1, bx2):
                    return True

        return False

    def _detect_rooms_from_boundary(
        self,
        walls: List[WallSegment],
        w: int,
        h: int
    ) -> List[Room]:
        """Create a simple room from the boundary."""
        if not walls:
            return []

        # Get all points from walls
        points = []
        for wall in walls:
            points.append([wall.start.x, wall.start.y])
            points.append([wall.end.x, wall.end.y])

        if len(points) < 3:
            return []

        # Calculate area using shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0

        # Convert to square meters
        area_sqm = area * (self.scale ** 2)

        return [Room(
            name="Main Area",
            area=area_sqm,
            polygon=points
        )]


CANNY_BOUNDARY_AVAILABLE = True
