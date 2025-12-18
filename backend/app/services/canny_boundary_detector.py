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

        # Post-process: merge collinear segments and extend to corners
        logger.info(f"Post-processing {len(walls)} walls...")
        walls = self._post_process_walls(walls)
        logger.info(f"  After post-processing: {len(walls)} walls")

        return walls

    def _post_process_walls(self, walls: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Post-process walls: merge collinear segments and extend to corners."""
        if not walls:
            return walls

        # Separate horizontal and vertical walls
        h_walls = [(x1, y1, x2, y2) for x1, y1, x2, y2 in walls if y1 == y2]
        v_walls = [(x1, y1, x2, y2) for x1, y1, x2, y2 in walls if x1 == x2]

        # Merge collinear segments
        merged_h = self._merge_collinear_horizontal(h_walls)
        merged_v = self._merge_collinear_vertical(v_walls)

        # Extend walls to meet at corners
        extended_h, extended_v = self._extend_to_corners(merged_h, merged_v)

        # Connect nearby endpoints
        connected_h, connected_v = self._connect_nearby_endpoints(extended_h, extended_v)

        return connected_h + connected_v

    def _merge_collinear_horizontal(self, walls: List[Tuple], gap_threshold: int = 60, y_tolerance: int = 20) -> List[Tuple]:
        """Merge horizontal walls that are on the same line and close together."""
        if not walls:
            return []

        # Group by y-coordinate (with tolerance)
        groups = {}
        for x1, y1, x2, y2 in walls:
            y = y1
            found = False
            for key in list(groups.keys()):
                if abs(key - y) <= y_tolerance:
                    groups[key].append((min(x1, x2), max(x1, x2), y))
                    found = True
                    break
            if not found:
                groups[y] = [(min(x1, x2), max(x1, x2), y)]

        merged = []
        for y_key, segments in groups.items():
            segments.sort(key=lambda s: s[0])
            current = list(segments[0])
            for x1, x2, y in segments[1:]:
                if x1 <= current[1] + gap_threshold:
                    current[1] = max(current[1], x2)
                    current[2] = (current[2] + y) // 2  # Average y
                else:
                    merged.append((current[0], current[2], current[1], current[2]))
                    current = [x1, x2, y]
            merged.append((current[0], current[2], current[1], current[2]))

        return merged

    def _merge_collinear_vertical(self, walls: List[Tuple], gap_threshold: int = 60, x_tolerance: int = 20) -> List[Tuple]:
        """Merge vertical walls that are on the same line and close together."""
        if not walls:
            return []

        # Group by x-coordinate (with tolerance)
        groups = {}
        for x1, y1, x2, y2 in walls:
            x = x1
            found = False
            for key in list(groups.keys()):
                if abs(key - x) <= x_tolerance:
                    groups[key].append((min(y1, y2), max(y1, y2), x))
                    found = True
                    break
            if not found:
                groups[x] = [(min(y1, y2), max(y1, y2), x)]

        merged = []
        for x_key, segments in groups.items():
            segments.sort(key=lambda s: s[0])
            current = list(segments[0])
            for y1, y2, x in segments[1:]:
                if y1 <= current[1] + gap_threshold:
                    current[1] = max(current[1], y2)
                    current[2] = (current[2] + x) // 2  # Average x
                else:
                    merged.append((current[2], current[0], current[2], current[1]))
                    current = [y1, y2, x]
            merged.append((current[2], current[0], current[2], current[1]))

        return merged

    def _extend_to_corners(
        self,
        h_walls: List[Tuple],
        v_walls: List[Tuple],
        snap_distance: int = 50
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Extend walls to meet at corners where they're close."""
        extended_h = []
        extended_v = list(v_walls)

        for hx1, hy, hx2, _ in h_walls:
            new_x1, new_x2 = hx1, hx2

            # Check if left end should extend to meet a vertical wall
            for i, (vx, vy1, _, vy2) in enumerate(extended_v):
                # Left end of H meets V
                if abs(hx1 - vx) <= snap_distance and vy1 <= hy <= vy2:
                    new_x1 = vx
                # Right end of H meets V
                if abs(hx2 - vx) <= snap_distance and vy1 <= hy <= vy2:
                    new_x2 = vx

            extended_h.append((new_x1, hy, new_x2, hy))

        # Now extend vertical walls to meet horizontal
        final_v = []
        for vx, vy1, _, vy2 in extended_v:
            new_y1, new_y2 = vy1, vy2

            for hx1, hy, hx2, _ in extended_h:
                # Top end of V meets H
                if abs(vy1 - hy) <= snap_distance and hx1 <= vx <= hx2:
                    new_y1 = hy
                # Bottom end of V meets H
                if abs(vy2 - hy) <= snap_distance and hx1 <= vx <= hx2:
                    new_y2 = hy

            final_v.append((vx, new_y1, vx, new_y2))

        return extended_h, final_v

    def _connect_nearby_endpoints(
        self,
        h_walls: List[Tuple],
        v_walls: List[Tuple],
        connect_distance: int = 40
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Connect wall endpoints that are close to each other."""
        # Build list of all endpoints
        h_endpoints = []  # (wall_idx, is_start, x, y)
        for i, (x1, y, x2, _) in enumerate(h_walls):
            h_endpoints.append((i, True, x1, y))   # start
            h_endpoints.append((i, False, x2, y))  # end

        v_endpoints = []
        for i, (x, y1, _, y2) in enumerate(v_walls):
            v_endpoints.append((i, True, x, y1))   # start (top)
            v_endpoints.append((i, False, x, y2))  # end (bottom)

        # Convert to mutable lists
        new_h = [list(w) for w in h_walls]
        new_v = [list(w) for w in v_walls]

        # For each H endpoint, find nearby V endpoints to connect
        for h_idx, is_start, hx, hy in h_endpoints:
            for v_idx, v_is_start, vx, vy in v_endpoints:
                dist = np.sqrt((hx - vx)**2 + (hy - vy)**2)
                if dist <= connect_distance and dist > 0:
                    # Snap H wall endpoint to V wall
                    if is_start:
                        new_h[h_idx][0] = vx  # x1
                        new_h[h_idx][1] = vy  # y1
                    else:
                        new_h[h_idx][2] = vx  # x2
                        new_h[h_idx][3] = vy  # y2 (keep same as y1 for H)

                    # Snap V wall endpoint to H wall
                    if v_is_start:
                        new_v[v_idx][1] = hy  # y1
                    else:
                        new_v[v_idx][3] = hy  # y2

        # Convert back to tuples and ensure H walls have same y
        result_h = []
        for x1, y1, x2, y2 in new_h:
            avg_y = (y1 + y2) // 2
            result_h.append((x1, avg_y, x2, avg_y))

        # Ensure V walls have same x
        result_v = []
        for x1, y1, x2, y2 in new_v:
            avg_x = (x1 + x2) // 2
            result_v.append((avg_x, y1, avg_x, y2))

        return result_h, result_v

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
