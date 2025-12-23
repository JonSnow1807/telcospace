"""
Wall Post-Processor - Connect gaps and remove furniture.

Two focused tasks:
1. Connect disconnected wall segments (small gaps)
2. Remove furniture lines (isolated segments inside rooms)
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.schemas.project import WallSegment, Point

logger = logging.getLogger(__name__)


@dataclass
class WallLine:
    """Internal representation of a wall line."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y2 - self.y1) < abs(self.x2 - self.x1) * 0.15

    @property
    def is_vertical(self) -> bool:
        return abs(self.x2 - self.x1) < abs(self.y2 - self.y1) * 0.15

    def endpoint_distance(self, other: 'WallLine') -> float:
        """Minimum distance between any endpoints of two walls."""
        distances = [
            np.sqrt((self.x1 - other.x1)**2 + (self.y1 - other.y1)**2),
            np.sqrt((self.x1 - other.x2)**2 + (self.y1 - other.y2)**2),
            np.sqrt((self.x2 - other.x1)**2 + (self.y2 - other.y1)**2),
            np.sqrt((self.x2 - other.x2)**2 + (self.y2 - other.y2)**2),
        ]
        return min(distances)


class WallPostProcessor:
    """
    Post-process detected walls to:
    1. Connect small gaps between wall segments
    2. Remove isolated furniture lines
    """

    def __init__(
        self,
        gap_threshold: int = 15,      # Max gap to bridge (pixels)
        min_wall_length: int = 30,    # Minimum wall length to keep
        isolation_threshold: int = 50  # Distance to consider "isolated"
    ):
        self.gap_threshold = gap_threshold
        self.min_wall_length = min_wall_length
        self.isolation_threshold = isolation_threshold

    def process(
        self,
        walls: List[WallSegment],
        image: Optional[np.ndarray] = None
    ) -> List[WallSegment]:
        """
        Process walls: connect gaps and remove furniture.

        Args:
            walls: List of detected wall segments
            image: Original image (optional, for boundary detection)

        Returns:
            Cleaned list of wall segments
        """
        if not walls:
            return walls

        # Convert to internal format
        lines = [
            WallLine(
                int(w.start.x), int(w.start.y),
                int(w.end.x), int(w.end.y)
            )
            for w in walls
        ]

        original_count = len(lines)

        # Step 1: Connect gaps
        lines = self._connect_gaps(lines)
        logger.info(f"After connecting gaps: {len(lines)} walls")

        # Step 2: Remove isolated segments (furniture)
        lines = self._remove_isolated(lines, image)
        logger.info(f"After removing isolated: {len(lines)} walls")

        # Step 3: Remove short segments
        lines = [l for l in lines if l.length >= self.min_wall_length]
        logger.info(f"After length filter: {len(lines)} walls")

        logger.info(f"Post-processing: {original_count} → {len(lines)} walls")

        # Convert back to WallSegment
        return [
            WallSegment(
                start=Point(x=float(l.x1), y=float(l.y1)),
                end=Point(x=float(l.x2), y=float(l.y2)),
                thickness=0.15,
                material="concrete",
                attenuation_db=15.0
            )
            for l in lines
        ]

    def _connect_gaps(self, lines: List[WallLine]) -> List[WallLine]:
        """
        Connect wall segments that have small gaps between them.
        Only connects if walls are collinear (same line) or perpendicular (T/L junction).
        """
        if len(lines) < 2:
            return lines

        # Separate by orientation
        h_lines = [l for l in lines if l.is_horizontal]
        v_lines = [l for l in lines if l.is_vertical]
        other_lines = [l for l in lines if not l.is_horizontal and not l.is_vertical]

        # Connect horizontal lines on same Y
        h_lines = self._connect_collinear_horizontal(h_lines)

        # Connect vertical lines on same X
        v_lines = self._connect_collinear_vertical(v_lines)

        # Connect T-junctions and L-junctions
        all_lines = h_lines + v_lines + other_lines
        all_lines = self._connect_junctions(all_lines)

        # Extend endpoints to meet perpendicular wall lines (T-junction creation)
        all_lines = self._extend_to_perpendicular_lines(all_lines)

        return all_lines

    def _connect_collinear_horizontal(self, lines: List[WallLine]) -> List[WallLine]:
        """Connect horizontal lines that are on the same Y coordinate."""
        if len(lines) < 2:
            return lines

        # Group by Y coordinate (with tolerance)
        y_tolerance = 10
        groups = {}

        for line in lines:
            y = (line.y1 + line.y2) // 2
            found = False
            for key in list(groups.keys()):
                if abs(key - y) <= y_tolerance:
                    groups[key].append(line)
                    found = True
                    break
            if not found:
                groups[y] = [line]

        result = []
        for y_key, group in groups.items():
            # Sort by x position
            group.sort(key=lambda l: min(l.x1, l.x2))

            # Merge overlapping/close segments
            merged = []
            current = group[0]

            for line in group[1:]:
                current_end = max(current.x1, current.x2)
                line_start = min(line.x1, line.x2)

                # If gap is small enough, merge
                if line_start <= current_end + self.gap_threshold:
                    # Extend current line
                    new_end = max(current_end, max(line.x1, line.x2))
                    current = WallLine(
                        min(current.x1, current.x2),
                        y_key,
                        new_end,
                        y_key
                    )
                else:
                    merged.append(current)
                    current = line

            merged.append(current)
            result.extend(merged)

        return result

    def _connect_collinear_vertical(self, lines: List[WallLine]) -> List[WallLine]:
        """Connect vertical lines that are on the same X coordinate."""
        if len(lines) < 2:
            return lines

        x_tolerance = 10
        groups = {}

        for line in lines:
            x = (line.x1 + line.x2) // 2
            found = False
            for key in list(groups.keys()):
                if abs(key - x) <= x_tolerance:
                    groups[key].append(line)
                    found = True
                    break
            if not found:
                groups[x] = [line]

        result = []
        for x_key, group in groups.items():
            group.sort(key=lambda l: min(l.y1, l.y2))

            merged = []
            current = group[0]

            for line in group[1:]:
                current_end = max(current.y1, current.y2)
                line_start = min(line.y1, line.y2)

                if line_start <= current_end + self.gap_threshold:
                    new_end = max(current_end, max(line.y1, line.y2))
                    current = WallLine(
                        x_key,
                        min(current.y1, current.y2),
                        x_key,
                        new_end
                    )
                else:
                    merged.append(current)
                    current = line

            merged.append(current)
            result.extend(merged)

        return result

    def _connect_junctions(self, lines: List[WallLine]) -> List[WallLine]:
        """
        Connect walls at T-junctions and L-junctions.
        Extends walls slightly to meet at corners.
        """
        if len(lines) < 2:
            return lines

        # For each wall, check if any endpoint is close to another wall
        # If so, extend to meet

        result = list(lines)

        for i, line1 in enumerate(result):
            for j, line2 in enumerate(result):
                if i >= j:
                    continue

                # Check if endpoints are close
                dist = line1.endpoint_distance(line2)

                if dist < self.gap_threshold and dist > 3:
                    # Check if they're perpendicular (T or L junction)
                    if (line1.is_horizontal and line2.is_vertical) or \
                       (line1.is_vertical and line2.is_horizontal):
                        # Extend to meet
                        result[i], result[j] = self._extend_to_meet(line1, line2)

        return result

    def _extend_to_meet(self, line1: WallLine, line2: WallLine) -> Tuple[WallLine, WallLine]:
        """Extend two perpendicular lines to meet at a corner."""
        # Find which endpoints are closest
        endpoints1 = [(line1.x1, line1.y1), (line1.x2, line1.y2)]
        endpoints2 = [(line2.x1, line2.y1), (line2.x2, line2.y2)]

        min_dist = float('inf')
        closest = None

        for i, (x1, y1) in enumerate(endpoints1):
            for j, (x2, y2) in enumerate(endpoints2):
                d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if d < min_dist:
                    min_dist = d
                    closest = (i, j, x1, y1, x2, y2)

        if closest is None or min_dist > self.gap_threshold:
            return line1, line2

        i, j, x1, y1, x2, y2 = closest

        # Calculate meeting point
        if line1.is_horizontal:
            # Horizontal line: keep its Y, use vertical line's X
            meet_x = (line2.x1 + line2.x2) // 2
            meet_y = (line1.y1 + line1.y2) // 2
        else:
            # Vertical line: keep its X, use horizontal line's Y
            meet_x = (line1.x1 + line1.x2) // 2
            meet_y = (line2.y1 + line2.y2) // 2

        # Extend line1
        if i == 0:  # First endpoint is closest
            new_line1 = WallLine(meet_x, meet_y, line1.x2, line1.y2)
        else:
            new_line1 = WallLine(line1.x1, line1.y1, meet_x, meet_y)

        # Extend line2
        if j == 0:
            new_line2 = WallLine(meet_x, meet_y, line2.x2, line2.y2)
        else:
            new_line2 = WallLine(line2.x1, line2.y1, meet_x, meet_y)

        return new_line1, new_line2

    def _extend_to_perpendicular_lines(self, lines: List[WallLine], threshold: int = 60) -> List[WallLine]:
        """
        Extend wall endpoints to meet nearby perpendicular wall LINES (T-junction creation).

        This is different from _connect_junctions which only connects nearby endpoints.
        This method extends a wall to meet the LINE of another wall, even if the
        other wall's endpoints are far away.
        """
        if len(lines) < 2:
            return lines

        result = [WallLine(l.x1, l.y1, l.x2, l.y2) for l in lines]
        modified = True
        iterations = 0
        max_iterations = 3  # Prevent infinite loops

        while modified and iterations < max_iterations:
            modified = False
            iterations += 1

            for i, line1 in enumerate(result):
                for end_idx, (ex, ey) in enumerate([(line1.x1, line1.y1), (line1.x2, line1.y2)]):
                    best_extension = None
                    best_dist = threshold

                    for j, line2 in enumerate(result):
                        if i == j:
                            continue

                        # Only connect perpendicular walls
                        if not ((line1.is_horizontal and line2.is_vertical) or
                                (line1.is_vertical and line2.is_horizontal)):
                            continue

                        if line2.is_vertical:
                            # line2 is vertical at x = line2.x1 (approximately)
                            line2_x = (line2.x1 + line2.x2) // 2
                            y_min, y_max = min(line2.y1, line2.y2), max(line2.y1, line2.y2)

                            # Check if endpoint Y is within the line's Y-range (with threshold extension)
                            if y_min - threshold <= ey <= y_max + threshold:
                                dist = abs(ex - line2_x)
                                if dist <= best_dist and dist > 3:  # Avoid already-connected endpoints
                                    best_dist = dist
                                    best_extension = ('vertical', line2_x, ey, end_idx)
                        else:
                            # line2 is horizontal at y = line2.y1 (approximately)
                            line2_y = (line2.y1 + line2.y2) // 2
                            x_min, x_max = min(line2.x1, line2.x2), max(line2.x1, line2.x2)

                            # Check if endpoint X is within the line's X-range (with threshold extension)
                            if x_min - threshold <= ex <= x_max + threshold:
                                dist = abs(ey - line2_y)
                                if dist <= best_dist and dist > 3:
                                    best_dist = dist
                                    best_extension = ('horizontal', ex, line2_y, end_idx)

                    # Apply the best extension found
                    if best_extension:
                        orientation, new_x, new_y, endpoint_idx = best_extension
                        if endpoint_idx == 0:
                            result[i] = WallLine(new_x, new_y, line1.x2, line1.y2)
                        else:
                            result[i] = WallLine(line1.x1, line1.y1, new_x, new_y)
                        modified = True
                        logger.debug(f"Extended wall {i} endpoint to ({new_x}, {new_y})")

        logger.info(f"Endpoint-to-line extension: {iterations} iterations")
        return result

    def _remove_isolated(
        self,
        lines: List[WallLine],
        image: Optional[np.ndarray] = None
    ) -> List[WallLine]:
        """
        Remove isolated line segments (likely furniture).

        A line is considered isolated if:
        1. It doesn't connect to any other wall (no nearby endpoints)
        2. It's relatively short
        """
        if len(lines) < 2:
            return lines

        # Build connectivity: count how many walls each wall connects to
        connectivity = [0] * len(lines)

        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if i >= j:
                    continue

                dist = line1.endpoint_distance(line2)

                if dist < self.isolation_threshold:
                    connectivity[i] += 1
                    connectivity[j] += 1

        # Keep walls that:
        # 1. Connect to at least 1 other wall, OR
        # 2. Are long enough to be significant (likely outer boundary)

        result = []
        for i, line in enumerate(lines):
            is_connected = connectivity[i] >= 1
            is_long = line.length > self.min_wall_length * 2

            if is_connected or is_long:
                result.append(line)
            else:
                logger.debug(f"Removing isolated segment: ({line.x1},{line.y1})->({line.x2},{line.y2})")

        return result


def post_process_walls(
    walls: List[WallSegment],
    image: Optional[np.ndarray] = None,
    gap_threshold: int = 25,
    min_wall_length: int = 40,
    isolation_threshold: int = 80
) -> List[WallSegment]:
    """
    Convenience function to post-process walls.

    Args:
        walls: Detected wall segments
        image: Original floor plan image (optional)
        gap_threshold: Maximum gap to bridge (pixels)
        min_wall_length: Minimum wall length to keep

    Returns:
        Cleaned wall segments
    """
    processor = WallPostProcessor(
        gap_threshold=gap_threshold,
        min_wall_length=min_wall_length,
        isolation_threshold=isolation_threshold
    )
    return processor.process(walls, image)


class ContourBoundaryProcessor:
    """
    Find and close the outer boundary using contour detection on a wall image.

    This approach:
    1. Draws all walls onto a binary image
    2. Dilates to connect nearby walls
    3. Finds the outer contour (this IS the closed boundary)
    4. Simplifies to line segments
    5. Adds missing segments to close gaps
    """

    def __init__(
        self,
        dilation_kernel: int = 15,  # How much to dilate walls to find boundary
        simplify_epsilon: float = 5.0,  # Contour simplification tolerance
    ):
        self.dilation_kernel = dilation_kernel
        self.simplify_epsilon = simplify_epsilon

    def process(
        self,
        walls: List[WallSegment],
        image_shape: Tuple[int, int]  # (height, width)
    ) -> List[WallSegment]:
        """
        Find outer boundary and return it as wall segments.

        Args:
            walls: Detected wall segments
            image_shape: (height, width) of the floor plan image

        Returns:
            Wall segments forming a closed outer boundary
        """
        if not walls or len(walls) < 3:
            return walls

        height, width = image_shape

        # Step 1: Draw walls onto binary image
        wall_img = np.zeros((height, width), dtype=np.uint8)
        for w in walls:
            pt1 = (int(w.start.x), int(w.start.y))
            pt2 = (int(w.end.x), int(w.end.y))
            cv2.line(wall_img, pt1, pt2, 255, thickness=3)

        # Step 2: Dilate to connect nearby walls
        kernel = np.ones((self.dilation_kernel, self.dilation_kernel), np.uint8)
        dilated = cv2.dilate(wall_img, kernel, iterations=1)

        # Step 3: Fill holes to get solid floor plan shape
        # Flood fill from corners (outside the floor plan)
        filled = dilated.copy()
        h, w = filled.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, mask, (0, 0), 255)
        filled = cv2.bitwise_not(filled)  # Invert: floor plan is white
        filled = cv2.bitwise_or(dilated, filled)  # Combine with original

        # Step 4: Find outer contour
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found for boundary detection")
            return walls

        # Get largest contour (the floor plan boundary)
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 5: Simplify contour to polygon
        epsilon = self.simplify_epsilon
        simplified = cv2.approxPolyDP(largest_contour, epsilon, closed=True)

        # Step 6: Convert contour points to wall segments
        boundary_walls = []
        points = simplified.reshape(-1, 2)

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            boundary_walls.append(WallSegment(
                start=Point(x=float(p1[0]), y=float(p1[1])),
                end=Point(x=float(p2[0]), y=float(p2[1])),
                thickness=0.15,
                material="concrete",
                attenuation_db=15.0
            ))

        logger.info(f"Contour boundary: {len(boundary_walls)} segments from {len(points)} vertices")

        # Return ONLY the boundary (closed polygon)
        # The original interior walls are not included - this is just the boundary
        return boundary_walls


def extract_outer_boundary(
    walls: List[WallSegment],
    image_shape: Tuple[int, int]
) -> List[WallSegment]:
    """
    Extract a closed outer boundary from detected walls.

    Args:
        walls: All detected wall segments
        image_shape: (height, width) of the floor plan

    Returns:
        Wall segments forming closed outer boundary
    """
    processor = ContourBoundaryProcessor()
    return processor.process(walls, image_shape)
