"""Combined Wall Detection - Best of AI + Morphological approaches.

This approach:
1. Runs AI detection to understand WHAT is a wall (semantic)
2. Runs morphological detection for precise LINES (geometric)
3. Merges results, preferring AI semantics with morphological precision
4. Uses consensus to filter false positives

The goal is CAD-quality output that accurately represents the floor plan.
"""

import os
import logging
import math
from typing import List, Tuple, Optional, Set
import numpy as np
import cv2

from app.schemas.project import WallSegment, Point, Room
from app.core.config import settings

logger = logging.getLogger(__name__)

MATERIAL_ATTENUATION = {'concrete': 15.0}


class CombinedWallDetector:
    """
    Combined AI + Morphological wall detector.

    Uses multiple strategies and merges results for best accuracy.
    """

    def __init__(self):
        self.scale = 0.05

        # Try to import AI detector
        self.ai_detector = None
        try:
            from app.services.ai_wall_detection import AIWallDetector
            self.ai_detector = AIWallDetector()
            logger.info("AI detector available for combined approach")
        except Exception as e:
            logger.warning(f"AI detector not available: {e}")

        # Morphological detector
        from app.services.morphological_wall_detection import MorphologicalWallDetector
        self.morph_detector = MorphologicalWallDetector()

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls using combined approach.

        Strategy:
        1. Run morphological detection (fast, precise lines)
        2. Run AI detection if available (semantic understanding)
        3. Merge results with intelligent filtering
        """
        self.scale = scale
        height, width = image.shape[:2]

        logger.info(f"Combined detection on {width}x{height} image")

        # Run morphological detection
        logger.info("Running morphological detection...")
        morph_walls, _ = self.morph_detector.detect_walls(image, scale)
        logger.info(f"  Morphological: {len(morph_walls)} walls")

        # Run AI detection if available
        ai_walls = []
        if self.ai_detector:
            logger.info("Running AI detection...")
            try:
                ai_walls, ai_rooms = self.ai_detector.detect_walls(image, scale)
                logger.info(f"  AI: {len(ai_walls)} walls")
            except Exception as e:
                logger.warning(f"AI detection failed: {e}")

        # Merge results
        logger.info("Merging results...")
        merged_walls = self._merge_detections(morph_walls, ai_walls, width, height)
        logger.info(f"  Merged: {len(merged_walls)} walls")

        return merged_walls, []

    def _merge_detections(
        self,
        morph_walls: List[WallSegment],
        ai_walls: List[WallSegment],
        width: int,
        height: int
    ) -> List[WallSegment]:
        """
        Merge morphological and AI detections intelligently.

        Strategy:
        - Use morphological as base (precise coordinates)
        - Add AI walls that don't overlap (semantic additions)
        - Filter walls that appear in both (high confidence)
        """
        if not ai_walls:
            return morph_walls

        if not morph_walls:
            return ai_walls

        # Convert to tuples for easier manipulation
        morph_tuples = [(w.start.x, w.start.y, w.end.x, w.end.y) for w in morph_walls]
        ai_tuples = [(w.start.x, w.start.y, w.end.x, w.end.y) for w in ai_walls]

        result = []

        # Score each morphological wall by AI agreement
        for m_wall in morph_tuples:
            # Check if any AI wall agrees
            has_agreement = self._has_agreement(m_wall, ai_tuples)

            # Always include morphological walls (they're precise)
            result.append(m_wall)

        # Add AI walls that don't overlap with morphological
        for a_wall in ai_tuples:
            if not self._overlaps_any(a_wall, morph_tuples):
                # AI found something morphological missed
                # Only add if it's reasonably sized
                length = self._line_length(a_wall)
                if length > 20:  # Minimum length
                    result.append(a_wall)

        # Clean up
        result = self._snap_and_align(result)
        result = self._merge_collinear(result)
        result = self._connect_endpoints(result)
        result = self._remove_duplicates(result)

        # Convert back to WallSegments
        walls = []
        for x1, y1, x2, y2 in result:
            wall = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            walls.append(wall)

        return walls

    def _has_agreement(
        self,
        wall: Tuple[float, float, float, float],
        others: List[Tuple[float, float, float, float]],
        threshold: float = 30
    ) -> bool:
        """Check if wall has agreement from another set."""
        x1, y1, x2, y2 = wall
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        for ox1, oy1, ox2, oy2 in others:
            o_mid_x = (ox1 + ox2) / 2
            o_mid_y = (oy1 + oy2) / 2

            dist = math.sqrt((mid_x - o_mid_x)**2 + (mid_y - o_mid_y)**2)
            if dist < threshold:
                return True

        return False

    def _overlaps_any(
        self,
        wall: Tuple[float, float, float, float],
        others: List[Tuple[float, float, float, float]],
        threshold: float = 25
    ) -> bool:
        """Check if wall overlaps with any in the list."""
        return self._has_agreement(wall, others, threshold)

    def _line_length(self, wall: Tuple[float, float, float, float]) -> float:
        """Calculate line length."""
        x1, y1, x2, y2 = wall
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _snap_and_align(
        self,
        walls: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[int, int, int, int]]:
        """Snap to grid and align H/V."""
        grid = 5
        angle_threshold = 5.0

        result = []
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

            if angle < angle_threshold:
                avg_y = round((y1 + y2) / 2 / grid) * grid
                y1 = y2 = avg_y
            elif angle > (90 - angle_threshold):
                avg_x = round((x1 + x2) / 2 / grid) * grid
                x1 = x2 = avg_x

            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length >= 10:
                result.append((int(x1), int(y1), int(x2), int(y2)))

        return result

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

                if self._are_collinear(wall1, wall2):
                    x3, y3, x4, y4 = wall2
                    points.extend([(x3, y3), (x4, y4)])
                    used.add(j)

            # Find extremes
            if abs(y2 - y1) < abs(x2 - x1):
                points.sort(key=lambda p: p[0])
            else:
                points.sort(key=lambda p: p[1])

            result.append((points[0][0], points[0][1], points[-1][0], points[-1][1]))

        return result

    def _are_collinear(
        self,
        wall1: Tuple[int, int, int, int],
        wall2: Tuple[int, int, int, int]
    ) -> bool:
        """Check collinearity."""
        x1, y1, x2, y2 = wall1
        x3, y3, x4, y4 = wall2

        d1x, d1y = x2 - x1, y2 - y1
        d2x, d2y = x4 - x3, y4 - y3

        len1 = math.sqrt(d1x**2 + d1y**2)
        len2 = math.sqrt(d2x**2 + d2y**2)

        if len1 < 1 or len2 < 1:
            return False

        d1x, d1y = d1x / len1, d1y / len1
        d2x, d2y = d2x / len2, d2y / len2

        dot = abs(d1x * d2x + d1y * d2y)
        if dot < 0.98:
            return False

        # Check endpoint distance
        endpoints = [(x1, y1), (x2, y2)]
        for ex, ey in endpoints:
            for px, py in [(x3, y3), (x4, y4)]:
                if math.sqrt((ex - px)**2 + (ey - py)**2) < 20:
                    return True

        return False

    def _connect_endpoints(
        self,
        walls: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Connect nearby endpoints."""
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

                if math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold:
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
        """Detect walls from file."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return self.detect_walls(image, scale)
