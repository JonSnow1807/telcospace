"""Hybrid AI + Graph Algorithm Wall Detection.

Combines:
1. AI semantic segmentation - Claude identifies wall regions (binary mask)
2. Skeletonization - Extract 1-pixel centerlines
3. Hough Transform - Detect line segments from skeleton
4. Graph algorithm - Clean up wall network (merge collinear, snap intersections)

This approach gives CAD-quality output by leveraging AI's semantic understanding
to identify what IS a wall (ignoring furniture, text, etc.) while using precise
CV algorithms for exact coordinate extraction.
"""

import os
import base64
import json
import logging
import math
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict
import numpy as np
import cv2

import anthropic

from app.schemas.project import WallSegment, Point, Room
from app.core.config import settings

logger = logging.getLogger(__name__)

# Material attenuation database
MATERIAL_ATTENUATION = {
    'concrete': 15.0,
    'brick': 12.0,
    'wood': 6.0,
    'glass': 5.0,
    'drywall': 3.0,
    'metal': 25.0,
    'unknown': 10.0
}

# AI prompt for semantic wall segmentation
WALL_SEGMENTATION_PROMPT = """You are analyzing a floor plan image to identify ONLY the structural walls.

**YOUR TASK:** Create a JSON description of rectangular regions that contain walls.

**WHAT ARE WALLS (include these):**
- Thick black/dark lines forming room boundaries
- Exterior building outline/perimeter
- Interior room dividers
- Include the wall line even through door/window openings

**WHAT ARE NOT WALLS (exclude these):**
- Furniture (beds, sofas, tables, chairs, etc.)
- Appliances (refrigerators, stoves, washing machines)
- Fixtures (toilets, sinks, bathtubs)
- Text, labels, dimensions, annotations
- Thin lines, arrows, symbols
- Stairs, elevators
- Cars, vehicles
- Landscaping, outdoor features

**OUTPUT FORMAT:**
Return a JSON object with wall regions. Each region is a rectangle containing wall pixels.
For best results, trace along walls creating thin rectangular strips.

{
  "image_width": <int>,
  "image_height": <int>,
  "wall_regions": [
    {"x": <int>, "y": <int>, "width": <int>, "height": <int>},
    ...
  ],
  "wall_lines": [
    {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>},
    ...
  ]
}

For wall_lines, trace the CENTER LINE of each visible wall segment.
Coordinates should be as precise as possible.

Return ONLY valid JSON, no other text."""


class WallGraph:
    """Graph representation of wall network for cleanup algorithms."""

    def __init__(self):
        self.nodes: Dict[Tuple[int, int], Set[int]] = defaultdict(set)  # point -> connected wall indices
        self.walls: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)

    def add_wall(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Add a wall segment to the graph."""
        wall_idx = len(self.walls)
        self.walls.append((x1, y1, x2, y2))
        self.nodes[(x1, y1)].add(wall_idx)
        self.nodes[(x2, y2)].add(wall_idx)
        return wall_idx

    def get_wall_endpoints(self, wall_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get endpoints of a wall."""
        x1, y1, x2, y2 = self.walls[wall_idx]
        return (x1, y1), (x2, y2)

    def find_nearby_nodes(self, point: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """Find nodes within radius of a point."""
        nearby = []
        px, py = point
        for node in self.nodes.keys():
            nx, ny = node
            dist = math.sqrt((nx - px)**2 + (ny - py)**2)
            if dist <= radius and node != point:
                nearby.append(node)
        return nearby

    def merge_nearby_nodes(self, radius: int = 10) -> 'WallGraph':
        """Merge nodes that are within radius pixels of each other."""
        # Build clusters of nearby nodes
        clusters = []
        used = set()

        for node in self.nodes.keys():
            if node in used:
                continue

            cluster = {node}
            used.add(node)

            # Find all nearby nodes
            to_check = [node]
            while to_check:
                current = to_check.pop()
                nearby = self.find_nearby_nodes(current, radius)
                for n in nearby:
                    if n not in used:
                        cluster.add(n)
                        used.add(n)
                        to_check.append(n)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Create new graph with merged nodes
        new_graph = WallGraph()

        # Build mapping from old nodes to new nodes (cluster centroids)
        node_mapping = {}
        for cluster in clusters:
            # Calculate centroid
            cx = int(sum(n[0] for n in cluster) / len(cluster))
            cy = int(sum(n[1] for n in cluster) / len(cluster))
            centroid = (cx, cy)
            for node in cluster:
                node_mapping[node] = centroid

        # Add walls with remapped endpoints
        seen_walls = set()
        for x1, y1, x2, y2 in self.walls:
            # Remap endpoints
            p1 = node_mapping.get((x1, y1), (x1, y1))
            p2 = node_mapping.get((x2, y2), (x2, y2))

            # Skip degenerate walls
            if p1 == p2:
                continue

            # Normalize direction for deduplication
            if p1 > p2:
                p1, p2 = p2, p1

            wall_key = (p1[0], p1[1], p2[0], p2[1])
            if wall_key not in seen_walls:
                seen_walls.add(wall_key)
                new_graph.add_wall(p1[0], p1[1], p2[0], p2[1])

        return new_graph

    def get_walls(self) -> List[Tuple[int, int, int, int]]:
        """Get all walls."""
        return self.walls.copy()


class HybridWallDetector:
    """
    Hybrid AI + Graph Algorithm wall detector.

    Uses AI for semantic understanding (what IS a wall) and
    CV/graph algorithms for precise coordinate extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the hybrid wall detector."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Model selection from config
        model_choice = getattr(settings, 'AI_WALL_DETECTION_MODEL', 'opus').lower()
        if model_choice == "opus":
            self.model = "claude-opus-4-20250514"
            logger.info("Hybrid detector using Claude Opus for maximum accuracy")
        else:
            self.model = "claude-sonnet-4-20250514"
            logger.info("Hybrid detector using Claude Sonnet for speed/cost balance")

        self.scale = 0.05

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls using hybrid AI + graph algorithm approach.

        Steps:
        1. AI identifies wall regions (semantic segmentation)
        2. Create binary wall mask from AI output
        3. Skeletonize to get centerlines
        4. Hough Transform to detect line segments
        5. Build wall graph and apply cleanup algorithms

        Args:
            image: Input BGR or grayscale image
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale
        height, width = image.shape[:2]

        logger.info(f"Starting hybrid wall detection on {width}x{height} image")

        # Step 1: Get AI wall analysis
        logger.info("Step 1: Getting AI wall analysis...")
        ai_walls, wall_mask = self._get_ai_wall_mask(image)

        if wall_mask is not None:
            # Step 2: Skeletonize the mask
            logger.info("Step 2: Skeletonizing wall mask...")
            skeleton = self._skeletonize(wall_mask)

            # Step 3: Detect lines from skeleton
            logger.info("Step 3: Detecting lines from skeleton...")
            skeleton_walls = self._detect_lines_from_skeleton(skeleton)
            logger.info(f"  Detected {len(skeleton_walls)} lines from skeleton")

            # Combine AI direct walls with skeleton-detected walls
            all_walls = ai_walls + skeleton_walls
        else:
            all_walls = ai_walls

        if not all_walls:
            logger.warning("No walls detected, falling back to pure CV")
            return self._fallback_cv_detection(image)

        logger.info(f"Total raw walls: {len(all_walls)}")

        # Step 4: Build wall graph
        logger.info("Step 4: Building wall graph...")
        graph = WallGraph()
        for x1, y1, x2, y2 in all_walls:
            # Snap to grid
            grid = 5
            x1 = round(x1 / grid) * grid
            y1 = round(y1 / grid) * grid
            x2 = round(x2 / grid) * grid
            y2 = round(y2 / grid) * grid

            # Snap nearly H/V lines
            x1, y1, x2, y2 = self._snap_to_axis(x1, y1, x2, y2)

            # Skip very short walls
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 15:
                continue

            graph.add_wall(x1, y1, x2, y2)

        # Step 5: Apply graph cleanup algorithms
        logger.info("Step 5: Applying graph cleanup algorithms...")

        # Merge nearby nodes (snap corners together)
        graph = graph.merge_nearby_nodes(radius=15)
        logger.info(f"  After node merging: {len(graph.walls)} walls")

        # Merge collinear walls
        walls = self._merge_collinear_walls(graph.get_walls())
        logger.info(f"  After collinear merge: {len(walls)} walls")

        # Remove duplicates
        walls = self._remove_duplicates(walls)
        logger.info(f"  After duplicate removal: {len(walls)} walls")

        # Convert to WallSegment objects
        wall_segments = []
        for x1, y1, x2, y2 in walls:
            segment = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            wall_segments.append(segment)

        logger.info(f"Hybrid detection complete: {len(wall_segments)} walls")
        return wall_segments, []

    def _get_ai_wall_mask(
        self,
        image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Get wall mask from AI analysis.

        Returns:
            Tuple of (direct_wall_lines, binary_mask)
        """
        height, width = image.shape[:2]

        # Convert image to base64
        image_b64 = self._image_to_base64(image)

        try:
            # Call Claude Vision API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": WALL_SEGMENTATION_PROMPT
                            }
                        ]
                    }
                ]
            )

            response = message.content[0].text

            # Parse response
            data = self._parse_json_response(response)

            # Get scale factors if image was resized
            ai_width = data.get("image_width", width)
            ai_height = data.get("image_height", height)
            scale_x = width / ai_width if ai_width else 1.0
            scale_y = height / ai_height if ai_height else 1.0

            # Create binary mask from wall regions
            wall_mask = np.zeros((height, width), dtype=np.uint8)

            for region in data.get("wall_regions", []):
                x = int(region.get("x", 0) * scale_x)
                y = int(region.get("y", 0) * scale_y)
                w = int(region.get("width", 0) * scale_x)
                h = int(region.get("height", 0) * scale_y)

                # Clamp to image bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)

                if w > 0 and h > 0:
                    wall_mask[y:y+h, x:x+w] = 255

            # Extract direct wall lines from AI
            direct_walls = []
            for line in data.get("wall_lines", []):
                try:
                    x1 = int(float(line.get("x1", 0)) * scale_x)
                    y1 = int(float(line.get("y1", 0)) * scale_y)
                    x2 = int(float(line.get("x2", 0)) * scale_x)
                    y2 = int(float(line.get("y2", 0)) * scale_y)
                    direct_walls.append((x1, y1, x2, y2))
                except (KeyError, TypeError, ValueError):
                    continue

            logger.info(f"AI returned {len(data.get('wall_regions', []))} regions and {len(direct_walls)} lines")

            # If we have regions but not many direct lines, use the mask
            if np.sum(wall_mask) > 0:
                return direct_walls, wall_mask
            else:
                return direct_walls, None

        except Exception as e:
            logger.error(f"AI wall analysis failed: {e}")
            return [], None

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        Skeletonize a binary mask to get 1-pixel-wide centerlines.
        """
        # Ensure binary mask
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Use cv2.ximgproc.thinning if available, otherwise morphological skeleton
        if hasattr(cv2, 'ximgproc'):
            skeleton = cv2.ximgproc.thinning(binary)
        else:
            # Morphological skeletonization
            skeleton = np.zeros_like(binary)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            temp = binary.copy()
            while True:
                eroded = cv2.erode(temp, element)
                opened = cv2.dilate(eroded, element)
                temp_skeleton = cv2.subtract(temp, opened)
                skeleton = cv2.bitwise_or(skeleton, temp_skeleton)
                temp = eroded.copy()

                if cv2.countNonZero(temp) == 0:
                    break

        return skeleton

    def _detect_lines_from_skeleton(
        self,
        skeleton: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect line segments from a skeleton image using Hough Transform.
        """
        # Apply Hough transform on skeleton
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=15,
            maxLineGap=10
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            walls.append((int(x1), int(y1), int(x2), int(y2)))

        return walls

    def _snap_to_axis(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        threshold_degrees: float = 5.0
    ) -> Tuple[int, int, int, int]:
        """Snap nearly horizontal/vertical lines to exact H/V."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx == 0 and dy == 0:
            return x1, y1, x2, y2

        angle = math.degrees(math.atan2(dy, dx))

        # Nearly horizontal
        if angle < threshold_degrees:
            avg_y = (y1 + y2) // 2
            return x1, avg_y, x2, avg_y

        # Nearly vertical
        if angle > (90 - threshold_degrees):
            avg_x = (x1 + x2) // 2
            return avg_x, y1, avg_x, y2

        return x1, y1, x2, y2

    def _merge_collinear_walls(
        self,
        walls: List[Tuple[int, int, int, int]],
        angle_threshold: float = 5.0,
        distance_threshold: int = 20
    ) -> List[Tuple[int, int, int, int]]:
        """
        Merge collinear wall segments that are close together.
        """
        if len(walls) < 2:
            return walls

        merged = []
        used = set()

        for i, wall1 in enumerate(walls):
            if i in used:
                continue

            # Start a group with this wall
            group = [wall1]
            used.add(i)

            for j, wall2 in enumerate(walls):
                if j in used or j <= i:
                    continue

                if self._are_collinear(wall1, wall2, angle_threshold, distance_threshold):
                    group.append(wall2)
                    used.add(j)

            # Merge the group into a single wall
            merged_wall = self._merge_wall_group(group)
            merged.append(merged_wall)

        return merged

    def _are_collinear(
        self,
        wall1: Tuple[int, int, int, int],
        wall2: Tuple[int, int, int, int],
        angle_threshold: float,
        distance_threshold: int
    ) -> bool:
        """Check if two walls are collinear and close enough to merge."""
        x1_1, y1_1, x2_1, y2_1 = wall1
        x1_2, y1_2, x2_2, y2_2 = wall2

        # Get direction vectors
        d1x = x2_1 - x1_1
        d1y = y2_1 - y1_1
        d2x = x2_2 - x1_2
        d2y = y2_2 - y1_2

        len1 = math.sqrt(d1x**2 + d1y**2)
        len2 = math.sqrt(d2x**2 + d2y**2)

        if len1 < 1 or len2 < 1:
            return False

        # Normalize
        d1x, d1y = d1x / len1, d1y / len1
        d2x, d2y = d2x / len2, d2y / len2

        # Check angle (dot product)
        dot = abs(d1x * d2x + d1y * d2y)
        if dot < math.cos(math.radians(angle_threshold)):
            return False

        # Check distance between closest endpoints
        endpoints1 = [(x1_1, y1_1), (x2_1, y2_1)]
        endpoints2 = [(x1_2, y1_2), (x2_2, y2_2)]

        min_dist = float('inf')
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_dist = min(min_dist, dist)

        return min_dist < distance_threshold

    def _merge_wall_group(
        self,
        walls: List[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """Merge a group of collinear walls into a single wall."""
        if len(walls) == 1:
            return walls[0]

        # Collect all endpoints
        points = []
        for x1, y1, x2, y2 in walls:
            points.append((x1, y1))
            points.append((x2, y2))

        # Find the two points farthest apart
        max_dist = 0
        best_p1, best_p2 = points[0], points[1]

        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i >= j:
                    continue
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_p1, best_p2 = p1, p2

        return (best_p1[0], best_p1[1], best_p2[0], best_p2[1])

    def _remove_duplicates(
        self,
        walls: List[Tuple[int, int, int, int]],
        threshold: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """Remove duplicate wall segments."""
        if len(walls) < 2:
            return walls

        unique = []

        for wall in walls:
            x1, y1, x2, y2 = wall

            # Normalize direction
            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Check if similar wall exists
            is_dup = False
            for ux1, uy1, ux2, uy2 in unique:
                d1 = math.sqrt((x1 - ux1)**2 + (y1 - uy1)**2)
                d2 = math.sqrt((x2 - ux2)**2 + (y2 - uy2)**2)

                if d1 < threshold and d2 < threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append((x1, y1, x2, y2))

        return unique

    def _fallback_cv_detection(
        self,
        image: np.ndarray
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Fallback to pure CV detection if AI fails.
        """
        logger.info("Using fallback CV wall detection")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Preprocess
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Edge detection
        edges = cv2.Canny(binary, 50, 150)

        # Hough lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=25,
            maxLineGap=15
        )

        if lines is None:
            return [], []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Snap to axis
            x1, y1, x2, y2 = self._snap_to_axis(x1, y1, x2, y2)

            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 15:
                continue

            wall = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            walls.append(wall)

        return walls, []

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        max_dimension = 2000
        height, width = image.shape[:2]

        if max(width, height) > max_dimension:
            scale_factor = max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height))

        _, buffer = cv2.imencode('.png', image)
        return base64.standard_b64encode(buffer).decode('utf-8')

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from AI response, handling markdown code blocks."""
        import re

        json_str = response.strip()

        # Extract JSON from code blocks
        if "```json" in json_str or "```" in json_str:
            match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', json_str)
            if match:
                json_str = match.group(1).strip()
            else:
                match = re.search(r'\{[\s\S]*\}', json_str)
                if match:
                    json_str = match.group(0)
        elif not json_str.startswith("{"):
            match = re.search(r'\{[\s\S]*\}', json_str)
            if match:
                json_str = match.group(0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {response[:500]}...")
            return {"wall_regions": [], "wall_lines": []}

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


def detect_walls_hybrid(
    image_path: str,
    scale: float = 0.05,
    api_key: Optional[str] = None
) -> Tuple[List[WallSegment], List[Room]]:
    """
    Convenience function to detect walls using hybrid approach.

    Args:
        image_path: Path to floor plan image
        scale: Meters per pixel
        api_key: Optional Anthropic API key

    Returns:
        Tuple of (walls, rooms)
    """
    detector = HybridWallDetector(api_key=api_key)
    return detector.detect_walls_from_file(image_path, scale)
