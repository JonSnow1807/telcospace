"""AI Vision-based wall detection using Claude API.

Uses Claude's vision capabilities to accurately identify walls in floor plans.
Much more accurate than traditional CV approaches as the model understands
the semantic meaning of floor plan elements.
"""

import os
import base64
import json
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import anthropic
import cv2
import numpy as np

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

WALL_DETECTION_PROMPT = """You are a CAD digitization expert. Convert this floor plan image into precise wall coordinates.

**YOUR TASK:** Extract EVERY wall as a line segment with EXACT pixel coordinates.

**DETECTION ORDER - FOLLOW THIS EXACTLY:**

**STEP 1: OUTER BOUNDARY (PERIMETER) - DO THIS FIRST!**
- Trace the COMPLETE outer edge of the floor plan
- Start at one corner and go around the entire perimeter
- The outer boundary forms a closed polygon (may be rectangular, L-shaped, or irregular)
- Every corner of the building exterior must have connecting walls
- This is the most important step - do not skip any exterior walls!

**STEP 2: INTERIOR WALLS**
- Walls that divide the interior into rooms
- Must connect to the outer boundary or other interior walls
- Include walls between: bedrooms, bathrooms, kitchen, living room, hallways, closets

**CRITICAL RULES FOR CAD-QUALITY OUTPUT:**

1. **TRACE THE CENTER LINE** of each wall (middle of the thick black/dark line)

2. **WALLS MUST CONNECT PRECISELY:**
   - Where walls meet at corners, endpoints MUST have IDENTICAL coordinates
   - T-junctions: perpendicular wall endpoint must land EXACTLY on the other wall
   - No gaps between connected walls

3. **SNAP TO GRID (round to nearest 5 pixels):**
   - All coordinates should be multiples of 5 for clean CAD output
   - Example: 127 → 125, 283 → 285

4. **HORIZONTAL/VERTICAL ALIGNMENT:**
   - If a wall is nearly horizontal (within 3°), make y1 = y2 EXACTLY
   - If a wall is nearly vertical (within 3°), make x1 = x2 EXACTLY

5. **WHAT IS A WALL:**
   - Thick black/dark lines forming room boundaries
   - EXTERIOR BUILDING OUTLINE (the perimeter - most important!)
   - Interior room dividers
   - Include wall even where there are door/window openings

6. **IGNORE COMPLETELY:**
   - Furniture, appliances, fixtures, cars
   - Text, labels, dimensions, annotations
   - Thin lines, arrows, symbols
   - Anything not a structural wall

**OUTPUT FORMAT (valid JSON only, no other text):**
{
  "image_width": <int>,
  "image_height": <int>,
  "walls": [
    {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>},
    {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}
  ]
}

**EXAMPLE - L-shaped apartment with outer boundary + interior:**
Outer boundary (must form closed loop):
- {"x1": 20, "y1": 20, "x2": 300, "y2": 20}    // top
- {"x1": 300, "y1": 20, "x2": 300, "y2": 200}  // right upper
- {"x1": 300, "y1": 200, "x2": 400, "y2": 200} // step out
- {"x1": 400, "y1": 200, "x2": 400, "y2": 400} // right lower
- {"x1": 400, "y1": 400, "x2": 20, "y2": 400}  // bottom
- {"x1": 20, "y1": 400, "x2": 20, "y2": 20}    // left (closes loop)

Interior walls:
- {"x1": 150, "y1": 20, "x2": 150, "y2": 200}  // bedroom divider
- {"x1": 20, "y1": 200, "x2": 300, "y2": 200}  // horizontal divider

Analyze the floor plan. First trace the COMPLETE outer boundary, then all interior walls. Return as JSON."""


class AIWallDetector:
    """
    AI-powered wall detection using Claude Vision API.

    This provides much higher accuracy than traditional CV methods
    by leveraging Claude's ability to understand floor plan semantics.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI wall detector.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Model selection from config: "opus" for max accuracy, "sonnet" for speed/cost
        model_choice = getattr(settings, 'AI_WALL_DETECTION_MODEL', 'opus').lower()
        if model_choice == "opus":
            self.model = "claude-opus-4-20250514"
            logger.info("Using Claude Opus for maximum wall detection accuracy")
        else:
            self.model = "claude-sonnet-4-20250514"
            logger.info("Using Claude Sonnet for faster/cheaper wall detection")

        self.scale = 0.05  # Default scale, will be updated
        self.detected_scale = None  # Scale detected from dimension annotations
        self.scale_method = None  # How scale was determined

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls in a floor plan image using AI vision + CV refinement.

        Uses hybrid approach:
        1. AI identifies semantic wall regions (ignores furniture, text, cars)
        2. CV snaps walls to exact pixel edges for perfect alignment

        Args:
            image: Input BGR or grayscale image
            scale: Meters per pixel for thickness conversion

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale

        # Store original image for CV refinement
        self.original_image = image.copy()

        # Convert image to base64
        image_b64 = self._image_to_base64(image)

        # Get image dimensions
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        # Call Claude Vision API
        try:
            response = self._call_vision_api(image_b64)
            walls, rooms = self._parse_response(response, width, height)

            logger.info(f"AI detected {len(walls)} walls and {len(rooms)} rooms")

            # Skip CV refinement - it was snapping to wrong edges (text, furniture, etc.)
            # The AI's semantic understanding gives more accurate wall positions
            # than trying to snap to any detected edge

            return walls, rooms

        except Exception as e:
            logger.error(f"AI wall detection failed: {e}")
            raise

    def _refine_walls_with_cv(
        self,
        walls: List[WallSegment],
        image: np.ndarray
    ) -> List[WallSegment]:
        """
        Refine AI-detected wall positions using CV edge detection.

        This snaps wall endpoints to actual edges in the image for
        pixel-perfect alignment.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect edges in the image
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges slightly to make snapping easier
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        refined_walls = []
        for wall in walls:
            # Refine start point
            start_x, start_y = self._snap_to_edge(
                int(wall.start.x), int(wall.start.y),
                edges_dilated, search_radius=15
            )

            # Refine end point
            end_x, end_y = self._snap_to_edge(
                int(wall.end.x), int(wall.end.y),
                edges_dilated, search_radius=15
            )

            # Snap to horizontal/vertical if close
            start_x, start_y, end_x, end_y = self._snap_to_axis(
                start_x, start_y, end_x, end_y, threshold=5
            )

            refined_wall = WallSegment(
                start=Point(x=float(start_x), y=float(start_y)),
                end=Point(x=float(end_x), y=float(end_y)),
                thickness=wall.thickness,
                material=wall.material,
                attenuation_db=wall.attenuation_db
            )
            refined_walls.append(refined_wall)

        # Connect nearby wall endpoints (snap corners together)
        refined_walls = self._connect_wall_endpoints(refined_walls, threshold=10)

        return refined_walls

    def _snap_to_edge(
        self,
        x: int,
        y: int,
        edges: np.ndarray,
        search_radius: int = 10
    ) -> Tuple[int, int]:
        """
        Snap a point to the nearest edge pixel within search radius.
        """
        height, width = edges.shape

        # Search in a square region around the point
        best_x, best_y = x, y
        best_dist = float('inf')

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy

                # Check bounds
                if 0 <= nx < width and 0 <= ny < height:
                    # Check if this is an edge pixel
                    if edges[ny, nx] > 0:
                        dist = dx*dx + dy*dy
                        if dist < best_dist:
                            best_dist = dist
                            best_x, best_y = nx, ny

        return best_x, best_y

    def _snap_to_axis(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        threshold: int = 5
    ) -> Tuple[int, int, int, int]:
        """
        Snap nearly horizontal/vertical lines to exact H/V.
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Nearly horizontal
        if dy < threshold and dx > dy:
            avg_y = (y1 + y2) // 2
            return x1, avg_y, x2, avg_y

        # Nearly vertical
        if dx < threshold and dy > dx:
            avg_x = (x1 + x2) // 2
            return avg_x, y1, avg_x, y2

        return x1, y1, x2, y2

    def _connect_wall_endpoints(
        self,
        walls: List[WallSegment],
        threshold: int = 10
    ) -> List[WallSegment]:
        """
        Connect nearby wall endpoints to form proper corners.
        """
        if len(walls) < 2:
            return walls

        # Collect all endpoints
        endpoints = []
        for i, wall in enumerate(walls):
            endpoints.append((i, 'start', wall.start.x, wall.start.y))
            endpoints.append((i, 'end', wall.end.x, wall.end.y))

        # Find clusters of nearby endpoints
        clusters = []
        used = set()

        for i, (wall_idx, end_type, x, y) in enumerate(endpoints):
            if i in used:
                continue

            cluster = [(wall_idx, end_type, x, y)]
            used.add(i)

            for j, (w2, e2, x2, y2) in enumerate(endpoints):
                if j in used:
                    continue

                dist = ((x - x2)**2 + (y - y2)**2)**0.5
                if dist < threshold:
                    cluster.append((w2, e2, x2, y2))
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Snap clustered endpoints to their centroid
        walls_list = list(walls)
        for cluster in clusters:
            # Calculate centroid
            avg_x = sum(p[2] for p in cluster) / len(cluster)
            avg_y = sum(p[3] for p in cluster) / len(cluster)

            # Update all endpoints in cluster
            for wall_idx, end_type, _, _ in cluster:
                wall = walls_list[wall_idx]
                if end_type == 'start':
                    walls_list[wall_idx] = WallSegment(
                        start=Point(x=avg_x, y=avg_y),
                        end=wall.end,
                        thickness=wall.thickness,
                        material=wall.material,
                        attenuation_db=wall.attenuation_db
                    )
                else:
                    walls_list[wall_idx] = WallSegment(
                        start=wall.start,
                        end=Point(x=avg_x, y=avg_y),
                        thickness=wall.thickness,
                        material=wall.material,
                        attenuation_db=wall.attenuation_db
                    )

        return walls_list

    def detect_walls_from_file(
        self,
        file_path: str,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls from an image file.

        Args:
            file_path: Path to the floor plan image
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")

        return self.detect_walls(image, scale)

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        # Resize if too large (Claude has limits)
        max_dimension = 2000
        height, width = image.shape[:2]

        if max(width, height) > max_dimension:
            scale_factor = max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        # Encode as PNG
        _, buffer = cv2.imencode('.png', image)
        return base64.standard_b64encode(buffer).decode('utf-8')

    def _call_vision_api(self, image_b64: str) -> str:
        """Call Claude Vision API with the floor plan image."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=8192,  # Increased for complex floor plans with many walls
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
                            "text": WALL_DETECTION_PROMPT
                        }
                    ]
                }
            ]
        )

        return message.content[0].text

    def _parse_response(
        self,
        response: str,
        expected_width: int,
        expected_height: int
    ) -> Tuple[List[WallSegment], List[Room]]:
        """Parse Claude's JSON response into wall segments with CAD-quality processing."""
        # Extract JSON from response (handle markdown code blocks and preamble text)
        json_str = response.strip()

        # If response contains markdown code block, extract the JSON from it
        if "```json" in json_str or "```" in json_str:
            import re
            # Find JSON inside code blocks
            match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', json_str)
            if match:
                json_str = match.group(1).strip()
            else:
                # Fallback: try to find raw JSON object
                match = re.search(r'\{[\s\S]*\}', json_str)
                if match:
                    json_str = match.group(0)
        elif not json_str.startswith("{"):
            # No code block but text before JSON - find the JSON object
            import re
            match = re.search(r'\{[\s\S]*\}', json_str)
            if match:
                json_str = match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            raise ValueError(f"AI returned invalid JSON: {e}")

        # Get scale factor if image was resized
        ai_width = data.get("image_width", expected_width)
        ai_height = data.get("image_height", expected_height)

        scale_x = expected_width / ai_width if ai_width else 1.0
        scale_y = expected_height / ai_height if ai_height else 1.0

        # Parse walls with simplified format (x1, y1, x2, y2)
        raw_walls = []
        for wall_data in data.get("walls", []):
            try:
                # Handle both old format (start_x/end_x) and new format (x1/x2)
                if "x1" in wall_data:
                    x1 = float(wall_data["x1"]) * scale_x
                    y1 = float(wall_data["y1"]) * scale_y
                    x2 = float(wall_data["x2"]) * scale_x
                    y2 = float(wall_data["y2"]) * scale_y
                else:
                    x1 = float(wall_data["start_x"]) * scale_x
                    y1 = float(wall_data["start_y"]) * scale_y
                    x2 = float(wall_data["end_x"]) * scale_x
                    y2 = float(wall_data["end_y"]) * scale_y

                raw_walls.append((x1, y1, x2, y2))

            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping invalid wall data: {e}")
                continue

        logger.info(f"Parsed {len(raw_walls)} raw walls from AI response")

        # Apply CAD post-processing for precision
        processed_walls = self._cad_post_process(raw_walls)
        logger.info(f"After CAD processing: {len(processed_walls)} walls")

        # Convert to WallSegment objects
        walls = []
        for x1, y1, x2, y2 in processed_walls:
            wall = WallSegment(
                start=Point(x=x1, y=y1),
                end=Point(x=x2, y=y2),
                thickness=0.2,  # Default 20cm wall
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            walls.append(wall)

        return walls, []  # No rooms in simplified format

    def _cad_post_process(self, walls: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Apply CAD-quality post-processing to walls."""
        if not walls:
            return walls

        # Step 1: Snap to grid (5 pixel grid)
        grid_size = 5
        walls = [
            (
                round(x1 / grid_size) * grid_size,
                round(y1 / grid_size) * grid_size,
                round(x2 / grid_size) * grid_size,
                round(y2 / grid_size) * grid_size
            )
            for x1, y1, x2, y2 in walls
        ]

        # Step 2: Snap nearly horizontal/vertical walls
        threshold = 10  # pixels
        aligned_walls = []
        for x1, y1, x2, y2 in walls:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dy < threshold and dx > dy:  # Nearly horizontal
                avg_y = round((y1 + y2) / 2 / grid_size) * grid_size
                y1 = y2 = avg_y
            elif dx < threshold and dy > dx:  # Nearly vertical
                avg_x = round((x1 + x2) / 2 / grid_size) * grid_size
                x1 = x2 = avg_x

            aligned_walls.append((x1, y1, x2, y2))
        walls = aligned_walls

        # Step 3: Connect nearby endpoints (snap corners together)
        endpoint_threshold = 15  # pixels
        endpoints = []
        for i, (x1, y1, x2, y2) in enumerate(walls):
            endpoints.append((i, 'start', x1, y1))
            endpoints.append((i, 'end', x2, y2))

        # Find clusters
        used = set()
        clusters = []
        for i, (wall_idx, end_type, x, y) in enumerate(endpoints):
            if i in used:
                continue
            cluster = [(wall_idx, end_type, x, y)]
            used.add(i)
            for j, (w2, e2, x2, y2) in enumerate(endpoints):
                if j in used:
                    continue
                dist = ((x - x2)**2 + (y - y2)**2)**0.5
                if dist < endpoint_threshold:
                    cluster.append((w2, e2, x2, y2))
                    used.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        # Snap clusters to centroid
        walls = list(walls)
        for cluster in clusters:
            avg_x = round(sum(p[2] for p in cluster) / len(cluster) / grid_size) * grid_size
            avg_y = round(sum(p[3] for p in cluster) / len(cluster) / grid_size) * grid_size

            for wall_idx, end_type, _, _ in cluster:
                x1, y1, x2, y2 = walls[wall_idx]
                if end_type == 'start':
                    walls[wall_idx] = (avg_x, avg_y, x2, y2)
                else:
                    walls[wall_idx] = (x1, y1, avg_x, avg_y)

        # Step 4: Remove duplicate/very short walls
        final_walls = []
        seen = set()
        for x1, y1, x2, y2 in walls:
            # Skip very short walls
            length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            if length < 10:
                continue

            # Normalize direction (smaller point first)
            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            key = (x1, y1, x2, y2)
            if key not in seen:
                seen.add(key)
                final_walls.append((x1, y1, x2, y2))

        return final_walls

    def _calculate_polygon_area(self, polygon: List[List[float]]) -> float:
        """Calculate area of a polygon using the shoelace formula."""
        n = len(polygon)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]

        return abs(area) / 2.0


def detect_walls_with_ai(
    image_path: str,
    scale: float = 0.05,
    api_key: Optional[str] = None
) -> Tuple[List[WallSegment], List[Room]]:
    """
    Convenience function to detect walls using AI.

    Args:
        image_path: Path to floor plan image
        scale: Meters per pixel
        api_key: Optional Anthropic API key

    Returns:
        Tuple of (walls, rooms)
    """
    detector = AIWallDetector(api_key=api_key)
    return detector.detect_walls_from_file(image_path, scale)
