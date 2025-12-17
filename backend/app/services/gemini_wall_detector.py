"""Gemini-based wall detection for floor plans.

Uses Google's Gemini Vision API to analyze floor plan images and extract
wall coordinates with high accuracy.
"""

import os
import json
import base64
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

# Credentials path
CREDENTIALS_PATH = os.environ.get(
    'GOOGLE_APPLICATION_CREDENTIALS',
    '/tmp/gemini-credentials.json'
)

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


class GeminiWallDetector:
    """
    Gemini Vision-based wall detection for floor plans.

    Uses Gemini's multimodal capabilities to analyze floor plan images
    and extract precise wall coordinates.
    """

    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        self.model = None
        self.scale = 0.05

        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client - prefer API key over Vertex AI."""
        # Try standard Gemini API first (simpler, more reliable)
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                # Use gemini-2.0-flash (gemini-1.5-flash is deprecated)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.use_vertex = False
                logger.info("Standard Gemini API initialized with API key (gemini-2.0-flash)")
                return
            except Exception as e:
                logger.warning(f"Standard Gemini API failed: {e}")

        # Fall back to Vertex AI if API key not available
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path

            with open(self.credentials_path, 'r') as f:
                creds = json.load(f)
            self.project_id = creds.get('project_id', 'mmodel-462701')

            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project_id, location='us-central1')
            self.model = GenerativeModel('gemini-1.5-flash-001')
            self.use_vertex = True
            logger.info(f"Vertex AI Gemini initialized for project {self.project_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise ValueError(f"Could not initialize Gemini: {e}")

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls in a floor plan image using Gemini Vision.

        Args:
            image: Input BGR image (numpy array)
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale
        height, width = image.shape[:2]

        # Convert image to base64
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Create the prompt for wall detection
        prompt = f"""Analyze this floor plan image and extract ALL wall segments.

IMAGE DIMENSIONS: {width} x {height} pixels

TASK: Identify every wall in the floor plan and return their pixel coordinates.

IMPORTANT INSTRUCTIONS:
1. A wall is any solid line that forms the boundary of rooms or the building perimeter
2. Include BOTH exterior walls (building perimeter) AND interior walls (room dividers)
3. Walls are typically drawn as thick black or dark lines
4. DO NOT include: furniture, doors (the door itself, though include the wall around door openings), windows, text labels, dimension lines, or any annotations
5. For each wall, identify the start point (x1, y1) and end point (x2, y2) in PIXEL coordinates
6. Coordinates should be relative to top-left corner (0,0)
7. Be PRECISE - trace the CENTER LINE of each wall

Return ONLY a JSON array with this exact format, no other text:
[
  {{"x1": 100, "y1": 50, "x2": 500, "y2": 50, "type": "exterior"}},
  {{"x1": 100, "y1": 50, "x2": 100, "y2": 400, "type": "interior"}},
  ...
]

Analyze the floor plan carefully and return ALL walls as JSON."""

        try:
            # Call Gemini API (different format for Vertex AI vs standard API)
            if self.use_vertex:
                from vertexai.generative_models import Part, Image as VertexImage

                # Create image part from bytes
                image_bytes = buffer.tobytes()
                image_part = Part.from_data(image_bytes, mime_type='image/png')

                response = self.model.generate_content([prompt, image_part])
            else:
                response = self.model.generate_content([
                    prompt,
                    {
                        'mime_type': 'image/png',
                        'data': image_base64
                    }
                ])

            # Parse the response
            response_text = response.text.strip()
            logger.info(f"Gemini response length: {len(response_text)}")
            logger.debug(f"Gemini response: {response_text[:500]}")

            # Extract JSON from response
            walls = self._parse_wall_response(response_text, width, height)

            logger.info(f"Gemini detected {len(walls)} walls")

            return walls, []

        except Exception as e:
            logger.error(f"Gemini wall detection failed: {e}")
            raise

    def _parse_wall_response(
        self,
        response_text: str,
        width: int,
        height: int
    ) -> List[WallSegment]:
        """Parse Gemini's response into WallSegment objects."""
        walls = []

        try:
            # Try to find JSON array in the response
            # Sometimes Gemini wraps it in markdown code blocks
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                json_str = json_match.group()
                wall_data = json.loads(json_str)
            else:
                logger.warning(f"Could not find JSON in response: {response_text[:500]}")
                return walls

            for item in wall_data:
                try:
                    x1 = float(item.get('x1', 0))
                    y1 = float(item.get('y1', 0))
                    x2 = float(item.get('x2', 0))
                    y2 = float(item.get('y2', 0))
                    wall_type = item.get('type', 'interior')

                    # Validate coordinates are within image bounds
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # Skip zero-length walls
                    length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    if length < 10:  # Skip very short segments
                        continue

                    # Determine material based on wall type
                    if wall_type == 'exterior':
                        material = 'concrete'
                        thickness = 0.3
                    else:
                        material = 'drywall'
                        thickness = 0.15

                    walls.append(WallSegment(
                        start=Point(x=x1, y=y1),
                        end=Point(x=x2, y=y2),
                        thickness=thickness,
                        material=material,
                        attenuation_db=MATERIAL_ATTENUATION[material]
                    ))

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid wall data: {item}, error: {e}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response_text[:1000]}")

        # Post-process: snap to grid and connect endpoints
        walls = self._snap_to_grid(walls)
        walls = self._connect_endpoints(walls)

        return walls

    def _snap_to_grid(self, walls: List[WallSegment], grid_size: int = 5) -> List[WallSegment]:
        """Snap wall endpoints to a grid for cleaner results."""
        snapped = []
        for wall in walls:
            x1 = round(wall.start.x / grid_size) * grid_size
            y1 = round(wall.start.y / grid_size) * grid_size
            x2 = round(wall.end.x / grid_size) * grid_size
            y2 = round(wall.end.y / grid_size) * grid_size

            # Also snap to H/V if close
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dy < 10 and dx > dy:  # Nearly horizontal
                avg_y = (y1 + y2) / 2
                y1 = y2 = round(avg_y / grid_size) * grid_size
            elif dx < 10 and dy > dx:  # Nearly vertical
                avg_x = (x1 + x2) / 2
                x1 = x2 = round(avg_x / grid_size) * grid_size

            snapped.append(WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=wall.thickness,
                material=wall.material,
                attenuation_db=wall.attenuation_db
            ))

        return snapped

    def _connect_endpoints(
        self,
        walls: List[WallSegment],
        threshold: float = 15.0
    ) -> List[WallSegment]:
        """Connect nearby wall endpoints."""
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

        # Snap endpoints to cluster centroids
        walls_list = list(walls)
        for cluster in clusters:
            avg_x = sum(p[2] for p in cluster) / len(cluster)
            avg_y = sum(p[3] for p in cluster) / len(cluster)

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


# Check availability
try:
    import google.generativeai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
