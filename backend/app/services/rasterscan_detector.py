"""Wall detection using Rasterscan HuggingFace Space API.

This uses the free Rasterscan model hosted on HuggingFace Spaces
for accurate floor plan wall detection and room segmentation.
"""

import logging
import tempfile
import os
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

# Default material attenuation for WiFi signal
MATERIAL_ATTENUATION = {'concrete': 15.0, 'drywall': 3.0, 'default': 8.0}

# HuggingFace Space URL
RASTERSCAN_SPACE = "RasterScan/Automated-Floor-Plan-Digitalization"


class RasterscanDetector:
    """Wall detector using Rasterscan HuggingFace Space API."""

    def __init__(self):
        if not GRADIO_AVAILABLE:
            raise RuntimeError("gradio_client not installed. Run: pip install gradio_client")

        self.client = None
        self.scale = 0.05  # meters per pixel default

    def _ensure_client(self):
        """Lazy initialization of the Gradio client."""
        if self.client is None:
            logger.info(f"Connecting to Rasterscan HuggingFace Space: {RASTERSCAN_SPACE}")
            self.client = Client(RASTERSCAN_SPACE)
            logger.info("Connected to Rasterscan API")

    def detect_walls(self, image: np.ndarray, scale: float = 0.05) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls and rooms in a floor plan image using Rasterscan API.

        Args:
            image: BGR image as numpy array
            scale: Meters per pixel (used for area calculations)

        Returns:
            Tuple of (list of WallSegment, list of Room)
        """
        self.scale = scale
        self._ensure_client()

        # Save image to temp file (Gradio needs file path)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, image)

        try:
            logger.info("Sending image to Rasterscan API...")
            result = self.client.predict(
                file=handle_file(temp_path),
                api_name="/run"
            )

            output_image, output_json = result
            logger.info(f"Rasterscan returned: {len(output_json.get('walls', []))} walls, "
                       f"{len(output_json.get('rooms', []))} rooms, "
                       f"{len(output_json.get('doors', []))} doors")

            # Parse walls
            walls = self._parse_walls(output_json.get('walls', []))

            # Parse rooms
            rooms = self._parse_rooms(output_json.get('rooms', []))

            return walls, rooms

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _parse_walls(self, wall_data: List[dict]) -> List[WallSegment]:
        """Convert Rasterscan wall format to WallSegment objects."""
        walls = []

        for wall in wall_data:
            position = wall.get('position', [])
            if len(position) >= 2:
                p1, p2 = position[0], position[1]

                # Create WallSegment
                segment = WallSegment(
                    start=Point(x=float(p1[0]), y=float(p1[1])),
                    end=Point(x=float(p2[0]), y=float(p2[1])),
                    thickness=0.15,  # Default wall thickness in meters
                    material="concrete",
                    attenuation_db=MATERIAL_ATTENUATION['concrete']
                )
                walls.append(segment)

        return walls

    def _parse_rooms(self, room_data: List) -> List[Room]:
        """Convert Rasterscan room format to Room objects."""
        rooms = []

        for i, room_points in enumerate(room_data):
            if not room_points:
                continue

            # Extract polygon points
            polygon = []
            for point in room_points:
                if isinstance(point, dict) and 'x' in point and 'y' in point:
                    polygon.append([float(point['x']), float(point['y'])])

            if len(polygon) >= 3:
                # Calculate area using shoelace formula
                area = self._polygon_area(polygon) * (self.scale ** 2)

                rooms.append(Room(
                    name=f"Room {i + 1}",
                    area=area,
                    polygon=polygon
                ))

        return rooms

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

    def detect_walls_from_file(self, file_path: str, scale: float = 0.05) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls from an image file.

        Args:
            file_path: Path to floor plan image
            scale: Meters per pixel

        Returns:
            Tuple of (list of WallSegment, list of Room)
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return self.detect_walls(image, scale)


# Check if Rasterscan is available
RASTERSCAN_AVAILABLE = GRADIO_AVAILABLE
