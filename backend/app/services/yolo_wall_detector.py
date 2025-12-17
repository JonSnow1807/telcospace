"""YOLOv8-based floor plan wall detection.

Uses a pre-trained YOLOv8 model specifically trained on floor plan images
to detect walls, doors, windows, and other architectural elements.

Reference: https://github.com/sanatladkat/floor-plan-object-detection
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path(__file__).parent.parent.parent / "yolo_floorplan" / "best.pt"

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


class YOLOWallDetector:
    """
    YOLOv8-based wall detection for floor plans.

    This detector uses object detection to find wall segments,
    then converts bounding boxes to line segments.
    """

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25):
        """
        Initialize the YOLO wall detector.

        Args:
            model_path: Path to the YOLOv8 model file (.pt)
            confidence: Minimum confidence threshold for detections
        """
        self.model_path = model_path or str(MODEL_PATH)
        self.confidence = confidence
        self.model = None
        self.scale = 0.05

        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {self.model_path}. "
                "Please ensure the model file exists."
            )

        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls in a floor plan image using YOLOv8.

        Args:
            image: Input BGR image (numpy array)
            scale: Meters per pixel

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale

        # Run YOLO detection
        results = self.model(image, conf=self.confidence, verbose=False)

        walls = []
        rooms = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = self.model.names[cls].lower()

                # Convert wall detections to line segments
                if 'wall' in class_name:
                    wall_segments = self._box_to_wall_segments(x1, y1, x2, y2, conf)
                    walls.extend(wall_segments)

        # Post-process walls
        walls = self._merge_nearby_walls(walls)
        walls = self._connect_wall_endpoints(walls)

        logger.info(f"YOLO detected {len(walls)} walls")

        return walls, rooms

    def _box_to_wall_segments(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        confidence: float
    ) -> List[WallSegment]:
        """
        Convert a bounding box to wall line segments.

        For thin horizontal boxes -> horizontal wall
        For thin vertical boxes -> vertical wall
        For square-ish boxes -> might be a corner, create both walls
        """
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else float('inf')

        walls = []

        # Determine wall orientation based on aspect ratio
        if aspect_ratio > 3:  # Horizontal wall
            # Wall runs along the center of the box
            center_y = (y1 + y2) / 2
            walls.append(WallSegment(
                start=Point(x=x1, y=center_y),
                end=Point(x=x2, y=center_y),
                thickness=min(height * self.scale, 0.3),
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            ))
        elif aspect_ratio < 0.33:  # Vertical wall
            center_x = (x1 + x2) / 2
            walls.append(WallSegment(
                start=Point(x=center_x, y=y1),
                end=Point(x=center_x, y=y2),
                thickness=min(width * self.scale, 0.3),
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            ))
        else:
            # Could be a corner or small room section
            # Create both horizontal and vertical segments
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Only if the box is large enough
            if width > 20 and height > 20:
                # Horizontal segment
                walls.append(WallSegment(
                    start=Point(x=x1, y=center_y),
                    end=Point(x=x2, y=center_y),
                    thickness=0.2,
                    material='concrete',
                    attenuation_db=MATERIAL_ATTENUATION['concrete']
                ))
                # Vertical segment
                walls.append(WallSegment(
                    start=Point(x=center_x, y=y1),
                    end=Point(x=center_x, y=y2),
                    thickness=0.2,
                    material='concrete',
                    attenuation_db=MATERIAL_ATTENUATION['concrete']
                ))

        return walls

    def _merge_nearby_walls(
        self,
        walls: List[WallSegment],
        threshold: float = 20.0
    ) -> List[WallSegment]:
        """Merge walls that are close and parallel."""
        if len(walls) < 2:
            return walls

        # Separate horizontal and vertical walls
        horizontal = []
        vertical = []

        for wall in walls:
            dx = abs(wall.end.x - wall.start.x)
            dy = abs(wall.end.y - wall.start.y)

            if dx > dy:
                horizontal.append(wall)
            else:
                vertical.append(wall)

        # Merge each group
        merged_h = self._merge_parallel_walls(horizontal, is_horizontal=True, threshold=threshold)
        merged_v = self._merge_parallel_walls(vertical, is_horizontal=False, threshold=threshold)

        return merged_h + merged_v

    def _merge_parallel_walls(
        self,
        walls: List[WallSegment],
        is_horizontal: bool,
        threshold: float
    ) -> List[WallSegment]:
        """Merge parallel walls that are close together."""
        if len(walls) < 2:
            return walls

        # Sort by position
        if is_horizontal:
            walls = sorted(walls, key=lambda w: (w.start.y + w.end.y) / 2)
        else:
            walls = sorted(walls, key=lambda w: (w.start.x + w.end.x) / 2)

        merged = []
        used = set()

        for i, wall1 in enumerate(walls):
            if i in used:
                continue

            group = [wall1]
            used.add(i)

            pos1 = (wall1.start.y + wall1.end.y) / 2 if is_horizontal else (wall1.start.x + wall1.end.x) / 2

            for j, wall2 in enumerate(walls):
                if j in used:
                    continue

                pos2 = (wall2.start.y + wall2.end.y) / 2 if is_horizontal else (wall2.start.x + wall2.end.x) / 2

                if abs(pos2 - pos1) < threshold:
                    # Check if ranges overlap
                    if is_horizontal:
                        r1 = (min(wall1.start.x, wall1.end.x), max(wall1.start.x, wall1.end.x))
                        r2 = (min(wall2.start.x, wall2.end.x), max(wall2.start.x, wall2.end.x))
                    else:
                        r1 = (min(wall1.start.y, wall1.end.y), max(wall1.start.y, wall1.end.y))
                        r2 = (min(wall2.start.y, wall2.end.y), max(wall2.start.y, wall2.end.y))

                    overlap = min(r1[1], r2[1]) - max(r1[0], r2[0])
                    if overlap > -50:  # Allow small gaps
                        group.append(wall2)
                        used.add(j)

            # Merge the group
            if len(group) > 1:
                merged.append(self._merge_wall_group(group, is_horizontal))
            else:
                merged.append(wall1)

        return merged

    def _merge_wall_group(
        self,
        walls: List[WallSegment],
        is_horizontal: bool
    ) -> WallSegment:
        """Merge a group of walls into one."""
        if is_horizontal:
            avg_y = sum((w.start.y + w.end.y) / 2 for w in walls) / len(walls)
            min_x = min(min(w.start.x, w.end.x) for w in walls)
            max_x = max(max(w.start.x, w.end.x) for w in walls)
            return WallSegment(
                start=Point(x=min_x, y=avg_y),
                end=Point(x=max_x, y=avg_y),
                thickness=sum(w.thickness for w in walls) / len(walls),
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )
        else:
            avg_x = sum((w.start.x + w.end.x) / 2 for w in walls) / len(walls)
            min_y = min(min(w.start.y, w.end.y) for w in walls)
            max_y = max(max(w.start.y, w.end.y) for w in walls)
            return WallSegment(
                start=Point(x=avg_x, y=min_y),
                end=Point(x=avg_x, y=max_y),
                thickness=sum(w.thickness for w in walls) / len(walls),
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )

    def _connect_wall_endpoints(
        self,
        walls: List[WallSegment],
        threshold: float = 15.0
    ) -> List[WallSegment]:
        """Connect nearby wall endpoints to form corners."""
        if len(walls) < 2:
            return walls

        # Collect all endpoints
        endpoints = []
        for i, wall in enumerate(walls):
            endpoints.append((i, 'start', wall.start.x, wall.start.y))
            endpoints.append((i, 'end', wall.end.x, wall.end.y))

        # Find clusters
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

        # Snap to centroids
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


# Check if YOLO is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
