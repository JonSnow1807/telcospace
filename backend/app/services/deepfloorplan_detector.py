"""Deep learning-based floor plan wall detection using TF2DeepFloorplan.

This model was specifically trained on floor plans and provides accurate
semantic segmentation of walls, doors, windows, and room types.

Reference: https://github.com/zcemycl/TF2DeepFloorplan
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# TensorFlow import is optional and may fail due to version conflicts
tf = None
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TF_AVAILABLE = False
    logging.warning(f"TensorFlow not available for DeepFloorplan: {e}")

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

# Model path - relative to backend directory
MODEL_PATH = Path(__file__).parent.parent.parent / "dfp_model" / "model.tflite"

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

# Room type mapping from model output
ROOM_TYPES = {
    0: "background",
    1: "closet",
    2: "bathroom",
    3: "living_room",  # living/kitchen/dining
    4: "bedroom",
    5: "hall",
    6: "balcony",
}

# Boundary type mapping
BOUNDARY_TYPES = {
    0: "background",
    1: "opening",  # door/window
    2: "wall",
}


class DeepFloorplanDetector:
    """
    Deep learning-based wall detection using TF2DeepFloorplan model.

    This provides pixel-perfect wall segmentation masks that are then
    vectorized into line segments for the optimization algorithm.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector with the TFLite model.

        Args:
            model_path: Path to the TFLite model file. If not provided,
                       uses the default path.

        Raises:
            RuntimeError: If TensorFlow is not available
        """
        if not TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow is not available. DeepFloorplan requires TensorFlow. "
                "Try: pip install tensorflow-macos tensorflow-metal (on Mac) "
                "or pip install tensorflow (on other systems)"
            )

        self.model_path = model_path or str(MODEL_PATH)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.scale = 0.05  # Default scale

        self._load_model()

    def _load_model(self):
        """Load the TFLite model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"DeepFloorplan model not found at {self.model_path}. "
                "Please download from https://github.com/zcemycl/TF2DeepFloorplan"
            )

        logger.info(f"Loading DeepFloorplan model from {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info("DeepFloorplan model loaded successfully")

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls and rooms in a floor plan image.

        Args:
            image: Input BGR or RGB image (numpy array)
            scale: Meters per pixel for thickness conversion

        Returns:
            Tuple of (walls, rooms)
        """
        self.scale = scale
        original_shape = image.shape[:2]

        # Preprocess image for model
        input_tensor = self._preprocess(image)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        # Get outputs - room and boundary predictions
        # Note: output order depends on model architecture
        logits_r = self.interpreter.get_tensor(self.output_details[1]["index"])
        logits_cw = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Convert to class predictions
        room_pred = np.argmax(logits_r[0], axis=-1)
        boundary_pred = np.argmax(logits_cw[0], axis=-1)

        # Resize predictions back to original size
        room_pred = cv2.resize(
            room_pred.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        boundary_pred = cv2.resize(
            boundary_pred.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Extract walls from boundary prediction (label 2 = wall)
        wall_mask = (boundary_pred == 2).astype(np.uint8) * 255

        # Post-process wall mask
        wall_mask = self._postprocess_wall_mask(wall_mask)

        # Vectorize walls into line segments
        walls = self._vectorize_walls(wall_mask)

        # Extract rooms from room prediction
        rooms = self._extract_rooms(room_pred)

        logger.info(f"DeepFloorplan detected {len(walls)} walls and {len(rooms)} rooms")

        return walls, rooms

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from cv2.imread
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image

        # Resize to model input size (512x512)
        img = cv2.resize(img, (512, 512))

        # Normalize to [0, 1]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess_wall_mask(self, wall_mask: np.ndarray) -> np.ndarray:
        """Clean up the wall mask."""
        # Close small gaps
        kernel = np.ones((3, 3), np.uint8)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)

        return wall_mask

    def _vectorize_walls(self, wall_mask: np.ndarray) -> List[WallSegment]:
        """
        Convert wall mask to vector line segments.

        Uses skeletonization and line detection to extract wall centerlines.
        """
        # Skeletonize to get wall centerlines
        skeleton = self._skeletonize(wall_mask)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Snap to horizontal/vertical if close
            x1, y1, x2, y2 = self._snap_to_axis(x1, y1, x2, y2)

            # Estimate thickness from original mask
            thickness = self._estimate_thickness(wall_mask, x1, y1, x2, y2)
            thickness_m = max(0.1, min(0.5, thickness * self.scale))

            wall = WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=thickness_m,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )
            walls.append(wall)

        # Merge collinear walls
        walls = self._merge_collinear_walls(walls)

        # Connect nearby endpoints
        walls = self._connect_endpoints(walls)

        return walls

    def _skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """Skeletonize a binary image to get centerlines."""
        # Use morphological thinning
        skeleton = np.zeros_like(binary_image)
        img = binary_image.copy()

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            if cv2.countNonZero(img) == 0:
                break

        return skeleton

    def _snap_to_axis(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        threshold: int = 5
    ) -> Tuple[int, int, int, int]:
        """Snap nearly horizontal/vertical lines to exact H/V."""
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

    def _estimate_thickness(
        self,
        wall_mask: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int
    ) -> float:
        """Estimate wall thickness from the mask."""
        # Sample at midpoint
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2

        height, width = wall_mask.shape
        if not (0 <= mx < width and 0 <= my < height):
            return 5.0

        # Determine direction perpendicular to wall
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:
            # Horizontal wall - measure vertical thickness
            top = my
            while top > 0 and wall_mask[top, mx] > 0:
                top -= 1
            bottom = my
            while bottom < height - 1 and wall_mask[bottom, mx] > 0:
                bottom += 1
            return float(bottom - top)
        else:
            # Vertical wall - measure horizontal thickness
            left = mx
            while left > 0 and wall_mask[my, left] > 0:
                left -= 1
            right = mx
            while right < width - 1 and wall_mask[my, right] > 0:
                right += 1
            return float(right - left)

    def _merge_collinear_walls(
        self,
        walls: List[WallSegment],
        angle_threshold: float = 5.0,
        distance_threshold: float = 20.0
    ) -> List[WallSegment]:
        """Merge walls that are collinear and close together."""
        if len(walls) < 2:
            return walls

        merged = []
        used = set()

        for i, wall1 in enumerate(walls):
            if i in used:
                continue

            # Find walls to merge with this one
            group = [wall1]
            used.add(i)

            for j, wall2 in enumerate(walls):
                if j in used:
                    continue

                if self._are_collinear(wall1, wall2, angle_threshold, distance_threshold):
                    group.append(wall2)
                    used.add(j)

            # Merge the group
            if len(group) > 1:
                merged.append(self._merge_wall_group(group))
            else:
                merged.append(wall1)

        return merged

    def _are_collinear(
        self,
        wall1: WallSegment,
        wall2: WallSegment,
        angle_threshold: float,
        distance_threshold: float
    ) -> bool:
        """Check if two walls are collinear and close."""
        # Calculate angles
        angle1 = np.arctan2(
            wall1.end.y - wall1.start.y,
            wall1.end.x - wall1.start.x
        )
        angle2 = np.arctan2(
            wall2.end.y - wall2.start.y,
            wall2.end.x - wall2.start.x
        )

        angle_diff = abs(np.degrees(angle1 - angle2))
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff > angle_threshold and abs(angle_diff - 180) > angle_threshold:
            return False

        # Check distance between endpoints
        endpoints1 = [(wall1.start.x, wall1.start.y), (wall1.end.x, wall1.end.y)]
        endpoints2 = [(wall2.start.x, wall2.start.y), (wall2.end.x, wall2.end.y)]

        min_dist = float('inf')
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_dist = min(min_dist, dist)

        return min_dist < distance_threshold

    def _merge_wall_group(self, walls: List[WallSegment]) -> WallSegment:
        """Merge a group of collinear walls into one."""
        # Collect all endpoints
        points = []
        for wall in walls:
            points.append((wall.start.x, wall.start.y))
            points.append((wall.end.x, wall.end.y))

        # Find the two most distant points
        max_dist = 0
        best_pair = (points[0], points[1])

        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i >= j:
                    continue
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)

        # Use average thickness
        avg_thickness = sum(w.thickness for w in walls) / len(walls)

        return WallSegment(
            start=Point(x=best_pair[0][0], y=best_pair[0][1]),
            end=Point(x=best_pair[1][0], y=best_pair[1][1]),
            thickness=avg_thickness,
            material='concrete',
            attenuation_db=MATERIAL_ATTENUATION['concrete']
        )

    def _connect_endpoints(
        self,
        walls: List[WallSegment],
        threshold: float = 15.0
    ) -> List[WallSegment]:
        """Connect nearby wall endpoints to form proper corners."""
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

                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist < threshold:
                    cluster.append((w2, e2, x2, y2))
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Snap clustered endpoints to their centroid
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

    def _extract_rooms(self, room_pred: np.ndarray) -> List[Room]:
        """Extract room polygons from room prediction mask."""
        rooms = []

        for room_id in range(1, 7):  # Room types 1-6
            # Get mask for this room type
            room_mask = (room_pred == room_id).astype(np.uint8) * 255

            if cv2.countNonZero(room_mask) == 0:
                continue

            # Find contours
            contours, _ = cv2.findContours(
                room_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 1000:  # Minimum area in pixels
                    continue

                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Convert to polygon
                polygon = [[float(p[0][0]), float(p[0][1])] for p in approx]

                if len(polygon) >= 3:
                    room_type = ROOM_TYPES.get(room_id, f"room_{room_id}")
                    area_m2 = area * (self.scale ** 2)

                    room = Room(
                        name=room_type.replace('_', ' ').title(),
                        area=area_m2,
                        polygon=polygon
                    )
                    rooms.append(room)

        return rooms

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


def detect_walls_with_deepfloorplan(
    image_path: str,
    scale: float = 0.05
) -> Tuple[List[WallSegment], List[Room]]:
    """
    Convenience function to detect walls using DeepFloorplan model.

    Args:
        image_path: Path to floor plan image
        scale: Meters per pixel

    Returns:
        Tuple of (walls, rooms)
    """
    detector = DeepFloorplanDetector()
    return detector.detect_walls_from_file(image_path, scale)
