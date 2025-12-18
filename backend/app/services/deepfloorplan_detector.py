"""Deep Floor Plan Wall Detection using pre-trained TF2DeepFloorplan model."""

import os
import logging
import math
from typing import List, Tuple
from pathlib import Path
import numpy as np
import cv2

# Try tflite-runtime first (lighter), then tensorflow
try:
    import tflite_runtime.interpreter as tflite
    TF_AVAILABLE = True
    USING_TFLITE_RUNTIME = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TF_AVAILABLE = True
        USING_TFLITE_RUNTIME = False
    except ImportError:
        TF_AVAILABLE = False
        USING_TFLITE_RUNTIME = False

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

MATERIAL_ATTENUATION = {'concrete': 15.0}
MODEL_PATH = Path(__file__).parent.parent.parent / "dfp_model" / "model.tflite"


class DeepFloorplanDetector:
    """Wall detector using pre-trained TF2DeepFloorplan model."""

    def __init__(self):
        if not TF_AVAILABLE:
            raise RuntimeError("TFLite runtime not available")

        self.scale = 0.05
        self._load_model()

    def _load_model(self):
        model_path = str(MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading DeepFloorplan model (tflite_runtime={USING_TFLITE_RUNTIME})")
        
        if USING_TFLITE_RUNTIME:
            self.model = tflite.Interpreter(model_path=model_path)
        else:
            self.model = tflite.Interpreter(model_path=model_path)
        
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        logger.info("Model loaded successfully")

    def detect_walls(self, image: np.ndarray, scale: float = 0.05) -> Tuple[List[WallSegment], List[Room]]:
        self.scale = scale
        original_shape = image.shape[:2]

        logger.info(f"DeepFloorplan detection on {original_shape[1]}x{original_shape[0]}")

        # Preprocess
        input_tensor = self._preprocess(image)

        # Run inference
        logger.info("Running inference...")
        wall_mask, room_mask = self._run_inference(input_tensor, original_shape)

        # Vectorize walls
        walls = self._vectorize_walls(wall_mask)
        logger.info(f"  Raw walls: {len(walls)}")

        # Extract rooms
        rooms = self._extract_rooms(room_mask)

        # Cleanup
        walls = self._cleanup_walls(walls)
        logger.info(f"  Final: {len(walls)} walls, {len(rooms)} rooms")

        # Convert to WallSegments
        wall_segments = [
            WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=0.2,
                material="concrete",
                attenuation_db=MATERIAL_ATTENUATION["concrete"]
            )
            for x1, y1, x2, y2 in walls
        ]

        return wall_segments, rooms

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        resized = cv2.resize(rgb, (512, 512))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _run_inference(self, input_tensor: np.ndarray, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        self.model.set_tensor(self.input_details[0]["index"], input_tensor)
        self.model.invoke()

        output0 = self.model.get_tensor(self.output_details[0]["index"])
        output1 = self.model.get_tensor(self.output_details[1]["index"])

        # Boundary has 3 channels, room has more
        if output0.shape[-1] == 3:
            boundary_logits, room_logits = output0, output1
        else:
            boundary_logits, room_logits = output1, output0

        boundary_mask = np.argmax(boundary_logits[0], axis=-1)
        room_mask = np.argmax(room_logits[0], axis=-1)

        height, width = original_shape
        boundary_mask = cv2.resize(boundary_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        room_mask = cv2.resize(room_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

        wall_mask = (boundary_mask == 2).astype(np.uint8) * 255
        return wall_mask, room_mask

    def _vectorize_walls(self, wall_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if hasattr(cv2, 'ximgproc'):
            skeleton = cv2.ximgproc.thinning(wall_mask)
        else:
            skeleton = self._simple_skeleton(wall_mask)

        height, width = wall_mask.shape
        min_len = max(15, min(width, height) // 50)

        lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, 20, minLineLength=min_len, maxLineGap=10)
        if lines is None:
            lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, 10, minLineLength=min_len//2, maxLineGap=15)

        if lines is None:
            return []
        return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in lines]

    def _simple_skeleton(self, binary: np.ndarray) -> np.ndarray:
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()
        while cv2.countNonZero(temp) > 0:
            eroded = cv2.erode(temp, element)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(temp, cv2.dilate(eroded, element)))
            temp = eroded
        return skeleton

    def _extract_rooms(self, room_mask: np.ndarray) -> List[Room]:
        rooms = []
        room_names = {1: "Closet", 2: "Bathroom", 3: "Living Room", 4: "Bedroom", 5: "Hall", 6: "Balcony"}

        for label, name in room_names.items():
            mask = (room_mask == label).astype(np.uint8) * 255
            if np.sum(mask) == 0:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                area_sqm = cv2.contourArea(contour) * (self.scale ** 2)
                if area_sqm < 1.0:
                    continue

                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                polygon = [[float(p[0][0]), float(p[0][1])] for p in approx]

                if len(polygon) >= 3:
                    rooms.append(Room(name=f"{name}" if i == 0 else f"{name} {i+1}", area=area_sqm, polygon=polygon))

        return rooms

    def _cleanup_walls(self, walls: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not walls:
            return walls

        grid = 5
        aligned = []
        for x1, y1, x2, y2 in walls:
            x1, y1 = round(x1/grid)*grid, round(y1/grid)*grid
            x2, y2 = round(x2/grid)*grid, round(y2/grid)*grid
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dx == 0 and dy == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 5:
                y1 = y2 = round((y1+y2)/2/grid)*grid
            elif angle > 85:
                x1 = x2 = round((x1+x2)/2/grid)*grid
            if math.sqrt((x2-x1)**2 + (y2-y1)**2) >= 10:
                aligned.append((x1, y1, x2, y2))

        # Merge collinear and connect endpoints
        merged = self._merge_and_connect(aligned)
        return self._remove_duplicates(merged)

    def _merge_and_connect(self, walls):
        if len(walls) < 2:
            return walls
        
        # Simple merge: connect nearby endpoints
        threshold = 20
        for _ in range(3):  # Multiple passes
            endpoints = {i: [(walls[i][0], walls[i][1]), (walls[i][2], walls[i][3])] for i in range(len(walls))}
            
            for i in range(len(walls)):
                for j in range(i+1, len(walls)):
                    for ei, (ex, ey) in enumerate(endpoints[i]):
                        for ej, (fx, fy) in enumerate(endpoints[j]):
                            if math.sqrt((ex-fx)**2 + (ey-fy)**2) < threshold:
                                avg = (round((ex+fx)/2/5)*5, round((ey+fy)/2/5)*5)
                                endpoints[i][ei] = avg
                                endpoints[j][ej] = avg
            
            walls = [(endpoints[i][0][0], endpoints[i][0][1], endpoints[i][1][0], endpoints[i][1][1]) for i in range(len(walls))]
        
        return walls

    def _remove_duplicates(self, walls):
        unique = []
        for x1, y1, x2, y2 in walls:
            if x1 == x2 and y1 == y2:
                continue
            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1
            if not any(math.sqrt((x1-ux1)**2+(y1-uy1)**2) < 10 and math.sqrt((x2-ux2)**2+(y2-uy2)**2) < 10 
                       for ux1, uy1, ux2, uy2 in unique):
                unique.append((x1, y1, x2, y2))
        return unique

    def detect_walls_from_file(self, file_path: str, scale: float = 0.05):
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return self.detect_walls(image, scale)


DFP_AVAILABLE = TF_AVAILABLE and MODEL_PATH.exists()
