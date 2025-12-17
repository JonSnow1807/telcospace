"""SAM (Segment Anything Model) based wall detection.

Uses Meta's SAM for pixel-perfect wall segmentation, then vectorizes
the masks into line segments.

This approach gives the most accurate wall boundaries possible.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from app.schemas.project import WallSegment, Point, Room

logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path(__file__).parent.parent.parent / "sam_model" / "sam_vit_b.pth"

# Material attenuation
MATERIAL_ATTENUATION = {
    'concrete': 15.0,
    'brick': 12.0,
    'wood': 6.0,
    'glass': 5.0,
    'drywall': 3.0,
    'metal': 25.0,
    'unknown': 10.0
}


class SAMWallDetector:
    """
    SAM-based wall detection for pixel-perfect accuracy.

    Workflow:
    1. Create initial wall mask using CV (threshold + morphology)
    2. Find wall contours as prompts for SAM
    3. Use SAM to get pixel-perfect segmentation
    4. Vectorize the refined mask into line segments
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(MODEL_PATH)
        self.sam = None
        self.predictor = None
        self.scale = 0.05
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self._load_model()

    def _load_model(self):
        """Load SAM model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"SAM model not found at {self.model_path}. "
                "Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        try:
            from segment_anything import sam_model_registry, SamPredictor

            logger.info(f"Loading SAM model from {self.model_path}")
            self.sam = sam_model_registry["vit_b"](checkpoint=self.model_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            logger.info(f"SAM model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            raise

    def detect_walls(
        self,
        image: np.ndarray,
        scale: float = 0.05
    ) -> Tuple[List[WallSegment], List[Room]]:
        """
        Detect walls with pixel-perfect accuracy using SAM.
        """
        self.scale = scale
        height, width = image.shape[:2]

        # Step 1: Create initial wall mask using CV
        initial_mask = self._create_initial_wall_mask(image)

        # Step 2: Find wall contours to use as SAM prompts
        contours = self._find_wall_contours(initial_mask)

        if not contours:
            logger.warning("No wall contours found")
            return [], []

        # Step 3: Use SAM to refine each wall region
        logger.info(f"Refining {len(contours)} wall regions with SAM...")
        self.predictor.set_image(image)

        refined_mask = np.zeros((height, width), dtype=np.uint8)

        for contour in contours:
            # Get bounding box for this contour
            x, y, w, h = cv2.boundingRect(contour)

            # Skip very small contours
            if w < 20 and h < 20:
                continue

            # Create box prompt for SAM
            box = np.array([x, y, x + w, y + h])

            # Get SAM prediction
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True
            )

            # Use the mask with highest score
            best_mask = masks[np.argmax(scores)]
            refined_mask = np.maximum(refined_mask, best_mask.astype(np.uint8) * 255)

        # Step 4: Vectorize the refined mask
        walls = self._vectorize_mask(refined_mask)

        logger.info(f"SAM detected {len(walls)} walls with pixel-perfect accuracy")

        return walls, []

    def _create_initial_wall_mask(self, image: np.ndarray) -> np.ndarray:
        """Create initial binary wall mask using CV."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Otsu's thresholding to find dark regions (walls)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)

        # Close small gaps
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        return opened

    def _find_wall_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours that are likely walls."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by size and shape
        wall_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Skip tiny contours
                continue

            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / (min(w, h) + 1)

            # Walls are typically elongated (high aspect ratio)
            # or large rectangular regions
            if aspect > 2 or area > 2000:
                wall_contours.append(contour)

        return wall_contours

    def _vectorize_mask(self, mask: np.ndarray) -> List[WallSegment]:
        """Convert wall mask to vector line segments."""
        # Skeletonize to get centerlines
        skeleton = self._skeletonize(mask)

        # Use Hough transform to detect lines
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=15
        )

        if lines is None:
            return []

        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Snap to H/V if close
            x1, y1, x2, y2 = self._snap_to_axis(x1, y1, x2, y2)

            # Estimate thickness
            thickness = self._measure_thickness(mask, x1, y1, x2, y2)
            thickness_m = max(0.1, min(0.5, thickness * self.scale))

            walls.append(WallSegment(
                start=Point(x=float(x1), y=float(y1)),
                end=Point(x=float(x2), y=float(y2)),
                thickness=thickness_m,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            ))

        # Merge and connect walls
        walls = self._merge_collinear_walls(walls)
        walls = self._connect_endpoints(walls)

        return walls

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Skeletonize binary image to get centerlines."""
        skeleton = np.zeros_like(binary)
        img = binary.copy()
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

    def _snap_to_axis(self, x1, y1, x2, y2, threshold=8):
        """Snap nearly H/V lines to exact H/V."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy < threshold and dx > dy:
            avg_y = (y1 + y2) // 2
            return x1, avg_y, x2, avg_y
        if dx < threshold and dy > dx:
            avg_x = (x1 + x2) // 2
            return avg_x, y1, avg_x, y2

        return x1, y1, x2, y2

    def _measure_thickness(self, mask, x1, y1, x2, y2) -> float:
        """Measure wall thickness at midpoint."""
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        h, w = mask.shape

        if not (0 <= mx < w and 0 <= my < h):
            return 10.0

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:  # Horizontal - measure vertical
            top = my
            while top > 0 and mask[top, mx] > 0:
                top -= 1
            bottom = my
            while bottom < h - 1 and mask[bottom, mx] > 0:
                bottom += 1
            return float(bottom - top)
        else:  # Vertical - measure horizontal
            left = mx
            while left > 0 and mask[my, left] > 0:
                left -= 1
            right = mx
            while right < w - 1 and mask[my, right] > 0:
                right += 1
            return float(right - left)

    def _merge_collinear_walls(self, walls: List[WallSegment]) -> List[WallSegment]:
        """Merge walls that are collinear and close."""
        if len(walls) < 2:
            return walls

        horizontal = [w for w in walls if abs(w.end.x - w.start.x) > abs(w.end.y - w.start.y)]
        vertical = [w for w in walls if abs(w.end.y - w.start.y) >= abs(w.end.x - w.start.x)]

        merged_h = self._merge_parallel(horizontal, True)
        merged_v = self._merge_parallel(vertical, False)

        return merged_h + merged_v

    def _merge_parallel(self, walls: List[WallSegment], is_horizontal: bool) -> List[WallSegment]:
        """Merge parallel walls."""
        if len(walls) < 2:
            return walls

        if is_horizontal:
            walls = sorted(walls, key=lambda w: (w.start.y + w.end.y) / 2)
        else:
            walls = sorted(walls, key=lambda w: (w.start.x + w.end.x) / 2)

        merged = []
        used = set()

        for i, w1 in enumerate(walls):
            if i in used:
                continue

            group = [w1]
            used.add(i)
            pos1 = (w1.start.y + w1.end.y) / 2 if is_horizontal else (w1.start.x + w1.end.x) / 2

            for j, w2 in enumerate(walls):
                if j in used:
                    continue
                pos2 = (w2.start.y + w2.end.y) / 2 if is_horizontal else (w2.start.x + w2.end.x) / 2

                if abs(pos2 - pos1) < 15:
                    group.append(w2)
                    used.add(j)

            if len(group) > 1:
                merged.append(self._merge_group(group, is_horizontal))
            else:
                merged.append(w1)

        return merged

    def _merge_group(self, walls: List[WallSegment], is_horizontal: bool) -> WallSegment:
        """Merge a group of walls into one."""
        if is_horizontal:
            avg_y = sum((w.start.y + w.end.y) / 2 for w in walls) / len(walls)
            min_x = min(min(w.start.x, w.end.x) for w in walls)
            max_x = max(max(w.start.x, w.end.x) for w in walls)
            return WallSegment(
                start=Point(x=min_x, y=avg_y),
                end=Point(x=max_x, y=avg_y),
                thickness=walls[0].thickness,
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
                thickness=walls[0].thickness,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )

    def _connect_endpoints(self, walls: List[WallSegment], threshold=15) -> List[WallSegment]:
        """Connect nearby wall endpoints."""
        if len(walls) < 2:
            return walls

        endpoints = []
        for i, w in enumerate(walls):
            endpoints.append((i, 'start', w.start.x, w.start.y))
            endpoints.append((i, 'end', w.end.x, w.end.y))

        clusters = []
        used = set()

        for i, (wi, et, x, y) in enumerate(endpoints):
            if i in used:
                continue
            cluster = [(wi, et, x, y)]
            used.add(i)

            for j, (wj, et2, x2, y2) in enumerate(endpoints):
                if j in used:
                    continue
                if ((x - x2)**2 + (y - y2)**2)**0.5 < threshold:
                    cluster.append((wj, et2, x2, y2))
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        walls_list = list(walls)
        for cluster in clusters:
            avg_x = sum(p[2] for p in cluster) / len(cluster)
            avg_y = sum(p[3] for p in cluster) / len(cluster)

            for wi, et, _, _ in cluster:
                w = walls_list[wi]
                if et == 'start':
                    walls_list[wi] = WallSegment(
                        start=Point(x=avg_x, y=avg_y),
                        end=w.end,
                        thickness=w.thickness,
                        material=w.material,
                        attenuation_db=w.attenuation_db
                    )
                else:
                    walls_list[wi] = WallSegment(
                        start=w.start,
                        end=Point(x=avg_x, y=avg_y),
                        thickness=w.thickness,
                        material=w.material,
                        attenuation_db=w.attenuation_db
                    )

        return walls_list


# Check availability
try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
