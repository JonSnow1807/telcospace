"""OCR service for extracting text from floor plan images."""

import re
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Lazy imports for heavy libraries
_pytesseract = None
_easyocr_reader = None


def get_pytesseract():
    """Lazy import of pytesseract."""
    global _pytesseract
    if _pytesseract is None:
        try:
            import pytesseract
            _pytesseract = pytesseract
        except ImportError:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
    return _pytesseract


def get_easyocr_reader():
    """Lazy initialization of EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except ImportError:
            raise ImportError("easyocr is required. Install with: pip install easyocr")
    return _easyocr_reader


@dataclass
class TextRegion:
    """Detected text region in an image."""
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    angle: float = 0.0  # Rotation in degrees


@dataclass
class RoomLabel:
    """Detected room label."""
    name: str
    center_x: float
    center_y: float
    confidence: float


@dataclass
class DimensionText:
    """Detected dimension annotation."""
    value_meters: float
    raw_text: str
    x: float
    y: float
    confidence: float


@dataclass
class ScaleText:
    """Detected scale notation."""
    ratio: float  # e.g., 100 for 1:100
    raw_text: str
    confidence: float


# Common room name patterns
ROOM_PATTERNS = [
    r'\b(kitchen|küche|cuisine|cocina)\b',
    r'\b(bedroom|schlafzimmer|chambre|dormitorio)\b',
    r'\b(bathroom|badezimmer|salle de bain|baño)\b',
    r'\b(living\s*room|wohnzimmer|salon|sala)\b',
    r'\b(dining\s*room|esszimmer|salle à manger|comedor)\b',
    r'\b(office|büro|bureau|oficina)\b',
    r'\b(closet|schrank|placard|armario)\b',
    r'\b(hall|hallway|corridor|flur|couloir|pasillo)\b',
    r'\b(garage|garaje)\b',
    r'\b(laundry|waschküche|buanderie|lavandería)\b',
    r'\b(master|haupt)\b',
    r'\b(guest|gäste|invité|huésped)\b',
    r'\b(study|arbeitszimmer|étude|estudio)\b',
    r'\b(den|wohnbereich)\b',
    r'\b(foyer|eingang|entrée|vestíbulo)\b',
    r'\b(entry|entrance|eingang|entrée|entrada)\b',
    r'\b(utility|hauswirtschaft|utilité|utilidad)\b',
    r'\b(storage|lager|stockage|almacenamiento)\b',
    r'\b(patio|terrasse|terraza)\b',
    r'\b(balcony|balkon|balcón)\b',
    r'\b(terrace|terrasse|terraza)\b',
    r'\b(wc|toilet|toilette)\b',
    r'\b(pantry|speisekammer|garde-manger|despensa)\b',
    r'\b(attic|dachboden|grenier|ático)\b',
    r'\b(basement|keller|sous-sol|sótano)\b',
    r'\broom\s*\d+\b',
    r'\bbr\s*\d+\b',
    r'\bbed\s*\d+\b',
]

# Dimension patterns (meters, feet, inches)
DIMENSION_PATTERNS = [
    # Metric: 3.5m, 3.5 m, 3500mm, 350cm
    (r'(\d+(?:\.\d+)?)\s*m(?:eters?)?(?!\w)', 1.0),
    (r'(\d+(?:\.\d+)?)\s*mm(?:\w)?', 0.001),
    (r'(\d+(?:\.\d+)?)\s*cm(?:\w)?', 0.01),
    # Imperial: 12', 12ft, 12'6", 12ft 6in
    (r"(\d+(?:\.\d+)?)\s*['\u2032]\s*(?:(\d+(?:\.\d+)?)\s*[\"″\u2033])?", 'feet_inches'),
    (r'(\d+(?:\.\d+)?)\s*(?:ft|feet)(?:\s*(\d+(?:\.\d+)?)\s*(?:in|inch(?:es)?)?)?', 'feet_inches'),
    (r'(\d+(?:\.\d+)?)\s*(?:in|inch(?:es)?)', 0.0254),
]

# Scale patterns
SCALE_PATTERNS = [
    r'1\s*:\s*(\d+)',                           # 1:100
    r'scale\s*[:\-]?\s*1\s*:\s*(\d+)',          # Scale: 1:100
    r'(\d+)\s*:\s*1',                           # 100:1 (reverse)
    r'1/(\d+)',                                  # 1/100
    r"1/(\d+)['\"]\s*=\s*1['\"]",               # 1/4" = 1'
]


class OCRService:
    """Service for extracting text from floor plan images."""

    def __init__(self, use_easyocr: bool = True, use_tesseract: bool = True):
        """
        Initialize OCR service.

        Args:
            use_easyocr: Whether to use EasyOCR (better for rotated text)
            use_tesseract: Whether to use Tesseract (faster, more accurate for straight text)
        """
        self.use_easyocr = use_easyocr
        self.use_tesseract = use_tesseract

    def extract_all_text(self, image: np.ndarray) -> List[TextRegion]:
        """
        Extract all text regions from image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of detected text regions
        """
        results = []

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Try Tesseract first (faster)
        if self.use_tesseract:
            try:
                tesseract_results = self._extract_with_tesseract(gray)
                results.extend(tesseract_results)
            except Exception as e:
                logger.warning(f"Tesseract extraction failed: {e}")

        # Use EasyOCR for additional detection (better with rotated text)
        if self.use_easyocr:
            try:
                easyocr_results = self._extract_with_easyocr(image)
                # Merge results, avoiding duplicates
                results = self._merge_results(results, easyocr_results)
            except Exception as e:
                logger.warning(f"EasyOCR extraction failed: {e}")

        return results

    def _extract_with_tesseract(self, gray: np.ndarray) -> List[TextRegion]:
        """Extract text using Tesseract."""
        pytesseract = get_pytesseract()
        results = []

        # Preprocess image
        processed = self._preprocess_for_ocr(gray)

        # Get detailed OCR data
        try:
            data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config='--psm 11'  # Sparse text mode
            )

            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

                if text and conf > 30:  # Filter low confidence
                    results.append(TextRegion(
                        text=text,
                        bbox=(
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ),
                        confidence=conf / 100.0,
                        angle=0.0
                    ))
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")

        return results

    def _extract_with_easyocr(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text using EasyOCR."""
        reader = get_easyocr_reader()
        results = []

        try:
            # EasyOCR expects BGR or RGB image
            ocr_results = reader.readtext(image)

            for (bbox, text, conf) in ocr_results:
                if not text.strip():
                    continue

                # Convert polygon bbox to rectangle
                points = np.array(bbox)
                x = int(min(points[:, 0]))
                y = int(min(points[:, 1]))
                w = int(max(points[:, 0]) - x)
                h = int(max(points[:, 1]) - y)

                # Calculate rotation angle from bbox
                angle = self._calculate_text_angle(bbox)

                results.append(TextRegion(
                    text=text.strip(),
                    bbox=(x, y, w, h),
                    confidence=float(conf),
                    angle=angle
                ))
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")

        return results

    def _preprocess_for_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Denoise
        processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)

        return processed

    def _calculate_text_angle(self, bbox: List[List[float]]) -> float:
        """Calculate rotation angle from quadrilateral bbox."""
        try:
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Top edge vector
            dx = bbox[1][0] - bbox[0][0]
            dy = bbox[1][1] - bbox[0][1]
            angle = math.degrees(math.atan2(dy, dx))
            return angle
        except Exception:
            return 0.0

    def _merge_results(
        self,
        results1: List[TextRegion],
        results2: List[TextRegion]
    ) -> List[TextRegion]:
        """Merge OCR results, removing duplicates."""
        merged = list(results1)

        for r2 in results2:
            is_duplicate = False
            for r1 in results1:
                # Check if bboxes overlap significantly
                if self._bbox_iou(r1.bbox, r2.bbox) > 0.5:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if r2.confidence > r1.confidence:
                        merged.remove(r1)
                        merged.append(r2)
                    break

            if not is_duplicate:
                merged.append(r2)

        return merged

    def _bbox_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union for two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def find_room_labels(self, image: np.ndarray) -> List[RoomLabel]:
        """
        Find room name labels in image.

        Args:
            image: Input image

        Returns:
            List of detected room labels
        """
        text_regions = self.extract_all_text(image)
        room_labels = []

        for region in text_regions:
            text_lower = region.text.lower()

            # Check against room patterns
            for pattern in ROOM_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    x, y, w, h = region.bbox
                    room_labels.append(RoomLabel(
                        name=region.text,
                        center_x=x + w / 2,
                        center_y=y + h / 2,
                        confidence=region.confidence
                    ))
                    break

        return room_labels

    def find_dimension_text(self, image: np.ndarray) -> List[DimensionText]:
        """
        Find dimension annotations in image.

        Args:
            image: Input image

        Returns:
            List of detected dimensions with values in meters
        """
        text_regions = self.extract_all_text(image)
        dimensions = []

        for region in text_regions:
            value = self._parse_dimension(region.text)
            if value is not None:
                x, y, w, h = region.bbox
                dimensions.append(DimensionText(
                    value_meters=value,
                    raw_text=region.text,
                    x=x + w / 2,
                    y=y + h / 2,
                    confidence=region.confidence
                ))

        return dimensions

    def _parse_dimension(self, text: str) -> Optional[float]:
        """
        Parse dimension text to meters.

        Args:
            text: Text to parse (e.g., "3.5m", "12'6\"", "350cm")

        Returns:
            Value in meters, or None if not a dimension
        """
        text = text.strip().lower()

        for pattern, multiplier in DIMENSION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if multiplier == 'feet_inches':
                    # Handle feet and inches
                    feet = float(match.group(1))
                    inches = float(match.group(2)) if match.group(2) else 0
                    return feet * 0.3048 + inches * 0.0254
                else:
                    return float(match.group(1)) * multiplier

        return None

    def find_scale_text(self, image: np.ndarray) -> Optional[ScaleText]:
        """
        Find scale notation in image.

        Args:
            image: Input image

        Returns:
            Detected scale information, or None if not found
        """
        text_regions = self.extract_all_text(image)

        for region in text_regions:
            scale = self._parse_scale(region.text)
            if scale is not None:
                return ScaleText(
                    ratio=scale,
                    raw_text=region.text,
                    confidence=region.confidence
                )

        return None

    def _parse_scale(self, text: str) -> Optional[float]:
        """
        Parse scale text to ratio.

        Args:
            text: Text to parse (e.g., "1:100", "Scale 1:50")

        Returns:
            Scale ratio (e.g., 100 for 1:100), or None if not a scale
        """
        text = text.strip()

        for pattern in SCALE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def extract_annotations(
        self,
        image: np.ndarray
    ) -> Tuple[List[RoomLabel], List[DimensionText], Optional[ScaleText]]:
        """
        Extract all annotations from image in one pass.

        More efficient than calling individual methods separately.

        Args:
            image: Input image

        Returns:
            Tuple of (room_labels, dimensions, scale_text)
        """
        text_regions = self.extract_all_text(image)

        room_labels = []
        dimensions = []
        scale_text = None

        for region in text_regions:
            text_lower = region.text.lower()
            x, y, w, h = region.bbox

            # Check for room labels
            for pattern in ROOM_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    room_labels.append(RoomLabel(
                        name=region.text,
                        center_x=x + w / 2,
                        center_y=y + h / 2,
                        confidence=region.confidence
                    ))
                    break

            # Check for dimensions
            dim_value = self._parse_dimension(region.text)
            if dim_value is not None:
                dimensions.append(DimensionText(
                    value_meters=dim_value,
                    raw_text=region.text,
                    x=x + w / 2,
                    y=y + h / 2,
                    confidence=region.confidence
                ))

            # Check for scale (only keep first found)
            if scale_text is None:
                scale = self._parse_scale(region.text)
                if scale is not None:
                    scale_text = ScaleText(
                        ratio=scale,
                        raw_text=region.text,
                        confidence=region.confidence
                    )

        return room_labels, dimensions, scale_text
