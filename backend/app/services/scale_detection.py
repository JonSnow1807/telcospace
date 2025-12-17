"""Scale detection service for auto-detecting floor plan scale."""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import cv2

from app.services.ocr_service import TextRegion, DimensionText, ScaleText

logger = logging.getLogger(__name__)

# Standard object sizes for reference (in meters)
REFERENCE_OBJECTS = {
    'standard_door': 0.9,      # Standard interior door width
    'double_door': 1.8,        # Double door width
    'exterior_door': 0.914,    # 36" exterior door
    'garage_door': 2.4,        # Single car garage
    'window_small': 0.6,       # Small window
    'window_standard': 1.2,    # Standard window
    'toilet': 0.38,            # Toilet width
    'bathtub': 0.76,           # Standard bathtub width
    'sink': 0.45,              # Bathroom sink width
    'kitchen_counter': 0.6,    # Counter depth
}


@dataclass
class ScaleResult:
    """Result of scale detection."""
    meters_per_pixel: float
    confidence: float  # 0-1
    method: str        # Detection method used
    reference_points: List[Tuple[float, float]] = None  # Points used for calculation

    def __post_init__(self):
        if self.reference_points is None:
            self.reference_points = []


class ScaleDetector:
    """Auto-detect scale from floor plan images."""

    def __init__(self):
        self.ocr_service = None  # Lazy initialization

    def _get_ocr_service(self):
        """Lazy import of OCR service."""
        if self.ocr_service is None:
            from app.services.ocr_service import OCRService
            self.ocr_service = OCRService()
        return self.ocr_service

    def detect_scale(
        self,
        image: np.ndarray,
        ocr_results: Optional[List[TextRegion]] = None
    ) -> ScaleResult:
        """
        Detect scale using multiple methods.

        Tries different detection strategies and returns the best result.

        Args:
            image: Input image (BGR or grayscale)
            ocr_results: Pre-computed OCR results (optional, will compute if not provided)

        Returns:
            ScaleResult with detected scale and confidence
        """
        results = []

        # Get OCR results if not provided
        if ocr_results is None:
            try:
                ocr_service = self._get_ocr_service()
                ocr_results = ocr_service.extract_all_text(image)
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
                ocr_results = []

        # Try each detection method
        methods = [
            ('scale_text', self._detect_from_scale_text),
            ('scale_bar', self._detect_from_scale_bar),
            ('dimensions', self._detect_from_dimensions),
            ('reference_objects', self._detect_from_reference_objects),
        ]

        for method_name, method_func in methods:
            try:
                result = method_func(image, ocr_results)
                if result is not None:
                    result.method = method_name
                    results.append(result)
            except Exception as e:
                logger.debug(f"Scale detection method '{method_name}' failed: {e}")

        # Select best result
        return self._select_best_result(results)

    def _detect_from_scale_text(
        self,
        image: np.ndarray,
        ocr_results: List[TextRegion]
    ) -> Optional[ScaleResult]:
        """
        Detect scale from text notation like '1:100'.

        Args:
            image: Input image
            ocr_results: OCR text regions

        Returns:
            ScaleResult if scale text found, None otherwise
        """
        import re

        scale_patterns = [
            r'1\s*:\s*(\d+)',                    # 1:100
            r'scale\s*[:\-]?\s*1\s*:\s*(\d+)',   # Scale: 1:100
            r'1/(\d+)',                           # 1/100
        ]

        for region in ocr_results:
            text = region.text.strip()

            for pattern in scale_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    ratio = float(match.group(1))
                    if 10 <= ratio <= 1000:  # Reasonable scale range
                        # Scale 1:100 means 1 unit on paper = 100 units in reality
                        # We need to estimate DPI to convert to meters_per_pixel
                        # Assume typical screen/print DPI of 96
                        # For 1:100 scale at 96 DPI:
                        # 1 inch = 96 pixels = 100 inches in reality = 2.54 meters
                        # So meters_per_pixel = 2.54 / 96 * ratio / 100

                        # Simplified: assume drawing is displayed at scale
                        # Use a reasonable estimate
                        meters_per_pixel = ratio * 0.001  # Rough estimate

                        return ScaleResult(
                            meters_per_pixel=meters_per_pixel,
                            confidence=0.7,
                            method='scale_text'
                        )

        return None

    def _detect_from_scale_bar(
        self,
        image: np.ndarray,
        ocr_results: List[TextRegion]
    ) -> Optional[ScaleResult]:
        """
        Detect scale from visual scale bar.

        Looks for horizontal lines with tick marks and associated length text.

        Args:
            image: Input image
            ocr_results: OCR text regions

        Returns:
            ScaleResult if scale bar found, None otherwise
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return None

        # Find horizontal lines that could be scale bars
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1))

            # Check if mostly horizontal (within 5 degrees)
            if angle < 0.09 or angle > 3.05:
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if 30 < length < 500:  # Reasonable scale bar length
                    horizontal_lines.append({
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'length': length,
                        'center_y': (y1 + y2) / 2
                    })

        # Look for dimension text near horizontal lines
        import re

        for line in horizontal_lines:
            line_y = line['center_y']
            line_x1, line_x2 = min(line['x1'], line['x2']), max(line['x1'], line['x2'])

            for region in ocr_results:
                x, y, w, h = region.bbox
                text_center_y = y + h / 2
                text_center_x = x + w / 2

                # Check if text is near the line
                if abs(text_center_y - line_y) < 50:
                    if line_x1 - 50 < text_center_x < line_x2 + 50:
                        # Try to parse as dimension
                        value = self._parse_length_text(region.text)
                        if value is not None:
                            meters_per_pixel = value / line['length']

                            if 0.001 < meters_per_pixel < 1.0:  # Reasonable range
                                return ScaleResult(
                                    meters_per_pixel=meters_per_pixel,
                                    confidence=0.8,
                                    method='scale_bar',
                                    reference_points=[
                                        (line['x1'], line['y1']),
                                        (line['x2'], line['y2'])
                                    ]
                                )

        return None

    def _detect_from_dimensions(
        self,
        image: np.ndarray,
        ocr_results: List[TextRegion]
    ) -> Optional[ScaleResult]:
        """
        Detect scale from dimension annotations on the drawing.

        Finds dimension text near walls and calculates scale from pixel measurements.

        Args:
            image: Input image
            ocr_results: OCR text regions

        Returns:
            ScaleResult if dimensions found, None otherwise
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find dimension text
        dimensions = []
        for region in ocr_results:
            value = self._parse_length_text(region.text)
            if value is not None and 0.5 < value < 50:  # 0.5m to 50m reasonable
                x, y, w, h = region.bbox
                dimensions.append({
                    'value': value,
                    'x': x + w / 2,
                    'y': y + h / 2,
                    'text': region.text
                })

        if len(dimensions) < 2:
            return None

        # Detect lines (walls) in the image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return None

        # Try to associate dimensions with line lengths
        scale_estimates = []

        for dim in dimensions:
            # Find nearby lines that could correspond to this dimension
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Check if dimension is near either endpoint of the line
                dist1 = math.sqrt((dim['x'] - x1)**2 + (dim['y'] - y1)**2)
                dist2 = math.sqrt((dim['x'] - x2)**2 + (dim['y'] - y2)**2)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dist_mid = math.sqrt((dim['x'] - mid_x)**2 + (dim['y'] - mid_y)**2)

                # If dimension text is within reasonable distance of line
                if min(dist1, dist2, dist_mid) < 100:
                    if line_length > 20:  # Minimum line length
                        scale = dim['value'] / line_length
                        if 0.001 < scale < 1.0:
                            scale_estimates.append(scale)

        if not scale_estimates:
            return None

        # Use median to filter outliers
        median_scale = np.median(scale_estimates)
        filtered = [s for s in scale_estimates if 0.5 * median_scale < s < 2 * median_scale]

        if not filtered:
            return None

        final_scale = np.mean(filtered)
        confidence = min(0.9, 0.5 + 0.1 * len(filtered))  # Higher confidence with more data points

        return ScaleResult(
            meters_per_pixel=final_scale,
            confidence=confidence,
            method='dimensions'
        )

    def _detect_from_reference_objects(
        self,
        image: np.ndarray,
        ocr_results: List[TextRegion]
    ) -> Optional[ScaleResult]:
        """
        Detect scale from known-size objects like doors.

        Looks for door symbols/arcs and uses standard door width as reference.

        Args:
            image: Input image
            ocr_results: OCR text regions

        Returns:
            ScaleResult if reference objects found, None otherwise
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect circles/arcs that could be door swings
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        door_sizes = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Circle radius represents door swing arc
                # Door width ≈ radius (for 90° swing)
                radius = circle[2]
                if 15 < radius < 80:  # Reasonable door size range in pixels
                    door_sizes.append(radius)

        # Also look for small gaps in walls (door openings)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for rectangular openings
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]

            # Check if it looks like a door opening
            if min(width, height) > 5 and max(width, height) < 100:
                aspect_ratio = max(width, height) / (min(width, height) + 0.001)
                if 2 < aspect_ratio < 10:  # Door-like aspect ratio
                    door_sizes.append(min(width, height))

        if not door_sizes:
            return None

        # Assume standard door width
        standard_door = REFERENCE_OBJECTS['standard_door']
        median_pixels = np.median(door_sizes)

        meters_per_pixel = standard_door / median_pixels

        # Low confidence since we're assuming standard sizes
        confidence = min(0.5, 0.3 + 0.05 * len(door_sizes))

        if 0.005 < meters_per_pixel < 0.5:  # Reasonable range
            return ScaleResult(
                meters_per_pixel=meters_per_pixel,
                confidence=confidence,
                method='reference_objects'
            )

        return None

    def _parse_length_text(self, text: str) -> Optional[float]:
        """
        Parse length text to meters.

        Args:
            text: Text like "3.5m", "12ft", "350cm"

        Returns:
            Value in meters, or None if not parseable
        """
        import re
        text = text.strip().lower()

        patterns = [
            (r'(\d+(?:\.\d+)?)\s*m(?:eters?)?(?!\w)', 1.0),
            (r'(\d+(?:\.\d+)?)\s*mm', 0.001),
            (r'(\d+(?:\.\d+)?)\s*cm', 0.01),
            (r"(\d+(?:\.\d+)?)\s*['\u2032]", 0.3048),
            (r'(\d+(?:\.\d+)?)\s*(?:ft|feet)', 0.3048),
            (r'(\d+(?:\.\d+)?)\s*(?:in|inch)', 0.0254),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1)) * multiplier

        return None

    def _select_best_result(self, results: List[ScaleResult]) -> ScaleResult:
        """
        Select the best scale result from multiple detections.

        Args:
            results: List of scale detection results

        Returns:
            Best result, or default if no valid results
        """
        if not results:
            # Return default scale
            return ScaleResult(
                meters_per_pixel=0.05,  # Default: 5cm per pixel
                confidence=0.1,
                method='default'
            )

        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)

        # If top result has high confidence, use it
        if results[0].confidence > 0.6:
            return results[0]

        # If multiple results agree, increase confidence
        if len(results) >= 2:
            scale1 = results[0].meters_per_pixel
            scale2 = results[1].meters_per_pixel

            # Check if they're within 20% of each other
            if 0.8 < scale1 / scale2 < 1.2:
                # Average the scales
                avg_scale = (scale1 + scale2) / 2
                new_confidence = min(0.9, results[0].confidence + 0.2)

                return ScaleResult(
                    meters_per_pixel=avg_scale,
                    confidence=new_confidence,
                    method=f"{results[0].method}+{results[1].method}"
                )

        return results[0]


def estimate_scale_from_image_size(
    image_width: int,
    image_height: int,
    expected_room_size_m: float = 20.0
) -> float:
    """
    Rough scale estimate based on image size and expected room dimensions.

    Useful as a fallback when other methods fail.

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        expected_room_size_m: Expected typical room dimension in meters

    Returns:
        Estimated meters per pixel
    """
    # Assume the larger image dimension represents the room/building size
    larger_dim = max(image_width, image_height)
    return expected_room_size_m / larger_dim
