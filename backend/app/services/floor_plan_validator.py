"""Floor plan image validation service.

Validates whether an uploaded image is actually a floor plan
using multiple heuristic checks.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of floor plan validation."""
    is_valid: bool
    confidence: float  # 0-1 overall confidence
    scores: dict  # Individual check scores
    reasons: List[str]  # Human-readable explanations
    suggestions: List[str]  # Suggestions if invalid


class FloorPlanValidator:
    """Validates if an image is a floor plan using multiple heuristics."""

    # Thresholds (can be tuned)
    MIN_OVERALL_CONFIDENCE = 0.45  # Minimum to pass validation
    MIN_ORTHOGONALITY = 0.4       # Minimum orthogonality score required

    # Weight for each check (should sum to 1.0)
    WEIGHTS = {
        'orthogonality': 0.35,      # Walls at 90° angles (most important)
        'wall_structure': 0.20,     # Wall length/density patterns
        'color_profile': 0.15,      # Typical floor plan colors
        'line_density': 0.10,       # Appropriate line density
        'text_presence': 0.10,      # Room labels present
        'aspect_ratio': 0.10,       # Reasonable dimensions
    }

    def validate(
        self,
        image: np.ndarray,
        walls: Optional[List] = None,
        room_labels: Optional[List] = None,
        scale_confidence: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate if an image is a floor plan.

        Args:
            image: Input image (BGR format)
            walls: Detected wall segments (optional, for deeper analysis)
            room_labels: Detected room labels from OCR (optional)
            scale_confidence: Confidence from scale detection (optional)

        Returns:
            ValidationResult with validation decision and details
        """
        scores = {}
        reasons = []
        suggestions = []

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Check orthogonality (walls at right angles)
        scores['orthogonality'] = self._check_orthogonality(gray)
        if scores['orthogonality'] < 0.3:
            reasons.append("Few orthogonal (90°) lines detected")
            suggestions.append("Floor plans typically have horizontal and vertical walls")

        # 2. Check wall structure patterns
        scores['wall_structure'] = self._check_wall_structure(gray, walls)
        if scores['wall_structure'] < 0.3:
            reasons.append("Wall pattern doesn't match typical floor plans")
            suggestions.append("Floor plans have connected walls forming rooms")

        # 3. Check color profile
        scores['color_profile'] = self._check_color_profile(image)
        if scores['color_profile'] < 0.3:
            reasons.append("Color profile doesn't match floor plans")
            suggestions.append("Floor plans are typically high-contrast (black lines on white)")

        # 4. Check line density
        scores['line_density'] = self._check_line_density(gray, width, height)
        if scores['line_density'] < 0.3:
            reasons.append("Line density is unusual for floor plans")
            suggestions.append("Too few or too many lines detected")

        # 5. Check for text/labels
        scores['text_presence'] = self._check_text_presence(gray, room_labels)
        if scores['text_presence'] < 0.2:
            reasons.append("No room labels or dimension text found")
            suggestions.append("Floor plans usually have room names (Kitchen, Bedroom, etc.)")

        # 6. Check aspect ratio
        scores['aspect_ratio'] = self._check_aspect_ratio(width, height)
        if scores['aspect_ratio'] < 0.5:
            reasons.append("Unusual aspect ratio for a floor plan")

        # Boost score if scale was confidently detected
        if scale_confidence and scale_confidence > 0.5:
            scores['text_presence'] = min(1.0, scores['text_presence'] + 0.3)

        # Calculate weighted overall confidence
        overall_confidence = sum(
            scores[key] * self.WEIGHTS[key]
            for key in scores
        )

        # Determine if valid - must meet overall threshold AND minimum orthogonality
        is_valid = (
            overall_confidence >= self.MIN_OVERALL_CONFIDENCE and
            scores['orthogonality'] >= self.MIN_ORTHOGONALITY
        )

        # Special case: high text but low wall structure = probably a document, not floor plan
        if scores['text_presence'] > 0.6 and scores['wall_structure'] < 0.5 and scores['orthogonality'] < 0.6:
            is_valid = False
            reasons.append("Appears to be a text document, not a floor plan")
            suggestions.append("Floor plans have walls forming rooms, not just text")

        # Add positive reasons if valid
        if is_valid:
            if scores['orthogonality'] > 0.6:
                reasons.insert(0, "Strong orthogonal wall structure detected")
            if scores['text_presence'] > 0.5:
                reasons.insert(0, "Room labels or dimensions found")
            if scores['color_profile'] > 0.7:
                reasons.insert(0, "Typical floor plan color scheme")

        if not is_valid and not suggestions:
            suggestions.append("Please upload a clear architectural floor plan image")

        return ValidationResult(
            is_valid=is_valid,
            confidence=overall_confidence,
            scores=scores,
            reasons=reasons,
            suggestions=suggestions
        )

    def _check_orthogonality(self, gray: np.ndarray) -> float:
        """Check if lines are predominantly horizontal/vertical (orthogonal)."""
        height, width = gray.shape[:2]
        min_wall_length = min(width, height) * 0.05  # Minimum 5% of image size

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform - require longer lines for floor plans
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=50,
            minLineLength=max(50, int(min_wall_length)),  # Longer min length
            maxLineGap=10
        )

        if lines is None or len(lines) < 5:
            return 0.1  # Too few lines

        orthogonal_count = 0
        total_length = 0
        orthogonal_length = 0
        horizontal_count = 0  # Count horizontal lines
        vertical_count = 0    # Count vertical lines
        long_horizontal = 0   # Long horizontal lines
        long_vertical = 0     # Long vertical lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_length += length

            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(math.degrees(math.atan((y2-y1)/(x2-x1))))

            is_long = length > min(width, height) * 0.10

            # Check if horizontal (0° ± 10°) or vertical (90° ± 10°)
            if angle < 10:  # Horizontal
                orthogonal_count += 1
                orthogonal_length += length
                horizontal_count += 1
                if is_long:
                    long_horizontal += 1
            elif angle > 80:  # Vertical
                orthogonal_count += 1
                orthogonal_length += length
                vertical_count += 1
                if is_long:
                    long_vertical += 1

        if total_length == 0:
            return 0.1

        # Score based on proportion of orthogonal lines (weighted by length)
        ortho_ratio = orthogonal_length / total_length

        # CRITICAL: Floor plans need BOTH horizontal AND vertical lines
        # Text documents have mostly horizontal lines (text rows) with few verticals
        has_horizontal = long_horizontal >= 2
        has_vertical = long_vertical >= 2
        has_both_directions = has_horizontal and has_vertical

        if not has_both_directions:
            # Major penalty if missing vertical or horizontal walls
            ortho_ratio *= 0.3
            # Additional check: ratio of horizontal to vertical should be reasonable
            if horizontal_count > 0 and vertical_count > 0:
                h_v_ratio = horizontal_count / vertical_count
                if h_v_ratio > 5 or h_v_ratio < 0.2:
                    # Too imbalanced (like text with all horizontal lines)
                    ortho_ratio *= 0.5

        # Floor plans typically have 70%+ orthogonal lines
        if ortho_ratio > 0.8:
            return 1.0
        elif ortho_ratio > 0.6:
            return 0.8
        elif ortho_ratio > 0.4:
            return 0.5
        elif ortho_ratio > 0.2:
            return 0.3
        else:
            return 0.1

    def _check_wall_structure(self, gray: np.ndarray, walls: Optional[List]) -> float:
        """Check if detected structures look like walls."""
        height, width = gray.shape[:2]

        if walls is not None and len(walls) > 0:
            # Analyze provided walls
            if len(walls) < 4:
                return 0.2  # Too few walls for a room
            elif len(walls) > 500:
                return 0.3  # Too many - probably noise

            # Check for wall connectivity (walls should connect at endpoints)
            # This is a simplified check
            wall_count = len(walls)
            if 10 <= wall_count <= 200:
                return 0.8
            elif 4 <= wall_count < 10:
                return 0.5
            else:
                return 0.4

        # Fallback: analyze contours and check for thick lines (walls)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Check for thick lines using morphological operations
        # Floor plan walls are typically 3-10 pixels thick
        # Text lines are 1-2 pixels thick
        kernel_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(binary, kernel_thick, iterations=1)
        thick_pixel_ratio = np.count_nonzero(eroded) / np.count_nonzero(binary) if np.count_nonzero(binary) > 0 else 0

        # If very few pixels remain after erosion, lines are too thin (text-like)
        if thick_pixel_ratio < 0.3:
            return 0.2  # Lines are too thin - probably text, not walls

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0.1

        # Check for rectangular-ish contours (rooms)
        rectangular_count = 0
        large_rect_count = 0
        min_room_area = (width * height) * 0.01  # Room should be at least 1% of image

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            area = cv2.contourArea(contour)

            # Rectangles have 4 vertices
            if len(approx) == 4:
                rectangular_count += 1
                if area > min_room_area:
                    large_rect_count += 1

        # Floor plans should have at least some large rectangular regions (rooms)
        if large_rect_count >= 2:
            return 0.9
        elif large_rect_count >= 1:
            return 0.7

        rect_ratio = rectangular_count / max(len(contours), 1)

        if rect_ratio > 0.3:
            return 0.6
        elif rect_ratio > 0.1:
            return 0.4
        else:
            return 0.2

    def _check_color_profile(self, image: np.ndarray) -> float:
        """Check if colors match typical floor plan (high contrast, limited palette)."""
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Floor plans typically have:
        # - High concentration at white (background)
        # - Some concentration at black (lines)
        # - Little in between

        white_region = hist[200:256].sum()  # Near white
        black_region = hist[0:56].sum()     # Near black
        mid_region = hist[56:200].sum()     # Middle grays

        # Ideal floor plan: lots of white, some black, little middle
        if white_region > 0.5 and black_region > 0.05 and mid_region < 0.4:
            return 1.0
        elif white_region > 0.4 and black_region > 0.03:
            return 0.7
        elif white_region > 0.3:
            return 0.5
        elif mid_region > 0.7:
            # Photographic image (lots of mid-tones)
            return 0.2
        else:
            return 0.3

    def _check_line_density(self, gray: np.ndarray, width: int, height: int) -> float:
        """Check if line density is appropriate for a floor plan."""
        edges = cv2.Canny(gray, 50, 150)

        # Count edge pixels
        edge_pixels = np.count_nonzero(edges)
        total_pixels = width * height
        edge_ratio = edge_pixels / total_pixels

        # Floor plans typically have 1-10% edge pixels
        if 0.01 <= edge_ratio <= 0.10:
            return 1.0
        elif 0.005 <= edge_ratio <= 0.15:
            return 0.7
        elif edge_ratio < 0.005:
            # Too few lines (blank or photo)
            return 0.2
        else:
            # Too many lines (noise or complex image)
            return 0.3

    def _check_text_presence(self, gray: np.ndarray, room_labels: Optional[List]) -> float:
        """Check for presence of text (room labels, dimensions)."""
        if room_labels and len(room_labels) > 0:
            # OCR found room labels
            label_count = len(room_labels)
            if label_count >= 3:
                return 1.0
            elif label_count >= 1:
                return 0.7

        # Fallback: look for text-like regions using morphology
        # Text appears as small connected components
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        text_like_count = 0
        for i in range(1, num_labels):  # Skip background
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Text characters are small, roughly square-ish components
            if 5 < w < 100 and 5 < h < 100:
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect < 5:  # Not too elongated
                    text_like_count += 1

        # Floor plans typically have many text-like components
        if text_like_count > 50:
            return 0.8
        elif text_like_count > 20:
            return 0.5
        elif text_like_count > 5:
            return 0.3
        else:
            return 0.1

    def _check_aspect_ratio(self, width: int, height: int) -> float:
        """Check if aspect ratio is reasonable for a floor plan."""
        ratio = max(width, height) / max(min(width, height), 1)

        # Floor plans are typically 1:1 to 3:1 aspect ratio
        if 1.0 <= ratio <= 2.0:
            return 1.0
        elif ratio <= 3.0:
            return 0.8
        elif ratio <= 4.0:
            return 0.5
        else:
            return 0.3


def validate_floor_plan(
    image: np.ndarray,
    walls: Optional[List] = None,
    room_labels: Optional[List] = None,
    scale_confidence: Optional[float] = None
) -> ValidationResult:
    """Convenience function to validate a floor plan image."""
    validator = FloorPlanValidator()
    return validator.validate(image, walls, room_labels, scale_confidence)
