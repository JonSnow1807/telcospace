"""Unified floor plan processing pipeline.

Orchestrates CAD parsing, OCR, scale detection, and wall detection
to process floor plans from various input formats.
"""

import os
import sys
import asyncio
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Ensure homebrew libraries are found (for cairo/SVG support on macOS)
if sys.platform == 'darwin':
    homebrew_lib = '/opt/homebrew/lib'
    if os.path.exists(homebrew_lib):
        current_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        if homebrew_lib not in current_path:
            os.environ['DYLD_LIBRARY_PATH'] = f"{homebrew_lib}:{current_path}"

import cv2
import numpy as np

from app.schemas.project import MapData, WallSegment, Room, Point, MapDimensions, ForbiddenZone
from app.services.cad_parser import CADParser, CADParseResult
from app.services.ocr_service import OCRService, RoomLabel, DimensionText, ScaleText
from app.services.scale_detection import ScaleDetector, ScaleResult, estimate_scale_from_image_size
from app.services.map_processing import (
    EnhancedWallDetector,
    segment_rooms,
    preprocess_image,
    Opening
)
from app.services.wall_detection_v2 import ImprovedWallDetector
from app.services.floor_plan_validator import FloorPlanValidator, ValidationResult

logger = logging.getLogger(__name__)

# SAM (Segment Anything Model) - pixel-perfect segmentation
try:
    from app.services.sam_wall_detector import SAMWallDetector, SAM_AVAILABLE
    logger.info(f"SAM module imported successfully, SAM_AVAILABLE={SAM_AVAILABLE}")
except (ImportError, Exception) as e:
    SAM_AVAILABLE = False
    logger.warning(f"SAM detector import failed: {e}")

# YOLOv8 model - trained on floor plans (PyTorch-based, more compatible)
try:
    from app.services.yolo_wall_detector import YOLOWallDetector, YOLO_AVAILABLE
except (ImportError, Exception) as e:
    YOLO_AVAILABLE = False
    logger.warning(f"YOLO detector not available: {e}")

# AI detection is optional - only import if available
try:
    from app.services.ai_wall_detection import AIWallDetector
    AI_DETECTION_AVAILABLE = True
except ImportError:
    AI_DETECTION_AVAILABLE = False

# Supported file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
CAD_EXTENSIONS = {'.dxf', '.dwg', '.ifc'}
VECTOR_EXTENSIONS = {'.svg', '.pdf'}


@dataclass
class ProcessingResult:
    """Result of floor plan processing."""
    map_data: MapData
    detected_scale: Optional[float] = None
    scale_confidence: Optional[float] = None
    scale_method: Optional[str] = None
    room_labels: Optional[List[RoomLabel]] = None
    dimensions: Optional[List[DimensionText]] = None
    openings: Optional[List[Opening]] = None
    warnings: Optional[List[str]] = None
    # Validation results
    is_valid_floor_plan: bool = True
    validation_confidence: Optional[float] = None
    validation_scores: Optional[dict] = None
    validation_reasons: Optional[List[str]] = None
    validation_suggestions: Optional[List[str]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class FloorPlanProcessor:
    """Unified pipeline for processing floor plans from any supported format."""

    def __init__(
        self,
        skip_validation: bool = False,
        use_ai_detection: bool = True,
        use_improved_detector: bool = True,
        use_sam: bool = True,
        use_yolo: bool = True
    ):
        self.cad_parser = CADParser()
        self.ocr_service = OCRService()
        self.scale_detector = ScaleDetector()

        # AI detection (Claude Vision) - PRIMARY detector for best accuracy
        self.use_ai_detection = use_ai_detection and AI_DETECTION_AVAILABLE
        self.ai_detector = None

        if self.use_ai_detection:
            try:
                self.ai_detector = AIWallDetector()
                logger.info("Claude Vision wall detection enabled (PRIMARY)")
            except Exception as e:
                logger.warning(f"AI detection not available: {e}")
                self.use_ai_detection = False

        # SAM - pixel-perfect segmentation (fallback if AI fails)
        logger.info(f"FloorPlanProcessor init: use_sam={use_sam}, SAM_AVAILABLE={SAM_AVAILABLE}")
        self.use_sam = use_sam and SAM_AVAILABLE and not self.use_ai_detection
        self.sam_detector = None

        if self.use_sam:
            try:
                logger.info("Attempting to initialize SAMWallDetector...")
                self.sam_detector = SAMWallDetector()
                logger.info("SAM wall detection enabled (pixel-perfect segmentation)")
            except Exception as e:
                logger.error(f"SAM initialization failed: {e}", exc_info=True)
                self.use_sam = False

        # YOLOv8 - trained specifically on floor plans (fallback)
        logger.info(f"FloorPlanProcessor init: use_yolo={use_yolo}, YOLO_AVAILABLE={YOLO_AVAILABLE}")
        self.use_yolo = use_yolo and YOLO_AVAILABLE and not self.use_ai_detection and not self.use_sam
        self.yolo_detector = None

        if self.use_yolo:
            try:
                self.yolo_detector = YOLOWallDetector()
                logger.info("YOLO wall detection enabled (specialized ML model)")
            except Exception as e:
                logger.warning(f"YOLO not available: {e}")
                self.use_yolo = False

        # Fallback to CV-based detection
        if use_improved_detector:
            self.wall_detector = ImprovedWallDetector()
        else:
            self.wall_detector = EnhancedWallDetector()

        self.validator = FloorPlanValidator()
        self.skip_validation = skip_validation

    async def process(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        user_scale: Optional[float] = None
    ) -> ProcessingResult:
        """
        Process a floor plan file (image or CAD).

        Args:
            file_path: Path to the floor plan file
            progress_callback: Optional callback for progress updates (percent, message)
            user_scale: Optional user-provided scale (meters per pixel)

        Returns:
            ProcessingResult with extracted map data and metadata
        """
        if progress_callback is None:
            progress_callback = lambda p, m: None

        # Detect file type
        file_type = self._detect_file_type(file_path)
        progress_callback(5, "File type detected")

        if file_type == 'cad':
            return await self._process_cad(file_path, progress_callback, user_scale)
        elif file_type == 'image':
            return await self._process_image(file_path, progress_callback, user_scale)
        elif file_type == 'vector':
            return await self._process_vector(file_path, progress_callback, user_scale)
        else:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

    def _detect_file_type(self, file_path: str) -> str:
        """Detect the type of floor plan file."""
        ext = Path(file_path).suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            return 'image'
        elif ext in CAD_EXTENSIONS:
            return 'cad'
        elif ext in VECTOR_EXTENSIONS:
            return 'vector'
        else:
            # Try to detect by content
            try:
                # Check if it's a valid image
                img = cv2.imread(file_path)
                if img is not None:
                    return 'image'
            except Exception:
                pass

            return 'unknown'

    async def _process_cad(
        self,
        file_path: str,
        progress_callback: Callable[[int, str], None],
        user_scale: Optional[float] = None
    ) -> ProcessingResult:
        """Process CAD file - extract geometry directly."""
        progress_callback(10, "Parsing CAD file...")
        warnings = []

        ext = Path(file_path).suffix.lower()

        try:
            if ext == '.dwg':
                progress_callback(15, "Converting DWG to DXF...")
                converted_path = self.cad_parser.convert_dwg_to_dxf(file_path)
                if converted_path:
                    file_path = converted_path
                    ext = '.dxf'
                else:
                    warnings.append("DWG conversion failed, attempting direct parse")

            progress_callback(30, "Extracting geometry...")

            if ext == '.dxf':
                cad_result = self.cad_parser.parse_dxf(file_path)
            elif ext == '.ifc':
                cad_result = self.cad_parser.parse_ifc(file_path)
            else:
                raise ValueError(f"Unsupported CAD format: {ext}")

            progress_callback(70, "Building map data...")

            # Determine scale
            if user_scale:
                scale = user_scale
                scale_confidence = 1.0
                scale_method = 'user_provided'
            elif cad_result.scale:
                scale = cad_result.scale
                scale_confidence = 0.95
                scale_method = 'cad_units'
            else:
                # Default CAD scale (assume 1 unit = 1 meter)
                scale = 1.0
                scale_confidence = 0.5
                scale_method = 'default'
                warnings.append("Could not determine scale from CAD file, using default 1:1")

            # Convert CAD result to MapData
            map_data = self._cad_result_to_map_data(cad_result, scale)

            progress_callback(100, "CAD processing complete")

            return ProcessingResult(
                map_data=map_data,
                detected_scale=scale,
                scale_confidence=scale_confidence,
                scale_method=scale_method,
                room_labels=None,  # CAD files have room names in Room objects
                dimensions=None,
                openings=None,
                warnings=warnings if warnings else None
            )

        except Exception as e:
            logger.error(f"CAD processing failed: {e}")
            raise

    async def _process_image(
        self,
        file_path: str,
        progress_callback: Callable[[int, str], None],
        user_scale: Optional[float] = None
    ) -> ProcessingResult:
        """Process image file - use CV + OCR pipeline."""
        warnings = []

        progress_callback(10, "Loading image...")
        image = cv2.imread(file_path)

        if image is None:
            raise ValueError(f"Could not load image: {file_path}")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: OCR extraction
        progress_callback(15, "Extracting text with OCR...")
        try:
            room_labels, dimensions, scale_text = self.ocr_service.extract_annotations(image)
            ocr_results = self.ocr_service.extract_all_text(image)
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            room_labels, dimensions, scale_text = [], [], None
            ocr_results = []
            warnings.append(f"OCR extraction failed: {str(e)}")

        # Step 2: Scale detection
        progress_callback(35, "Detecting scale...")
        if user_scale:
            scale = user_scale
            scale_confidence = 1.0
            scale_method = 'user_provided'
        else:
            try:
                scale_result = self.scale_detector.detect_scale(image, ocr_results)
                scale = scale_result.meters_per_pixel
                scale_confidence = scale_result.confidence
                scale_method = scale_result.method
            except Exception as e:
                logger.warning(f"Scale detection failed: {e}")
                scale = estimate_scale_from_image_size(width, height)
                scale_confidence = 0.1
                scale_method = 'image_size_estimate'
                warnings.append(f"Scale detection failed, using estimate based on image size")

        # Update wall detector with detected scale
        self.wall_detector.scale = scale

        # Step 3: Wall detection
        rooms = []
        openings = []
        detection_method = None
        walls = None

        # Priority 1: Claude Vision (AI - most accurate)
        if self.use_ai_detection and self.ai_detector:
            progress_callback(55, "Detecting walls with AI Vision...")
            try:
                logger.info(f"Calling AI detector with scale={scale}, image shape={image.shape}")
                self.ai_detector.scale = scale
                walls, ai_rooms = self.ai_detector.detect_walls(image, scale)
                logger.info(f"AI detector returned: walls={walls is not None}, len={len(walls) if walls else 0}")
                if walls is not None and len(walls) > 0:
                    if ai_rooms and not rooms:
                        rooms = ai_rooms
                    detection_method = "ai_vision"
                    logger.info(f"AI Vision detected {len(walls)} walls and {len(ai_rooms)} rooms")
                else:
                    logger.warning("AI detector returned empty walls list")
                    walls = None
            except Exception as e:
                import traceback
                logger.error(f"AI wall detection failed with exception: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                warnings.append(f"AI detection failed: {str(e)}")
                walls = None

        # Priority 2: SAM (pixel-perfect segmentation - fallback)
        if walls is None and self.use_sam and self.sam_detector:
            progress_callback(58, "Detecting walls with SAM (fallback)...")
            try:
                walls, sam_rooms = self.sam_detector.detect_walls(image, scale)
                if sam_rooms and not rooms:
                    rooms = sam_rooms
                detection_method = "sam"
                logger.info(f"SAM detected {len(walls)} walls with pixel-perfect accuracy")
            except Exception as e:
                logger.warning(f"SAM detection failed: {e}")
                warnings.append(f"SAM failed: {str(e)}")
                walls = None

        # Priority 3: YOLO (ML model trained on floor plans - fallback)
        if walls is None and self.use_yolo and self.yolo_detector:
            progress_callback(60, "Detecting walls with YOLOv8 (fallback)...")
            try:
                walls, yolo_rooms = self.yolo_detector.detect_walls(image, scale)
                if yolo_rooms and not rooms:
                    rooms = yolo_rooms
                detection_method = "yolo"
                logger.info(f"YOLO detected {len(walls)} walls")
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
                warnings.append(f"YOLO failed: {str(e)}")
                walls = None

        # Priority 4: CV-based detection (final fallback)
        if walls is None:
            progress_callback(62, "Detecting walls with CV (final fallback)...")
            try:
                walls = self.wall_detector.detect_walls(image)
                detection_method = "cv"
                logger.info(f"CV detected {len(walls)} walls")
            except Exception as e:
                logger.warning(f"CV wall detection also failed: {e}")
                walls = []
                warnings.append(f"Wall detection failed: {str(e)}")

        if detection_method:
            logger.info(f"Wall detection completed using: {detection_method}")

        # Step 4: Opening detection (doors/windows) - only if CV detection used
        if detection_method == "cv":
            progress_callback(65, "Detecting openings...")
            try:
                openings = self.wall_detector.detect_openings(image, walls)
            except Exception as e:
                logger.warning(f"Opening detection failed: {e}")
                openings = []

        # Step 5: Room segmentation - only if AI didn't provide rooms
        if not rooms:
            progress_callback(75, "Identifying rooms...")
            try:
                processed = preprocess_image(gray)
                rooms = segment_rooms(processed, walls, scale)
            except Exception as e:
                logger.warning(f"Room segmentation failed: {e}")
                rooms = []
                warnings.append(f"Room segmentation failed: {str(e)}")

        # Apply OCR room labels to detected rooms
        rooms = self._apply_room_labels(rooms, room_labels)

        # Step 6: Build MapData
        progress_callback(85, "Building map data...")
        map_data = MapData(
            dimensions=MapDimensions(width=width, height=height),
            walls=walls,
            rooms=rooms,
            forbidden_zones=[]
        )

        # Step 7: Validate floor plan
        progress_callback(92, "Validating floor plan...")
        validation_result = None
        if not self.skip_validation:
            try:
                validation_result = self.validator.validate(
                    image=image,
                    walls=walls,
                    room_labels=room_labels,
                    scale_confidence=scale_confidence
                )
                if not validation_result.is_valid:
                    warnings.append(f"Image may not be a valid floor plan (confidence: {validation_result.confidence:.0%})")
            except Exception as e:
                logger.warning(f"Validation failed: {e}")

        progress_callback(100, "Processing complete")

        return ProcessingResult(
            map_data=map_data,
            detected_scale=scale,
            scale_confidence=scale_confidence,
            scale_method=scale_method,
            room_labels=room_labels,
            dimensions=dimensions,
            openings=openings,
            warnings=warnings if warnings else None,
            is_valid_floor_plan=validation_result.is_valid if validation_result else True,
            validation_confidence=validation_result.confidence if validation_result else None,
            validation_scores=validation_result.scores if validation_result else None,
            validation_reasons=validation_result.reasons if validation_result else None,
            validation_suggestions=validation_result.suggestions if validation_result else None
        )

    async def _process_vector(
        self,
        file_path: str,
        progress_callback: Callable[[int, str], None],
        user_scale: Optional[float] = None
    ) -> ProcessingResult:
        """Process vector file (SVG, PDF) by converting to image first."""
        warnings = []

        ext = Path(file_path).suffix.lower()
        progress_callback(10, f"Converting {ext.upper()} to image...")

        try:
            if ext == '.svg':
                image = self._convert_svg_to_image(file_path)
            elif ext == '.pdf':
                image = self._convert_pdf_to_image(file_path)
            else:
                raise ValueError(f"Unsupported vector format: {ext}")

            if image is None:
                raise ValueError(f"Failed to convert {ext} to image")

            # Save temporary image and process
            temp_path = file_path + '.converted.png'
            cv2.imwrite(temp_path, image)

            try:
                result = await self._process_image(temp_path, progress_callback, user_scale)
                result.warnings = result.warnings or []
                result.warnings.insert(0, f"Converted from {ext.upper()} format")
                return result
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Vector processing failed: {e}")
            raise

    def _convert_svg_to_image(self, svg_path: str) -> Optional[np.ndarray]:
        """Convert SVG to image using cairosvg or rsvg."""
        try:
            import cairosvg
            import io
            from PIL import Image

            # Convert SVG to PNG bytes
            png_data = cairosvg.svg2png(url=svg_path, output_width=2000)

            # Load as numpy array
            pil_image = Image.open(io.BytesIO(png_data))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
            return image

        except ImportError:
            logger.warning("cairosvg not installed, trying alternative method")
        except Exception as e:
            logger.warning(f"cairosvg failed: {e}, trying alternative method")

        try:
            # Try using inkscape as fallback
            import subprocess
            temp_png = svg_path + '.temp.png'

            subprocess.run([
                'inkscape', svg_path,
                '--export-type=png',
                f'--export-filename={temp_png}',
                '--export-width=2000'
            ], check=True, capture_output=True)

            image = cv2.imread(temp_png)
            os.remove(temp_png)
            return image

        except FileNotFoundError:
            logger.warning("inkscape not installed")
        except Exception as e:
            logger.error(f"inkscape conversion failed: {e}")

        # Try rsvg-convert as another fallback
        try:
            import subprocess
            temp_png = svg_path + '.temp.png'

            subprocess.run([
                'rsvg-convert', '-w', '2000', '-o', temp_png, svg_path
            ], check=True, capture_output=True)

            image = cv2.imread(temp_png)
            os.remove(temp_png)
            return image

        except FileNotFoundError:
            logger.warning("rsvg-convert not installed")
        except Exception as e:
            logger.error(f"rsvg-convert failed: {e}")

        logger.error("All SVG conversion methods failed")
        return None

    def _convert_pdf_to_image(self, pdf_path: str) -> Optional[np.ndarray]:
        """Convert PDF to image using pdf2image or poppler."""
        try:
            from pdf2image import convert_from_path

            # Convert first page only
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=150)
            if images:
                pil_image = images[0]
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                return image

        except ImportError:
            logger.warning("pdf2image not installed")

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")

        return None

    def _cad_result_to_map_data(self, cad_result: CADParseResult, scale: float) -> MapData:
        """Convert CAD parse result to MapData format."""
        # Get bounds for dimensions
        all_points = []
        for wall in cad_result.walls:
            all_points.extend([wall.start, wall.end])

        if all_points:
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            width = int((max_x - min_x) / scale) if scale else int(max_x - min_x)
            height = int((max_y - min_y) / scale) if scale else int(max_y - min_y)
        else:
            width, height = 1000, 1000

        return MapData(
            dimensions=MapDimensions(width=max(width, 100), height=max(height, 100)),
            walls=cad_result.walls,
            rooms=cad_result.rooms,
            forbidden_zones=[]
        )

    def _apply_room_labels(
        self,
        rooms: List[Room],
        room_labels: List[RoomLabel]
    ) -> List[Room]:
        """Apply OCR-detected room labels to segmented rooms."""
        if not room_labels or not rooms:
            return rooms

        labeled_rooms = []

        for room in rooms:
            # Find room center
            if room.polygon and len(room.polygon) >= 3:
                center_x = sum(p[0] for p in room.polygon) / len(room.polygon)
                center_y = sum(p[1] for p in room.polygon) / len(room.polygon)

                # Find closest label
                best_label = None
                best_dist = float('inf')

                for label in room_labels:
                    dist = ((label.center_x - center_x) ** 2 + (label.center_y - center_y) ** 2) ** 0.5

                    # Check if label is inside room polygon (simplified check)
                    if dist < best_dist:
                        # Additional check: is the label point inside the polygon?
                        if self._point_in_polygon(label.center_x, label.center_y, room.polygon):
                            best_dist = dist
                            best_label = label

                if best_label:
                    room = Room(
                        name=best_label.name,
                        area=room.area,
                        polygon=room.polygon
                    )

            labeled_rooms.append(room)

        return labeled_rooms

    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """Check if a point is inside a polygon using ray casting."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 0.0001) + xi):
                inside = not inside

            j = i

        return inside


def validate_floor_plan_image(file_path: str) -> ValidationResult:
    """
    Quick validation to check if an image is likely a floor plan.

    This is a lightweight check that can be run before full processing
    to reject obviously invalid uploads early.

    Args:
        file_path: Path to the image file

    Returns:
        ValidationResult with validation decision and details
    """
    image = cv2.imread(file_path)
    if image is None:
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            scores={},
            reasons=["Could not load image file"],
            suggestions=["Please upload a valid image file (PNG, JPG, etc.)"]
        )

    validator = FloorPlanValidator()
    return validator.validate(image)


async def process_floor_plan(
    file_path: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    user_scale: Optional[float] = None,
    validate_first: bool = False
) -> ProcessingResult:
    """
    Convenience function to process a floor plan.

    Args:
        file_path: Path to the floor plan file
        progress_callback: Optional progress callback
        user_scale: Optional user-provided scale
        validate_first: If True, run quick validation before processing
                       and raise ValueError if invalid

    Returns:
        ProcessingResult with extracted data

    Raises:
        ValueError: If validate_first=True and image is not a valid floor plan
    """
    if validate_first:
        validation = validate_floor_plan_image(file_path)
        if not validation.is_valid:
            suggestions = "; ".join(validation.suggestions) if validation.suggestions else "Upload a valid floor plan"
            raise ValueError(f"Image does not appear to be a floor plan: {suggestions}")

    processor = FloorPlanProcessor()
    return await processor.process(file_path, progress_callback, user_scale)
