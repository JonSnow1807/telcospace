"""Hybrid YOLO + LLM Floor Plan Detector.

Combines:
1. YOLO for fast, accurate structural detection (walls, doors, windows)
2. LLM (Gemini) for intelligent scale estimation and material classification

This provides the best of both worlds:
- YOLO: Precise geometric detection, works with any image format
- LLM: Smart scale inference from room context, material classification
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.project import WallSegment, Point, Room, MapData, MapDimensions

logger = logging.getLogger(__name__)

# Material RF attenuation (dB) - IEEE 802.11 standards
MATERIAL_ATTENUATION = {
    "concrete": 15.0,
    "brick": 12.0,
    "wood": 6.0,
    "glass": 5.0,
    "drywall": 3.0,
    "metal": 25.0,
    "unknown": 10.0
}

# Standard wall thicknesses (meters)
WALL_THICKNESS = {
    "exterior": 0.25,      # 250mm
    "loadbearing": 0.15,   # 150mm
    "partition": 0.10,     # 100mm
    "glass": 0.012,        # 12mm
}


@dataclass
class DetectionResult:
    """Result from hybrid detection."""
    walls: List[WallSegment]
    rooms: List[Room]
    doors: List[Dict]
    windows: List[Dict]
    scale_meters_per_pixel: float
    scale_confidence: float
    scale_method: str
    image_width: int
    image_height: int
    

class HybridYoloLlmDetector:
    """Hybrid detector combining YOLO and LLM for floor plan analysis."""
    
    def __init__(self):
        """Initialize both YOLO and LLM components."""
        self.yolo_detector = None
        self.llm_model = None
        self.yolo_available = False
        self.llm_available = False
        
        self._init_yolo()
        self._init_llm()
        
    def _init_yolo(self):
        """Initialize YOLO detector."""
        try:
            from app.services.yolo_wall_detector import YOLOWallDetector, YOLO_AVAILABLE
            if YOLO_AVAILABLE:
                self.yolo_detector = YOLOWallDetector()
                self.yolo_available = True
                logger.info("YOLO detector initialized for hybrid detection")
        except Exception as e:
            logger.warning(f"YOLO not available for hybrid: {e}")
            self.yolo_available = False
            
    def _init_llm(self):
        """Initialize LLM (Gemini) for intelligent analysis."""
        try:
            import google.generativeai as genai
            from google.oauth2 import service_account
            
            # Find credentials
            creds_paths = [
                Path("/app/info.json"),
                Path(__file__).parent.parent.parent / "info.json",
                Path(os.environ.get("GOOGLE_CREDENTIALS_PATH", "")),
            ]
            
            creds_path = None
            for p in creds_paths:
                if p and p.exists():
                    creds_path = p
                    break
                    
            if not creds_path:
                logger.warning("Google credentials not found for LLM")
                return
                
            credentials = service_account.Credentials.from_service_account_file(
                str(creds_path),
                scopes=["https://www.googleapis.com/auth/generative-language"]
            )
            genai.configure(credentials=credentials)
            
            # Use Gemini 2.5 Flash for speed (good balance of speed and quality)
            self.llm_model = genai.GenerativeModel('gemini-2.5-flash')
            self.llm_available = True
            logger.info("LLM (Gemini 2.5 Flash) initialized for hybrid detection")
            
        except Exception as e:
            logger.warning(f"LLM not available for hybrid: {e}")
            self.llm_available = False
            
    def _load_image(self, image_path: str) -> Tuple[np.ndarray, int, int]:
        """Load image from any supported format.
        
        Supports: PNG, JPEG, WebP, BMP, TIFF
        """
        # Read with cv2 (handles most formats)
        image = cv2.imread(image_path)
        
        if image is None:
            # Try PIL for additional format support
            try:
                from PIL import Image
                pil_image = Image.open(image_path)
                
                # Convert to RGB/BGR numpy array
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode not in ['RGB', 'L']:
                    pil_image = pil_image.convert('RGB')
                    
                image = np.array(pil_image)
                
                # Convert RGB to BGR for cv2 compatibility
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
            except Exception as e:
                raise ValueError(f"Could not load image {image_path}: {e}")
                
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        height, width = image.shape[:2]
        return image, width, height
        
    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        return mime_types.get(ext, "image/png")
        
    def _estimate_scale_with_llm(
        self, 
        image_path: str,
        image_width: int,
        image_height: int,
        detected_elements: Dict = None
    ) -> Tuple[float, float, str]:
        """Use LLM to intelligently estimate the scale.
        
        Returns:
            Tuple of (scale_meters_per_pixel, confidence, method_description)
        """
        if not self.llm_available:
            # Fallback: assume typical room size
            typical_room_width_m = 4.0  # 4 meters typical room
            assumed_room_pixels = min(image_width, image_height) / 3
            scale = typical_room_width_m / assumed_room_pixels
            return scale, 0.3, "fallback_typical_room"
            
        try:
            # Encode image
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                
            mime_type = self._get_mime_type(image_path)
            
            prompt = """Analyze this floor plan image to estimate its SCALE (meters per pixel).

Look for these clues:
1. SCALE BAR: If there's a scale bar (e.g., "1m", "5ft"), use it
2. DIMENSION LABELS: Any text showing measurements (e.g., "3.5m", "12ft")
3. STANDARD SIZES: 
   - Standard doors are 0.8-0.9m wide
   - Standard rooms: bedrooms 3-4m, bathrooms 2-3m, living rooms 4-6m
   - Standard corridors: 0.9-1.2m wide

Image dimensions: {width}x{height} pixels

Return ONLY a JSON object:
{{
    "scale_meters_per_pixel": 0.05,
    "confidence": 0.8,
    "reasoning": "Based on door width appearing to be ~20 pixels, standard door is 0.9m"
}}

Provide your best estimate even if uncertain.""".format(width=image_width, height=image_height)

            image_part = {
                "mime_type": mime_type,
                "data": image_base64
            }
            
            response = self.llm_model.generate_content([prompt, image_part])
            
            # Parse response
            text = response.text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                result = json.loads(text[json_start:json_end])
                scale = float(result.get("scale_meters_per_pixel", 0.05))
                confidence = float(result.get("confidence", 0.5))
                reasoning = result.get("reasoning", "llm_estimated")
                
                # Sanity check scale (0.001 to 0.5 m/px is reasonable)
                if 0.001 <= scale <= 0.5:
                    return scale, confidence, f"llm: {reasoning}"
                    
        except Exception as e:
            logger.warning(f"LLM scale estimation failed: {e}")
            
        # Fallback
        typical_room_width_m = 4.0
        assumed_room_pixels = min(image_width, image_height) / 3
        scale = typical_room_width_m / assumed_room_pixels
        return scale, 0.3, "fallback_after_llm_error"
        
    def _classify_wall_materials_with_llm(
        self,
        image_path: str,
        walls: List[Dict]
    ) -> List[Dict]:
        """Use LLM to classify wall materials."""
        if not self.llm_available or not walls:
            # Default classification based on position/thickness
            for wall in walls:
                if wall.get("is_exterior", False):
                    wall["material"] = "concrete"
                    wall["thickness"] = WALL_THICKNESS["exterior"]
                else:
                    wall["material"] = "drywall"
                    wall["thickness"] = WALL_THICKNESS["partition"]
            return walls
            
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                
            mime_type = self._get_mime_type(image_path)
            
            # Describe walls for LLM
            wall_desc = "\n".join([
                f"Wall {i}: ({w.get('start_x', 0):.0f},{w.get('start_y', 0):.0f}) to ({w.get('end_x', 0):.0f},{w.get('end_y', 0):.0f})"
                for i, w in enumerate(walls[:20])  # Limit to 20 walls
            ])
            
            prompt = f"""Analyze this floor plan and classify wall materials.

Detected walls:
{wall_desc}

For EACH wall, determine:
1. Is it exterior (building boundary) or interior?
2. Material: concrete, brick, drywall, glass, wood
3. Thickness category: exterior (250mm), loadbearing (150mm), partition (100mm), glass (12mm)

Return JSON array with wall classifications:
[
    {{"wall_index": 0, "is_exterior": true, "material": "concrete", "thickness_category": "exterior"}},
    ...
]

Only return the JSON array, no other text."""

            image_part = {
                "mime_type": mime_type,
                "data": image_base64
            }
            
            response = self.llm_model.generate_content([prompt, image_part])
            
            text = response.text
            json_start = text.find("[")
            json_end = text.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                classifications = json.loads(text[json_start:json_end])
                
                for cls in classifications:
                    idx = cls.get("wall_index", -1)
                    if 0 <= idx < len(walls):
                        walls[idx]["material"] = cls.get("material", "drywall")
                        walls[idx]["is_exterior"] = cls.get("is_exterior", False)
                        thickness_cat = cls.get("thickness_category", "partition")
                        walls[idx]["thickness"] = WALL_THICKNESS.get(thickness_cat, 0.10)
                        
        except Exception as e:
            logger.warning(f"LLM wall classification failed: {e}")
            # Apply defaults
            for wall in walls:
                wall["material"] = wall.get("material", "drywall")
                wall["thickness"] = wall.get("thickness", 0.10)
                
        return walls
        
    def detect(
        self,
        image_path: str,
        user_scale: Optional[float] = None,
        progress_callback=None
    ) -> DetectionResult:
        """
        Run hybrid detection on a floor plan image.
        
        Args:
            image_path: Path to floor plan image (any format)
            user_scale: Optional user-provided scale override
            progress_callback: Optional callback(percent, message)
            
        Returns:
            DetectionResult with walls, rooms, scale info
        """
        if progress_callback:
            progress_callback(5, "Loading image...")
            
        # Load image (handles all formats)
        image, width, height = self._load_image(image_path)
        
        if progress_callback:
            progress_callback(10, "Estimating scale...")
            
        # Step 1: Estimate scale with LLM (or fallback)
        if user_scale and user_scale > 0:
            scale = user_scale
            scale_confidence = 1.0
            scale_method = "user_provided"
        else:
            scale, scale_confidence, scale_method = self._estimate_scale_with_llm(
                image_path, width, height
            )
            
        logger.info(f"Scale: {scale:.4f} m/px, confidence: {scale_confidence:.2f}, method: {scale_method}")
        
        if progress_callback:
            progress_callback(30, "Detecting structures...")
            
        # Step 2: Detect walls with YOLO or fallback to CV
        raw_walls = []
        rooms = []
        doors = []
        windows = []
        
        if self.yolo_available:
            try:
                walls_yolo, rooms_yolo = self.yolo_detector.detect_walls(image, scale)
                raw_walls = [
                    {
                        "start_x": w.start.x / scale,  # Convert back to pixels for classification
                        "start_y": w.start.y / scale,
                        "end_x": w.end.x / scale,
                        "end_y": w.end.y / scale,
                        "material": w.material,
                        "thickness": w.thickness,
                    }
                    for w in walls_yolo
                ]
                rooms = rooms_yolo
                logger.info(f"YOLO detected {len(raw_walls)} walls, {len(rooms)} rooms")
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
                
        # Fallback: Use CV-based detection if YOLO failed
        if not raw_walls:
            raw_walls = self._detect_walls_cv(image, scale)
            logger.info(f"CV fallback detected {len(raw_walls)} walls")
            
        if progress_callback:
            progress_callback(60, "Classifying materials...")
            
        # Step 3: Classify materials with LLM
        classified_walls = self._classify_wall_materials_with_llm(image_path, raw_walls)
        
        if progress_callback:
            progress_callback(80, "Building wall segments...")
            
        # Step 4: Convert to WallSegment objects
        wall_segments = []
        for w in classified_walls:
            material = w.get("material", "drywall")
            attenuation = MATERIAL_ATTENUATION.get(material, 10.0)
            
            segment = WallSegment(
                start=Point(
                    x=float(w.get("start_x", 0)) * scale,
                    y=float(w.get("start_y", 0)) * scale
                ),
                end=Point(
                    x=float(w.get("end_x", 0)) * scale,
                    y=float(w.get("end_y", 0)) * scale
                ),
                thickness=float(w.get("thickness", 0.10)),
                material=material,
                attenuation_db=attenuation
            )
            wall_segments.append(segment)
            
        if progress_callback:
            progress_callback(100, "Detection complete")
            
        return DetectionResult(
            walls=wall_segments,
            rooms=rooms,
            doors=doors,
            windows=windows,
            scale_meters_per_pixel=scale,
            scale_confidence=scale_confidence,
            scale_method=scale_method,
            image_width=width,
            image_height=height
        )
        
    def _detect_walls_cv(self, image: np.ndarray, scale: float) -> List[Dict]:
        """Fallback CV-based wall detection using edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Filter out very short lines
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < 20:
                    continue
                    
                walls.append({
                    "start_x": float(x1),
                    "start_y": float(y1),
                    "end_x": float(x2),
                    "end_y": float(y2),
                    "material": "drywall",
                    "thickness": 0.10,
                })
                
        return walls
        
    def to_map_data(self, result: DetectionResult) -> MapData:
        """Convert DetectionResult to MapData for optimization."""
        return MapData(
            dimensions=MapDimensions(
                width=result.image_width * result.scale_meters_per_pixel,
                height=result.image_height * result.scale_meters_per_pixel
            ),
            walls=result.walls,
            rooms=result.rooms,
            forbidden_zones=[]
        )


# Global instance
_hybrid_detector = None


def get_hybrid_detector() -> HybridYoloLlmDetector:
    """Get or create the hybrid detector singleton."""
    global _hybrid_detector
    if _hybrid_detector is None:
        _hybrid_detector = HybridYoloLlmDetector()
    return _hybrid_detector
