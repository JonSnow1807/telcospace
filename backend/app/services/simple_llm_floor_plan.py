"""Simple LLM-based Floor Plan Analyzer.

Pass an image to Gemini, get HTML back. That's it.
"""

import os
import json
import base64
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

import google.generativeai as genai
from google.oauth2 import service_account

from app.schemas.project import WallSegment, Point, Room, MapData, MapDimensions

logger = logging.getLogger(__name__)

# Material RF attenuation (dB)
MATERIAL_ATTENUATION = {
    "concrete": 15.0,
    "brick": 12.0,
    "wood": 6.0,
    "glass": 5.0,
    "drywall": 3.0,
    "metal": 25.0,
}


@dataclass
class FloorPlanResult:
    """Result from floor plan analysis."""
    html: str
    walls: List[WallSegment]
    rooms: List[Room]
    width_px: int  # Pixel dimensions
    height_px: int
    width_m: float  # Meter dimensions
    height_m: float
    scale: float


class SimpleFloorPlanLLM:
    """Simple LLM-based floor plan analyzer."""
    
    def __init__(self):
        """Initialize with Gemini."""
        self._configure_gemini()
        
    def _configure_gemini(self):
        """Configure Google Gemini."""
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
            raise FileNotFoundError("Google credentials not found")
            
        credentials = service_account.Credentials.from_service_account_file(
            str(creds_path),
            scopes=["https://www.googleapis.com/auth/generative-language"]
        )
        genai.configure(credentials=credentials)
        
        # Use fast model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Gemini configured for simple floor plan analysis")
        
    def _get_mime_type(self, path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(path).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(ext, "image/png")
        
    def analyze(
        self,
        image_path: str,
        user_scale: Optional[float] = None
    ) -> FloorPlanResult:
        """
        Analyze a floor plan image and return HTML representation.
        
        Args:
            image_path: Path to floor plan image
            user_scale: Optional scale in meters per pixel
            
        Returns:
            FloorPlanResult with HTML, walls, and dimensions
        """
        # Load image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Get image dimensions
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            import cv2
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            
        logger.info(f"Image size: {img_width}x{img_height}")
        
        # Simple prompt - just ask for HTML
        prompt = f"""You are a floor plan analyzer. Look at this floor plan image and generate an HTML representation of it.

IMAGE SIZE: {img_width} x {img_height} pixels

REQUIREMENTS:
1. Create an HTML div that represents this floor plan
2. Use absolute positioning for all walls
3. Walls should be div elements with background colors based on material:
   - Exterior walls (thick, boundary): #333333 (dark gray) - concrete/brick
   - Interior walls (thinner): #666666 (medium gray) - drywall
   - Glass walls/windows: #87CEEB (light blue)
4. Doors should be shown as gaps in walls or with #8B4513 (brown) color
5. Include room labels as text elements
6. Match the layout as closely as possible to the original image

ALSO PROVIDE:
- Estimated scale (meters per pixel) based on typical room sizes or any visible dimensions
- List of walls with coordinates and materials

OUTPUT FORMAT (return ONLY this JSON, no other text):
{{
    "html": "<div style='position:relative;width:{img_width}px;height:{img_height}px;background:#f5f5f5;'>... walls and rooms ...</div>",
    "scale_meters_per_pixel": 0.05,
    "walls": [
        {{"x1": 10, "y1": 10, "x2": 200, "y2": 10, "thickness": 8, "material": "concrete"}},
        {{"x1": 200, "y1": 10, "x2": 200, "y2": 150, "thickness": 4, "material": "drywall"}}
    ],
    "rooms": [
        {{"name": "Living Room", "center_x": 100, "center_y": 80}},
        {{"name": "Bedroom", "center_x": 300, "center_y": 80}}
    ]
}}

Generate the HTML representation now. Make it accurate and detailed."""

        # Call Gemini
        image_part = {
            "mime_type": self._get_mime_type(image_path),
            "data": image_base64
        }
        
        logger.info("Sending image to Gemini...")
        response = self.model.generate_content([prompt, image_part])
        
        # Parse response
        text = response.text
        logger.info(f"Gemini response: {len(text)} chars")
        
        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            logger.error(f"No JSON found in response: {text[:500]}")
            raise ValueError("LLM did not return valid JSON")
            
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Text: {json_match.group()[:500]}")
            raise ValueError(f"Failed to parse LLM response: {e}")
            
        # Extract data
        html = data.get("html", "")
        scale = user_scale or float(data.get("scale_meters_per_pixel", 0.05))
        raw_walls = data.get("walls", [])
        raw_rooms = data.get("rooms", [])
        
        # Convert walls to WallSegment objects
        walls = []
        for w in raw_walls:
            material = w.get("material", "drywall")
            attenuation = MATERIAL_ATTENUATION.get(material, 10.0)
            
            # Convert pixels to meters
            walls.append(WallSegment(
                start=Point(x=float(w["x1"]) * scale, y=float(w["y1"]) * scale),
                end=Point(x=float(w["x2"]) * scale, y=float(w["y2"]) * scale),
                thickness=float(w.get("thickness", 4)) * scale,
                material=material,
                attenuation_db=attenuation
            ))
            
        # Convert rooms
        rooms = []
        for r in raw_rooms:
            rooms.append(Room(
                name=r.get("name", "Room"),
                polygon=[],  # We don't have polygon data
                area=1.0  # Placeholder area
            ))
            
        # Calculate dimensions in meters
        width_m = img_width * scale
        height_m = img_height * scale
        
        logger.info(f"Analysis complete: {len(walls)} walls, {len(rooms)} rooms, scale={scale:.4f} m/px")
        
        return FloorPlanResult(
            html=html,
            walls=walls,
            rooms=rooms,
            width_px=img_width,
            height_px=img_height,
            width_m=width_m,
            height_m=height_m,
            scale=scale
        )
        
    def to_map_data(self, result: FloorPlanResult) -> MapData:
        """Convert result to MapData for optimization."""
        return MapData(
            dimensions=MapDimensions(width=result.width_px, height=result.height_px),
            walls=result.walls,
            rooms=result.rooms,
            forbidden_zones=[]
        )


# Singleton
_instance = None

def get_simple_llm() -> SimpleFloorPlanLLM:
    """Get or create the simple LLM analyzer."""
    global _instance
    if _instance is None:
        _instance = SimpleFloorPlanLLM()
    return _instance
