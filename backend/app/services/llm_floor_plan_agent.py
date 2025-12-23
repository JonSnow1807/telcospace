"""LLM-based Floor Plan Analysis Agent using LangGraph and Google Gemini.

This module uses Google's Gemini Vision model to analyze floor plan images
and generate structured HTML/SVG representations that can be edited in the frontend.

Architecture Metrics Used (IEEE/ASHRAE Standards):
- Exterior walls: 200-300mm (8-12 inches) - typically concrete/brick
- Interior load-bearing walls: 150-200mm (6-8 inches) - concrete/brick
- Interior partition walls: 100-150mm (4-6 inches) - drywall/wood
- Glass walls/windows: 6-12mm panels
- Door openings: 800-900mm standard, 1200mm+ for double doors
"""

import os
import json
import base64
import logging
from typing import TypedDict, List, Optional, Tuple, Annotated
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from google.oauth2 import service_account

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.schemas.project import WallSegment, Room, MapData, MapDimensions

logger = logging.getLogger(__name__)

# Standard architectural wall thicknesses (meters)
WALL_THICKNESS_STANDARDS = {
    "exterior_concrete": 0.25,      # 250mm - exterior load-bearing
    "exterior_brick": 0.23,         # 230mm - double brick
    "interior_loadbearing": 0.15,   # 150mm - interior structural
    "interior_partition": 0.10,     # 100mm - drywall partition
    "glass": 0.012,                 # 12mm - glass panel
    "unknown": 0.15                 # Default assumption
}

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


@dataclass
class FloorPlanElement:
    """Detected element in floor plan."""
    element_type: str  # 'wall', 'door', 'window'
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    thickness: float
    material: str
    confidence: float


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    image_path: str
    image_base64: str
    image_width: int
    image_height: int
    scale_meters_per_pixel: float
    
    # Analysis results
    raw_analysis: str
    elements: List[dict]
    walls: List[dict]
    doors: List[dict]
    rooms: List[dict]
    
    # HTML representation
    html_layout: str
    svg_layout: str
    
    # Final output
    map_data: Optional[dict]
    
    # Agent state
    messages: Annotated[list, add_messages]
    error: Optional[str]
    current_step: str


class GeminiFloorPlanAgent:
    """LangGraph-based agent for floor plan analysis using Google Gemini."""
    
    def __init__(self, credentials_path: str = None):
        """Initialize the agent with Google credentials.
        
        Args:
            credentials_path: Path to Google service account JSON file.
                            If None, looks for info.json in project root.
        """
        self.credentials_path = credentials_path or self._find_credentials()
        self._configure_gemini()
        self.graph = self._build_graph()
        
    def _find_credentials(self) -> str:
        """Find credentials file in standard locations."""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "info.json",  # project root
            Path(__file__).parent.parent.parent / "info.json",  # backend root
            Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")),
        ]
        
        for path in possible_paths:
            if path and path.exists():
                logger.info(f"Found credentials at: {path}")
                return str(path)
        
        raise FileNotFoundError(
            "Could not find Google credentials. Place info.json in project root "
            "or set GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )
    
    def _configure_gemini(self):
        """Configure Google Gemini with service account credentials."""
        try:
            with open(self.credentials_path, 'r') as f:
                creds_data = json.load(f)
            
            # For Gemini API, we use the API key approach or service account
            # If using service account, extract project info
            self.project_id = creds_data.get("project_id")
            
            # Configure Gemini - use API key if available, otherwise service account
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                # Use service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=["https://www.googleapis.com/auth/generative-language"]
                )
                genai.configure(credentials=credentials)
            
            # Initialize the model - use gemini-2.5-flash for good balance of speed and quality
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini 2.5 Flash model configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for floor plan analysis."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("load_image", self._load_image_node)
        workflow.add_node("analyze_layout", self._analyze_layout_node)
        workflow.add_node("extract_walls", self._extract_walls_node)
        workflow.add_node("detect_doors", self._detect_doors_node)
        workflow.add_node("generate_html", self._generate_html_node)
        workflow.add_node("build_map_data", self._build_map_data_node)
        
        # Define edges
        workflow.set_entry_point("load_image")
        workflow.add_edge("load_image", "analyze_layout")
        workflow.add_edge("analyze_layout", "extract_walls")
        workflow.add_edge("extract_walls", "detect_doors")
        workflow.add_edge("detect_doors", "generate_html")
        workflow.add_edge("generate_html", "build_map_data")
        workflow.add_edge("build_map_data", END)
        
        return workflow.compile()
    
    def _load_image_node(self, state: AgentState) -> dict:
        """Load and encode the floor plan image.
        
        Supports: PNG, JPEG, WebP, BMP, TIFF, GIF
        """
        import cv2
        from pathlib import Path
        
        image_path = state["image_path"]
        
        try:
            # Load image to get dimensions - cv2 handles most formats
            image = cv2.imread(image_path)
            
            # If cv2 fails, try PIL for additional format support
            if image is None:
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    
                    # Get dimensions before conversion
                    width, height = pil_image.size
                    
                    # Encode original file to base64
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    
                    return {
                        "image_base64": image_base64,
                        "image_width": width,
                        "image_height": height,
                        "current_step": "load_image"
                    }
                except Exception as pil_error:
                    return {"error": f"Could not load image {image_path} with cv2 or PIL: {pil_error}"}
            
            height, width = image.shape[:2]
            
            # Encode original file to base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            return {
                "image_base64": image_base64,
                "image_width": width,
                "image_height": height,
                "current_step": "load_image"
            }
            
        except Exception as e:
            return {"error": f"Error loading image: {str(e)}"}
    
    def _analyze_layout_node(self, state: AgentState) -> dict:
        """Use Gemini to analyze the floor plan layout."""
        
        if state.get("error"):
            return state
        
        analysis_prompt = """Analyze this floor plan image and identify all structural elements.

FOCUS ONLY ON:
1. WALLS - Both exterior and interior walls
2. DOORS - Door openings in walls

IGNORE: Furniture, appliances, text labels, dimensions, scale bars, furniture symbols.

For each WALL, determine:
- Start and end coordinates (as percentage of image width/height, 0-100)
- Wall type based on architectural standards:
  * EXTERIOR WALLS (building boundary): Usually 200-300mm thick, material: concrete or brick
  * INTERIOR LOAD-BEARING: Usually 150-200mm thick, material: concrete or brick  
  * INTERIOR PARTITION: Usually 100-150mm thick, material: drywall or wood
  * GLASS WALLS: Thin lines, usually near windows, material: glass

For each DOOR, identify:
- Position (center x, y as percentage)
- Width of opening
- Which wall it belongs to

OUTPUT FORMAT (JSON):
{
    "walls": [
        {
            "id": "wall_1",
            "start_x": 5,
            "start_y": 10,
            "end_x": 95,
            "end_y": 10,
            "wall_type": "exterior",
            "material": "concrete",
            "thickness_mm": 250,
            "confidence": 0.9
        }
    ],
    "doors": [
        {
            "id": "door_1",
            "center_x": 50,
            "center_y": 10,
            "width_percent": 5,
            "wall_id": "wall_1",
            "confidence": 0.85
        }
    ],
    "estimated_scale": "1 pixel = X meters (estimate based on typical room sizes)"
}

Analyze the image and return ONLY valid JSON."""

        try:
            # Determine mime type
            image_path = state["image_path"]
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif"
            }
            mime_type = mime_types.get(ext, "image/png")
            
            # Create image part for Gemini
            image_part = {
                "mime_type": mime_type,
                "data": state["image_base64"]
            }
            
            # Call Gemini
            response = self.model.generate_content([
                analysis_prompt,
                image_part
            ])
            
            raw_analysis = response.text
            logger.info(f"Gemini analysis received: {len(raw_analysis)} chars")
            
            return {
                "raw_analysis": raw_analysis,
                "current_step": "analyze_layout"
            }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _extract_walls_node(self, state: AgentState) -> dict:
        """Extract and structure wall data from Gemini's analysis."""
        
        if state.get("error"):
            return state
        
        try:
            # Parse JSON from Gemini response
            raw = state["raw_analysis"]
            
            # Try to extract JSON from the response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                # If no JSON found, try to parse the entire response
                logger.warning("No JSON brackets found, attempting full parse")
                json_str = raw
            else:
                json_str = raw[json_start:json_end]
            
            analysis_data = json.loads(json_str)
            
            walls = analysis_data.get("walls", [])
            doors = analysis_data.get("doors", [])
            
            # Convert percentage coordinates to pixels
            width = state["image_width"]
            height = state["image_height"]
            
            processed_walls = []
            for wall in walls:
                processed_wall = {
                    "id": wall.get("id", f"wall_{len(processed_walls)}"),
                    "start_x": wall["start_x"] * width / 100,
                    "start_y": wall["start_y"] * height / 100,
                    "end_x": wall["end_x"] * width / 100,
                    "end_y": wall["end_y"] * height / 100,
                    "wall_type": wall.get("wall_type", "interior_partition"),
                    "material": wall.get("material", "drywall"),
                    "thickness_mm": wall.get("thickness_mm", 150),
                    "confidence": wall.get("confidence", 0.5)
                }
                processed_walls.append(processed_wall)
            
            return {
                "walls": processed_walls,
                "doors": doors,
                "current_step": "extract_walls"
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON: {e}")
            logger.error(f"Raw response: {state['raw_analysis'][:500]}")
            return {"error": f"Failed to parse analysis: {str(e)}"}
        except Exception as e:
            return {"error": f"Wall extraction failed: {str(e)}"}
    
    def _detect_doors_node(self, state: AgentState) -> dict:
        """Process door detection and adjust walls accordingly."""
        
        if state.get("error"):
            return state
        
        # Doors are already extracted in the analysis
        # This node can be used for additional door processing if needed
        
        return {"current_step": "detect_doors"}
    
    def _generate_html_node(self, state: AgentState) -> dict:
        """Generate HTML/SVG representation of the floor plan."""
        
        if state.get("error"):
            return state
        
        width = state["image_width"]
        height = state["image_height"]
        walls = state.get("walls", [])
        doors = state.get("doors", [])
        
        # Generate SVG representation
        svg_elements = []
        
        # Add walls
        for i, wall in enumerate(walls):
            material = wall.get("material", "drywall")
            thickness = wall.get("thickness_mm", 150) / 10  # Convert to stroke width
            
            # Color based on material
            colors = {
                "concrete": "#6b7280",
                "brick": "#dc2626",
                "drywall": "#fbbf24",
                "wood": "#92400e",
                "glass": "#06b6d4",
                "metal": "#374151"
            }
            color = colors.get(material, "#9ca3af")
            
            svg_elements.append(
                f'<line id="wall_{i}" class="wall wall-{material}" '
                f'x1="{wall["start_x"]:.1f}" y1="{wall["start_y"]:.1f}" '
                f'x2="{wall["end_x"]:.1f}" y2="{wall["end_y"]:.1f}" '
                f'stroke="{color}" stroke-width="{max(2, thickness/5)}" '
                f'data-material="{material}" data-thickness="{wall["thickness_mm"]}" '
                f'data-confidence="{wall.get("confidence", 0.5):.2f}"/>'
            )
        
        # Add door markers
        for i, door in enumerate(doors):
            cx = door.get("center_x", 50) * width / 100
            cy = door.get("center_y", 50) * height / 100
            svg_elements.append(
                f'<circle id="door_{i}" class="door" '
                f'cx="{cx:.1f}" cy="{cy:.1f}" r="8" '
                f'fill="#22c55e" stroke="#166534" stroke-width="2" '
                f'data-wall-id="{door.get("wall_id", "")}"/>'
            )
        
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" 
    viewBox="0 0 {width} {height}" 
    width="{width}" height="{height}"
    class="floor-plan-svg">
    <style>
        .wall {{ cursor: pointer; transition: stroke-opacity 0.2s; }}
        .wall:hover {{ stroke-opacity: 0.7; }}
        .door {{ cursor: pointer; transition: fill-opacity 0.2s; }}
        .door:hover {{ fill-opacity: 0.7; }}
        .wall-concrete {{ stroke: #6b7280; }}
        .wall-brick {{ stroke: #dc2626; }}
        .wall-drywall {{ stroke: #fbbf24; }}
        .wall-wood {{ stroke: #92400e; }}
        .wall-glass {{ stroke: #06b6d4; }}
    </style>
    <g id="walls-layer">
        {chr(10).join(svg_elements)}
    </g>
</svg>'''
        
        # Generate HTML wrapper
        html_layout = f'''<!DOCTYPE html>
<html>
<head>
    <style>
        .floor-plan-container {{
            position: relative;
            width: {width}px;
            height: {height}px;
            background: #f3f4f6;
        }}
        .floor-plan-svg {{
            position: absolute;
            top: 0;
            left: 0;
        }}
    </style>
</head>
<body>
    <div class="floor-plan-container" data-width="{width}" data-height="{height}">
        {svg_content}
    </div>
</body>
</html>'''
        
        return {
            "html_layout": html_layout,
            "svg_layout": svg_content,
            "current_step": "generate_html"
        }
    
    def _build_map_data_node(self, state: AgentState) -> dict:
        """Build the final MapData structure for the backend."""
        
        if state.get("error"):
            return state
        
        width = state["image_width"]
        height = state["image_height"]
        walls = state.get("walls", [])
        
        # Convert to WallSegment format
        wall_segments = []
        for wall in walls:
            material = wall.get("material", "drywall")
            thickness_m = wall.get("thickness_mm", 150) / 1000  # Convert mm to meters
            
            # Use standard thickness if detected value seems off
            if thickness_m < 0.05 or thickness_m > 0.5:
                wall_type = wall.get("wall_type", "interior_partition")
                thickness_m = WALL_THICKNESS_STANDARDS.get(wall_type, 0.15)
            
            segment = {
                "start": {"x": wall["start_x"], "y": wall["start_y"]},
                "end": {"x": wall["end_x"], "y": wall["end_y"]},
                "thickness": thickness_m,
                "material": material,
                "attenuation_db": MATERIAL_ATTENUATION.get(material, 10.0)
            }
            wall_segments.append(segment)
        
        map_data = {
            "dimensions": {"width": width, "height": height},
            "walls": wall_segments,
            "rooms": [],
            "forbidden_zones": []
        }
        
        return {
            "map_data": map_data,
            "current_step": "build_map_data"
        }
    
    async def analyze(
        self,
        image_path: str,
        scale_meters_per_pixel: float = 0.05,
        progress_callback=None
    ) -> Tuple[MapData, str, str]:
        """Analyze a floor plan image and return structured data.
        
        Args:
            image_path: Path to the floor plan image
            scale_meters_per_pixel: Scale factor for conversion
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (MapData, HTML layout string, SVG layout string)
        """
        
        if progress_callback:
            progress_callback(10, "Initializing LLM analysis...")
        
        # Initialize state
        initial_state = {
            "image_path": image_path,
            "image_base64": "",
            "image_width": 0,
            "image_height": 0,
            "scale_meters_per_pixel": scale_meters_per_pixel,
            "raw_analysis": "",
            "elements": [],
            "walls": [],
            "doors": [],
            "rooms": [],
            "html_layout": "",
            "svg_layout": "",
            "map_data": None,
            "messages": [],
            "error": None,
            "current_step": "init"
        }
        
        # Run the graph
        step_progress = {
            "load_image": 20,
            "analyze_layout": 50,
            "extract_walls": 70,
            "detect_doors": 80,
            "generate_html": 90,
            "build_map_data": 100
        }
        
        final_state = initial_state
        
        # Run synchronously since Gemini SDK is synchronous
        for step in self.graph.stream(initial_state):
            for node_name, state_update in step.items():
                final_state.update(state_update)
                
                if progress_callback and node_name in step_progress:
                    progress_callback(
                        step_progress[node_name],
                        f"Processing: {node_name.replace('_', ' ').title()}"
                    )
        
        if final_state.get("error"):
            raise RuntimeError(final_state["error"])
        
        # Convert dict to MapData
        map_dict = final_state.get("map_data", {})
        map_data = MapData(
            dimensions=MapDimensions(**map_dict.get("dimensions", {"width": 800, "height": 600})),
            walls=[WallSegment(**w) for w in map_dict.get("walls", [])],
            rooms=[Room(**r) for r in map_dict.get("rooms", [])],
            forbidden_zones=[]
        )
        
        return (
            map_data,
            final_state.get("html_layout", ""),
            final_state.get("svg_layout", "")
        )
    
    def process(
        self,
        image_path: str,
        scale_meters_per_pixel: float = 0.05
    ) -> dict:
        """Synchronous method to process a floor plan image.
        
        Args:
            image_path: Path to the floor plan image
            scale_meters_per_pixel: Scale factor for conversion
            
        Returns:
            dict with map_data, html_layout, svg_layout, or error
        """
        import asyncio
        
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method
            map_data, html_layout, svg_layout = loop.run_until_complete(
                self.analyze(image_path, scale_meters_per_pixel)
            )
            
            return {
                "map_data": {
                    "dimensions": {"width": map_data.dimensions.width, "height": map_data.dimensions.height},
                    "walls": [
                        {
                            "start": {"x": w.start.x, "y": w.start.y},
                            "end": {"x": w.end.x, "y": w.end.y},
                            "thickness": w.thickness,
                            "material": w.material,
                            "attenuation_db": w.attenuation_db
                        } for w in map_data.walls
                    ],
                    "rooms": [{"name": r.name, "polygon": r.polygon, "area_m2": r.area_m2} for r in map_data.rooms],
                    "forbidden_zones": []
                },
                "html_layout": html_layout,
                "svg_layout": svg_layout,
                "error": None
            }
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return {
                "map_data": None,
                "html_layout": "",
                "svg_layout": "",
                "error": str(e)
            }


# Singleton instance
_agent_instance = None


def get_floor_plan_agent() -> GeminiFloorPlanAgent:
    """Get or create the floor plan agent singleton."""
    global _agent_instance
    
    if _agent_instance is None:
        _agent_instance = GeminiFloorPlanAgent()
    
    return _agent_instance


async def analyze_floor_plan_with_llm(
    image_path: str,
    scale: float = 0.05,
    progress_callback=None
) -> Tuple[MapData, str, str]:
    """Convenience function to analyze a floor plan.
    
    Args:
        image_path: Path to floor plan image
        scale: Meters per pixel
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (MapData, HTML layout, SVG layout)
    """
    agent = get_floor_plan_agent()
    return await agent.analyze(image_path, scale, progress_callback)
