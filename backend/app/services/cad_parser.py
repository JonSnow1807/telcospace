"""CAD file parsing service for DXF, DWG, and IFC files."""

import os
import tempfile
import subprocess
import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

import ezdxf
from ezdxf.entities import Line, LWPolyline, Polyline, MText, Text

from app.schemas.project import MapData, MapDimensions, WallSegment, Room, Point, ForbiddenZone

logger = logging.getLogger(__name__)

# Material attenuation database (dB)
MATERIAL_ATTENUATION = {
    'concrete': 15.0,
    'brick': 12.0,
    'wood': 6.0,
    'glass': 5.0,
    'drywall': 3.0,
    'metal': 25.0,
    'unknown': 10.0
}

# Common wall layer names in CAD files
WALL_LAYER_PATTERNS = [
    'wall', 'walls', 'a-wall', 'a_wall', 'arch_wall',
    'mur', 'wand', 'parede', 'pared',  # International
    'boundary', 'structure', 'partition'
]

# Common door layer names
DOOR_LAYER_PATTERNS = [
    'door', 'doors', 'a-door', 'a_door', 'arch_door',
    'porte', 'tür', 'puerta', 'porta'
]

# Common room/text layer names
ROOM_LAYER_PATTERNS = [
    'room', 'rooms', 'a-room', 'room-name', 'room_name',
    'space', 'text', 'annotation', 'label'
]


@dataclass
class TextAnnotation:
    """Extracted text annotation from CAD file."""
    text: str
    x: float
    y: float
    height: float
    rotation: float = 0.0


@dataclass
class DoorOpening:
    """Door opening detected in CAD file."""
    start: Point
    end: Point
    width: float


@dataclass
class CADParseResult:
    """Result of parsing a CAD file."""
    walls: List[WallSegment]
    rooms: List[Room]
    scale: Optional[float] = None  # Detected scale (meters per unit)
    text_annotations: Optional[List[TextAnnotation]] = None
    doors: Optional[List[DoorOpening]] = None

    def __post_init__(self):
        if self.text_annotations is None:
            self.text_annotations = []
        if self.doors is None:
            self.doors = []


class DXFParser:
    """Parse DXF files using ezdxf library."""

    def __init__(self, default_wall_thickness: float = 0.2):
        self.default_wall_thickness = default_wall_thickness
        self.scale_factor = 1.0  # Will be determined from drawing units

    def parse(self, file_path: str) -> MapData:
        """
        Parse DXF file and extract floor plan data.

        Args:
            file_path: Path to DXF file

        Returns:
            MapData with walls, rooms, and dimensions
        """
        try:
            doc = ezdxf.readfile(file_path)
        except Exception as e:
            logger.error(f"Failed to read DXF file: {e}")
            raise ValueError(f"Invalid DXF file: {e}")

        msp = doc.modelspace()

        # Determine scale from drawing units
        self._detect_scale(doc)

        # Extract entities
        walls = self._extract_walls(msp)
        doors = self._extract_doors(msp)
        text_annotations = self._extract_text(msp)

        # Calculate bounding box for dimensions
        all_points = []
        for wall in walls:
            all_points.extend([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])

        if all_points:
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)

            # Normalize coordinates to start at 0
            offset_x = min_x
            offset_y = min_y

            for wall in walls:
                wall.start.x -= offset_x
                wall.start.y -= offset_y
                wall.end.x -= offset_x
                wall.end.y -= offset_y

            width = int(max_x - min_x)
            height = int(max_y - min_y)
        else:
            width = 800
            height = 600

        # Create rooms from text annotations
        rooms = self._create_rooms_from_text(text_annotations, walls, offset_x if all_points else 0, offset_y if all_points else 0)

        return MapData(
            dimensions=MapDimensions(width=max(width, 100), height=max(height, 100)),
            walls=walls,
            rooms=rooms,
            forbidden_zones=[]
        )

    def _detect_scale(self, doc):
        """Detect scale factor from DXF drawing units."""
        # Try to get units from header
        try:
            units = doc.header.get('$INSUNITS', 0)
            # Common unit codes: 0=Unitless, 1=Inches, 2=Feet, 4=Millimeters, 5=Centimeters, 6=Meters
            unit_scales = {
                0: 1.0,       # Unitless - assume meters
                1: 0.0254,    # Inches to meters
                2: 0.3048,    # Feet to meters
                4: 0.001,     # Millimeters to meters
                5: 0.01,      # Centimeters to meters
                6: 1.0,       # Meters
            }
            self.scale_factor = unit_scales.get(units, 1.0)
        except Exception:
            self.scale_factor = 1.0

    def _extract_walls(self, msp) -> List[WallSegment]:
        """Extract wall segments from modelspace."""
        walls = []

        for entity in msp:
            layer_name = entity.dxf.layer.lower() if hasattr(entity.dxf, 'layer') else ''

            # Check if entity is on a wall layer
            is_wall_layer = any(pattern in layer_name for pattern in WALL_LAYER_PATTERNS)

            if isinstance(entity, Line):
                wall = self._line_to_wall(entity, is_wall_layer)
                if wall:
                    walls.append(wall)

            elif isinstance(entity, (LWPolyline, Polyline)):
                polyline_walls = self._polyline_to_walls(entity, is_wall_layer)
                walls.extend(polyline_walls)

        # Filter out very short segments (noise)
        min_length = 0.1 * self.scale_factor  # 10cm minimum
        walls = [w for w in walls if self._wall_length(w) >= min_length]

        # Merge collinear walls
        walls = self._merge_collinear_walls(walls)

        return walls

    def _line_to_wall(self, line: Line, is_wall_layer: bool) -> Optional[WallSegment]:
        """Convert LINE entity to WallSegment."""
        start = line.dxf.start
        end = line.dxf.end

        # Only include if on wall layer or if it's a significant line
        length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
        if not is_wall_layer and length < 1.0:  # Skip short lines not on wall layers
            return None

        return WallSegment(
            start=Point(x=float(start.x), y=float(start.y)),
            end=Point(x=float(end.x), y=float(end.y)),
            thickness=self.default_wall_thickness,
            material='concrete',
            attenuation_db=MATERIAL_ATTENUATION['concrete']
        )

    def _polyline_to_walls(self, polyline, is_wall_layer: bool) -> List[WallSegment]:
        """Convert POLYLINE/LWPOLYLINE to list of WallSegments."""
        walls = []

        try:
            if isinstance(polyline, LWPolyline):
                points = list(polyline.get_points('xy'))
            else:
                points = [(v.dxf.location.x, v.dxf.location.y) for v in polyline.vertices]
        except Exception:
            return walls

        if len(points) < 2:
            return walls

        # Check if closed polyline
        is_closed = polyline.closed if hasattr(polyline, 'closed') else False

        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            wall = WallSegment(
                start=Point(x=float(start[0]), y=float(start[1])),
                end=Point(x=float(end[0]), y=float(end[1])),
                thickness=self.default_wall_thickness,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )
            walls.append(wall)

        # Close the polyline if needed
        if is_closed and len(points) >= 2:
            wall = WallSegment(
                start=Point(x=float(points[-1][0]), y=float(points[-1][1])),
                end=Point(x=float(points[0][0]), y=float(points[0][1])),
                thickness=self.default_wall_thickness,
                material='concrete',
                attenuation_db=MATERIAL_ATTENUATION['concrete']
            )
            walls.append(wall)

        return walls

    def _extract_doors(self, msp) -> List[DoorOpening]:
        """Extract door openings from modelspace."""
        doors = []

        for entity in msp:
            layer_name = entity.dxf.layer.lower() if hasattr(entity.dxf, 'layer') else ''

            if any(pattern in layer_name for pattern in DOOR_LAYER_PATTERNS):
                # Try to get door bounds
                if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'insert'):
                    # Block reference - door symbol
                    insert = entity.dxf.insert
                    # Estimate door width (standard ~0.9m)
                    doors.append(DoorOpening(
                        start=Point(x=float(insert.x), y=float(insert.y)),
                        end=Point(x=float(insert.x + 0.9), y=float(insert.y)),
                        width=0.9
                    ))

        return doors

    def _extract_text(self, msp) -> List[TextAnnotation]:
        """Extract text annotations from modelspace."""
        annotations = []

        for entity in msp:
            try:
                if isinstance(entity, (Text, MText)):
                    if isinstance(entity, MText):
                        text = entity.plain_text()
                        insert = entity.dxf.insert
                        height = entity.dxf.char_height
                        rotation = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0
                    else:
                        text = entity.dxf.text
                        insert = entity.dxf.insert
                        height = entity.dxf.height
                        rotation = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0

                    if text.strip():
                        annotations.append(TextAnnotation(
                            text=text.strip(),
                            x=float(insert.x),
                            y=float(insert.y),
                            height=float(height),
                            rotation=float(rotation)
                        ))
            except Exception as e:
                logger.debug(f"Failed to extract text from entity: {e}")
                continue

        return annotations

    def _create_rooms_from_text(
        self,
        annotations: List[TextAnnotation],
        walls: List[WallSegment],
        offset_x: float,
        offset_y: float
    ) -> List[Room]:
        """Create room definitions from text annotations."""
        rooms = []

        # Common room name patterns
        room_patterns = [
            'kitchen', 'bedroom', 'bathroom', 'living', 'dining',
            'office', 'closet', 'hall', 'garage', 'laundry',
            'master', 'guest', 'study', 'den', 'foyer', 'entry',
            'utility', 'storage', 'patio', 'balcony', 'terrace'
        ]

        for annotation in annotations:
            text_lower = annotation.text.lower()

            # Check if this looks like a room name
            if any(pattern in text_lower for pattern in room_patterns):
                # Create a simple room with estimated area
                rooms.append(Room(
                    name=annotation.text,
                    area=20.0,  # Default area - will be updated if polygon is computed
                    polygon=[
                        [annotation.x - offset_x - 50, annotation.y - offset_y - 50],
                        [annotation.x - offset_x + 50, annotation.y - offset_y - 50],
                        [annotation.x - offset_x + 50, annotation.y - offset_y + 50],
                        [annotation.x - offset_x - 50, annotation.y - offset_y + 50]
                    ]
                ))

        return rooms

    def _wall_length(self, wall: WallSegment) -> float:
        """Calculate wall segment length."""
        return math.sqrt(
            (wall.end.x - wall.start.x)**2 +
            (wall.end.y - wall.start.y)**2
        )

    def _merge_collinear_walls(self, walls: List[WallSegment], tolerance: float = 5.0) -> List[WallSegment]:
        """Merge walls that are collinear and close together."""
        if len(walls) <= 1:
            return walls

        merged = []
        used = set()

        for i, wall1 in enumerate(walls):
            if i in used:
                continue

            current = wall1
            used.add(i)

            # Try to merge with other walls
            for j, wall2 in enumerate(walls):
                if j in used or j == i:
                    continue

                if self._are_collinear(current, wall2, tolerance):
                    current = self._merge_two_walls(current, wall2)
                    used.add(j)

            merged.append(current)

        return merged

    def _are_collinear(self, wall1: WallSegment, wall2: WallSegment, tolerance: float) -> bool:
        """Check if two walls are collinear."""
        # Calculate angles
        angle1 = math.atan2(wall1.end.y - wall1.start.y, wall1.end.x - wall1.start.x)
        angle2 = math.atan2(wall2.end.y - wall2.start.y, wall2.end.x - wall2.start.x)

        # Normalize angles
        angle_diff = abs(angle1 - angle2)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # Check if angles are similar (within ~5 degrees)
        if angle_diff > 0.1 and abs(angle_diff - math.pi) > 0.1:
            return False

        # Check if walls are close to each other
        # Find closest points between the two walls
        min_dist = min(
            self._point_to_segment_distance(wall2.start, wall1),
            self._point_to_segment_distance(wall2.end, wall1),
            self._point_to_segment_distance(wall1.start, wall2),
            self._point_to_segment_distance(wall1.end, wall2)
        )

        return min_dist <= tolerance

    def _point_to_segment_distance(self, point: Point, wall: WallSegment) -> float:
        """Calculate distance from point to line segment."""
        x, y = point.x, point.y
        x1, y1 = wall.start.x, wall.start.y
        x2, y2 = wall.end.x, wall.end.y

        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)

        # Parameter t for closest point on line
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))

        # Closest point
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.sqrt((x - closest_x)**2 + (y - closest_y)**2)

    def _merge_two_walls(self, wall1: WallSegment, wall2: WallSegment) -> WallSegment:
        """Merge two collinear walls into one."""
        # Find the extreme points
        points = [
            (wall1.start.x, wall1.start.y),
            (wall1.end.x, wall1.end.y),
            (wall2.start.x, wall2.start.y),
            (wall2.end.x, wall2.end.y)
        ]

        # For horizontal/vertical walls, use min/max
        angle = math.atan2(wall1.end.y - wall1.start.y, wall1.end.x - wall1.start.x)

        if abs(math.cos(angle)) > abs(math.sin(angle)):
            # More horizontal - sort by x
            points.sort(key=lambda p: p[0])
        else:
            # More vertical - sort by y
            points.sort(key=lambda p: p[1])

        return WallSegment(
            start=Point(x=points[0][0], y=points[0][1]),
            end=Point(x=points[-1][0], y=points[-1][1]),
            thickness=wall1.thickness,
            material=wall1.material,
            attenuation_db=wall1.attenuation_db
        )


class DWGConverter:
    """Convert DWG files to DXF using ODA File Converter."""

    def __init__(self, oda_path: Optional[str] = None):
        """
        Initialize DWG converter.

        Args:
            oda_path: Path to ODA File Converter executable.
                      If not provided, will search common locations.
        """
        self.oda_path = oda_path or self._find_oda_converter()

    def _find_oda_converter(self) -> Optional[str]:
        """Find ODA File Converter in common locations."""
        common_paths = [
            '/usr/local/bin/ODAFileConverter',
            '/opt/ODAFileConverter/ODAFileConverter',
            'C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe',
            '/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter'
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'ODAFileConverter'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def is_available(self) -> bool:
        """Check if ODA File Converter is available."""
        return self.oda_path is not None and os.path.exists(self.oda_path)

    def convert_to_dxf(self, dwg_path: str) -> str:
        """
        Convert DWG file to DXF.

        Args:
            dwg_path: Path to input DWG file

        Returns:
            Path to converted DXF file

        Raises:
            RuntimeError: If conversion fails or ODA not available
        """
        if not self.is_available():
            raise RuntimeError(
                "ODA File Converter not found. Please install from: "
                "https://www.opendesign.com/guestfiles/oda_file_converter"
            )

        if not os.path.exists(dwg_path):
            raise FileNotFoundError(f"DWG file not found: {dwg_path}")

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.dirname(dwg_path)
            input_file = os.path.basename(dwg_path)
            output_file = input_file.replace('.dwg', '.dxf').replace('.DWG', '.dxf')

            # Run ODA File Converter
            # Arguments: input_folder output_folder output_version output_type recurse audit
            cmd = [
                self.oda_path,
                input_dir,
                temp_dir,
                'ACAD2018',  # Output version
                'DXF',       # Output type
                '0',         # Don't recurse
                '1'          # Audit
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                output_path = os.path.join(temp_dir, output_file)

                if os.path.exists(output_path):
                    # Copy to permanent location
                    final_path = dwg_path.replace('.dwg', '.dxf').replace('.DWG', '.dxf')
                    import shutil
                    shutil.copy2(output_path, final_path)
                    return final_path
                else:
                    raise RuntimeError(f"Conversion failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("DWG conversion timed out")
            except Exception as e:
                raise RuntimeError(f"DWG conversion failed: {e}")


class IFCParser:
    """Parse IFC (BIM) files using ifcopenshell."""

    def __init__(self):
        self._ifcopenshell = None

    def _get_ifcopenshell(self):
        """Lazy import of ifcopenshell."""
        if self._ifcopenshell is None:
            try:
                import ifcopenshell
                self._ifcopenshell = ifcopenshell
            except ImportError:
                raise ImportError(
                    "ifcopenshell is required for IFC parsing. "
                    "Install with: pip install ifcopenshell"
                )
        return self._ifcopenshell

    def is_available(self) -> bool:
        """Check if ifcopenshell is available."""
        try:
            self._get_ifcopenshell()
            return True
        except ImportError:
            return False

    def parse(self, file_path: str) -> MapData:
        """
        Parse IFC file and extract floor plan data.

        Args:
            file_path: Path to IFC file

        Returns:
            MapData with walls, rooms, and dimensions
        """
        ifc = self._get_ifcopenshell()

        try:
            model = ifc.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open IFC file: {e}")

        walls = self._extract_walls(model)
        rooms = self._extract_spaces(model)

        # Calculate bounding box
        all_points = []
        for wall in walls:
            all_points.extend([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])

        if all_points:
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)

            # Normalize coordinates
            for wall in walls:
                wall.start.x -= min_x
                wall.start.y -= min_y
                wall.end.x -= min_x
                wall.end.y -= min_y

            width = int(max_x - min_x)
            height = int(max_y - min_y)
        else:
            width = 800
            height = 600

        return MapData(
            dimensions=MapDimensions(width=max(width, 100), height=max(height, 100)),
            walls=walls,
            rooms=rooms,
            forbidden_zones=[]
        )

    def _extract_walls(self, model) -> List[WallSegment]:
        """Extract wall elements from IFC model."""
        walls = []

        try:
            for wall in model.by_type('IfcWall'):
                wall_segment = self._wall_to_segment(wall)
                if wall_segment:
                    walls.append(wall_segment)
        except Exception as e:
            logger.warning(f"Failed to extract walls from IFC: {e}")

        return walls

    def _wall_to_segment(self, wall) -> Optional[WallSegment]:
        """Convert IFC wall to WallSegment."""
        try:
            # Get wall placement
            placement = wall.ObjectPlacement
            if not placement:
                return None

            # Get local placement coordinates
            local_placement = placement.RelativePlacement
            if not local_placement:
                return None

            location = local_placement.Location.Coordinates
            x, y = float(location[0]), float(location[1])

            # Get wall length from representation or assume default
            length = 3.0  # Default wall length

            # Try to get actual dimensions from representation
            if wall.Representation:
                for rep in wall.Representation.Representations:
                    if rep.RepresentationType == 'SweptSolid':
                        for item in rep.Items:
                            if hasattr(item, 'SweptArea'):
                                # Get profile dimensions
                                profile = item.SweptArea
                                if hasattr(profile, 'XDim'):
                                    length = float(profile.XDim)

            # Get wall direction (default to X axis)
            direction = (1.0, 0.0)
            if local_placement.RefDirection:
                dir_ratios = local_placement.RefDirection.DirectionRatios
                direction = (float(dir_ratios[0]), float(dir_ratios[1]))

            # Calculate end point
            end_x = x + length * direction[0]
            end_y = y + length * direction[1]

            # Get material type
            material = 'concrete'
            if wall.HasAssociations:
                for assoc in wall.HasAssociations:
                    if assoc.is_a('IfcRelAssociatesMaterial'):
                        mat = assoc.RelatingMaterial
                        if hasattr(mat, 'Name') and mat.Name:
                            mat_name = mat.Name.lower()
                            if 'brick' in mat_name:
                                material = 'brick'
                            elif 'glass' in mat_name:
                                material = 'glass'
                            elif 'wood' in mat_name:
                                material = 'wood'
                            elif 'metal' in mat_name or 'steel' in mat_name:
                                material = 'metal'

            return WallSegment(
                start=Point(x=x, y=y),
                end=Point(x=end_x, y=end_y),
                thickness=0.2,
                material=material,
                attenuation_db=MATERIAL_ATTENUATION.get(material, 10.0)
            )

        except Exception as e:
            logger.debug(f"Failed to convert wall: {e}")
            return None

    def _extract_spaces(self, model) -> List[Room]:
        """Extract space/room elements from IFC model."""
        rooms = []

        try:
            for space in model.by_type('IfcSpace'):
                room = self._space_to_room(space)
                if room:
                    rooms.append(room)
        except Exception as e:
            logger.warning(f"Failed to extract spaces from IFC: {e}")

        return rooms

    def _space_to_room(self, space) -> Optional[Room]:
        """Convert IFC space to Room."""
        try:
            name = space.LongName or space.Name or "Room"

            # Get area from properties
            area = 20.0  # Default
            if space.IsDefinedBy:
                for rel in space.IsDefinedBy:
                    if rel.is_a('IfcRelDefinesByProperties'):
                        prop_set = rel.RelatingPropertyDefinition
                        if hasattr(prop_set, 'HasProperties'):
                            for prop in prop_set.HasProperties:
                                if prop.Name in ['Area', 'GrossFloorArea', 'NetFloorArea']:
                                    area = float(prop.NominalValue.wrappedValue)

            # Get placement for polygon
            polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]  # Default square

            if space.ObjectPlacement:
                placement = space.ObjectPlacement.RelativePlacement
                if placement and placement.Location:
                    loc = placement.Location.Coordinates
                    x, y = float(loc[0]), float(loc[1])
                    # Create approximate polygon based on area
                    side = math.sqrt(area)
                    polygon = [
                        [x, y],
                        [x + side, y],
                        [x + side, y + side],
                        [x, y + side]
                    ]

            return Room(
                name=name,
                area=area,
                polygon=polygon
            )

        except Exception as e:
            logger.debug(f"Failed to convert space: {e}")
            return None


class CADParser:
    """Unified CAD file parser supporting DXF, DWG, and IFC formats."""

    def __init__(self):
        self.dxf_parser = DXFParser()
        self.dwg_converter = DWGConverter()
        self.ifc_parser = IFCParser()

    def parse(self, file_path: str) -> MapData:
        """
        Parse CAD file and return floor plan data.

        Automatically detects file type and uses appropriate parser.

        Args:
            file_path: Path to CAD file (DXF, DWG, or IFC)

        Returns:
            MapData with extracted floor plan information
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.dxf':
            return self.dxf_parser.parse(file_path)

        elif ext == '.dwg':
            # Convert to DXF first
            dxf_path = self.dwg_converter.convert_to_dxf(file_path)
            return self.dxf_parser.parse(dxf_path)

        elif ext == '.ifc':
            return self.ifc_parser.parse(file_path)

        else:
            raise ValueError(f"Unsupported CAD file format: {ext}")

    def parse_dxf(self, file_path: str) -> CADParseResult:
        """
        Parse DXF file and return CADParseResult.

        Args:
            file_path: Path to DXF file

        Returns:
            CADParseResult with walls, rooms, scale
        """
        map_data = self.dxf_parser.parse(file_path)
        return CADParseResult(
            walls=map_data.walls,
            rooms=map_data.rooms,
            scale=self.dxf_parser.scale_factor
        )

    def parse_ifc(self, file_path: str) -> CADParseResult:
        """
        Parse IFC file and return CADParseResult.

        Args:
            file_path: Path to IFC file

        Returns:
            CADParseResult with walls, rooms, scale
        """
        map_data = self.ifc_parser.parse(file_path)
        return CADParseResult(
            walls=map_data.walls,
            rooms=map_data.rooms,
            scale=1.0  # IFC is typically in meters
        )

    def convert_dwg_to_dxf(self, file_path: str) -> Optional[str]:
        """
        Convert DWG to DXF.

        Args:
            file_path: Path to DWG file

        Returns:
            Path to converted DXF file, or None if conversion failed
        """
        try:
            return self.dwg_converter.convert_to_dxf(file_path)
        except Exception as e:
            logger.warning(f"DWG conversion failed: {e}")
            return None

    def get_supported_formats(self) -> Dict[str, bool]:
        """Get dictionary of supported formats and their availability."""
        return {
            'dxf': True,  # Always available (ezdxf is pure Python)
            'dwg': self.dwg_converter.is_available(),
            'ifc': self.ifc_parser.is_available()
        }
