"""Physics-based RF propagation simulation engine.

Implements proper RF physics including:
- ITU-R P.1238-10 indoor propagation model
- ITU-R P.2040-2 building material penetration losses
- Free Space Path Loss (Friis equation)
- Fresnel zone obstruction
- Multi-wall penetration with angle of incidence
- Log-normal shadow fading
"""

import numpy as np
from typing import List, Tuple, Optional, Protocol, Dict
from dataclasses import dataclass
import logging
import math

from app.schemas.project import MapData, WallSegment
from app.schemas.router import Router as RouterSchema

logger = logging.getLogger(__name__)

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


# ITU-R P.2040-2: Building material penetration loss (dB)
# Format: {material: {frequency_ghz: (loss_db, loss_per_cm_db)}}
# loss_per_cm_db allows thickness-dependent calculation
MATERIAL_PROPERTIES: Dict[str, Dict[float, Tuple[float, float]]] = {
    # (base_loss_db, additional_loss_per_cm)
    "concrete": {
        2.4: (12.0, 0.8),   # Heavy concrete
        5.0: (18.0, 1.2),
        6.0: (23.0, 1.5),
    },
    "reinforced_concrete": {
        2.4: (18.0, 1.2),
        5.0: (27.0, 1.8),
        6.0: (32.0, 2.2),
    },
    "brick": {
        2.4: (8.0, 0.5),
        5.0: (12.0, 0.8),
        6.0: (15.0, 1.0),
    },
    "drywall": {  # Gypsum board / plasterboard
        2.4: (2.5, 0.2),
        5.0: (3.5, 0.3),
        6.0: (4.5, 0.4),
    },
    "plasterboard": {
        2.4: (2.5, 0.2),
        5.0: (3.5, 0.3),
        6.0: (4.5, 0.4),
    },
    "wood": {
        2.4: (3.5, 0.3),
        5.0: (5.0, 0.4),
        6.0: (6.5, 0.5),
    },
    "glass": {  # Standard clear glass
        2.4: (2.5, 0.15),
        5.0: (4.0, 0.25),
        6.0: (6.0, 0.35),
    },
    "glass_tinted": {  # IRR (Infrared Reflective) coated glass
        2.4: (8.0, 0.5),
        5.0: (15.0, 1.0),
        6.0: (20.0, 1.3),
    },
    "metal": {
        2.4: (40.0, 3.0),
        5.0: (50.0, 4.0),
        6.0: (55.0, 4.5),
    },
    "default": {
        2.4: (6.0, 0.4),
        5.0: (10.0, 0.7),
        6.0: (13.0, 0.9),
    },
}

# ITU-R P.1238-10: Distance power loss coefficient N
# For different indoor environments and frequencies
ITU_POWER_LOSS_COEFFICIENT: Dict[str, Dict[float, float]] = {
    "residential": {2.4: 28.0, 5.0: 30.0, 6.0: 31.0},
    "office": {2.4: 30.0, 5.0: 31.0, 6.0: 32.0},
    "commercial": {2.4: 22.0, 5.0: 24.0, 6.0: 25.0},
    "corridor": {2.4: 18.0, 5.0: 20.0, 6.0: 21.0},  # Waveguide effect
    "open_plan": {2.4: 20.0, 5.0: 22.0, 6.0: 23.0},
    "default": {2.4: 28.0, 5.0: 30.0, 6.0: 31.0},
}

# Floor penetration loss factor Lf(n) in dB
# ITU-R P.1238-10 Table 3
FLOOR_PENETRATION_LOSS = {
    "residential": {2.4: 4.0, 5.0: 5.0, 6.0: 6.0},  # per floor
    "office": {2.4: 15.0, 5.0: 16.0, 6.0: 18.0},  # per floor
    "commercial": {2.4: 6.0, 5.0: 7.0, 6.0: 8.0},  # per floor
}

# Material reflection coefficients (ratio of reflected power)
# Based on typical indoor material properties
REFLECTION_COEFFICIENTS: Dict[str, float] = {
    "concrete": 0.6,           # High reflection
    "reinforced_concrete": 0.7,
    "brick": 0.5,
    "drywall": 0.3,            # Lower reflection, more absorption
    "plasterboard": 0.3,
    "wood": 0.25,
    "glass": 0.4,              # Some reflection
    "glass_tinted": 0.5,       # Coated glass reflects more
    "metal": 0.9,              # Very high reflection
    "default": 0.4,
}


def get_frequency_key(frequency_ghz: float) -> float:
    """Map frequency to nearest standard frequency key."""
    if frequency_ghz >= 5.5:
        return 6.0
    elif frequency_ghz >= 4.0:
        return 5.0
    return 2.4


def calculate_fspl(distance_m: float, frequency_hz: float) -> float:
    """
    Calculate Free Space Path Loss using Friis equation.

    FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
             = 20*log10(d) + 20*log10(f) - 147.55

    Args:
        distance_m: Distance in meters
        frequency_hz: Frequency in Hz

    Returns:
        Path loss in dB
    """
    if distance_m < 0.1:
        distance_m = 0.1  # Minimum 10cm to avoid log(0)

    # Friis equation
    fspl_db = (
        20 * math.log10(distance_m) +
        20 * math.log10(frequency_hz) +
        20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)
    )
    return fspl_db


def calculate_itu_p1238_loss(
    distance_m: float,
    frequency_ghz: float,
    environment: str = "residential",
    num_floors: int = 0
) -> float:
    """
    Calculate path loss using ITU-R P.1238-10 indoor model.

    L = 20*log10(f) + N*log10(d) + Lf(n) - 28 dB

    Where:
        f = frequency in MHz
        N = distance power loss coefficient
        d = distance in meters (d > 1m)
        Lf(n) = floor penetration loss factor

    Args:
        distance_m: Distance in meters
        frequency_ghz: Frequency in GHz
        environment: Environment type
        num_floors: Number of floors between TX and RX

    Returns:
        Total path loss in dB
    """
    if distance_m < 1.0:
        # For very short distances, use FSPL
        return calculate_fspl(distance_m, frequency_ghz * 1e9)

    freq_key = get_frequency_key(frequency_ghz)
    frequency_mhz = frequency_ghz * 1000

    # Get power loss coefficient N
    env_lower = environment.lower()
    if env_lower in ITU_POWER_LOSS_COEFFICIENT:
        N = ITU_POWER_LOSS_COEFFICIENT[env_lower].get(freq_key, 28.0)
    else:
        N = ITU_POWER_LOSS_COEFFICIENT["default"].get(freq_key, 28.0)

    # Basic path loss
    L = 20 * math.log10(frequency_mhz) + N * math.log10(distance_m) - 28.0

    # Add floor penetration loss if applicable
    if num_floors > 0 and env_lower in FLOOR_PENETRATION_LOSS:
        Lf = FLOOR_PENETRATION_LOSS[env_lower].get(freq_key, 5.0)
        L += Lf * num_floors

    return L


def calculate_wall_loss(
    wall: WallSegment,
    frequency_ghz: float,
    angle_of_incidence: float = 0.0
) -> float:
    """
    Calculate wall penetration loss based on ITU-R P.2040-2.

    Loss increases with:
    - Higher frequency
    - Denser material
    - Greater thickness
    - Higher angle of incidence (grazing angles)

    Args:
        wall: Wall segment with material and thickness
        frequency_ghz: Frequency in GHz
        angle_of_incidence: Angle in radians (0 = perpendicular)

    Returns:
        Wall penetration loss in dB
    """
    freq_key = get_frequency_key(frequency_ghz)
    material = (wall.material or "default").lower()

    # Get material properties
    if material in MATERIAL_PROPERTIES:
        props = MATERIAL_PROPERTIES[material]
    else:
        props = MATERIAL_PROPERTIES["default"]

    base_loss, loss_per_cm = props.get(freq_key, (6.0, 0.4))

    # Calculate thickness-dependent loss
    thickness_cm = (wall.thickness or 0.2) * 100  # Convert m to cm
    thickness_loss = base_loss + loss_per_cm * max(0, thickness_cm - 10)

    # Angle of incidence correction
    # Loss increases at grazing angles (Snell's law effect)
    if angle_of_incidence > 0:
        # Approximate correction: loss increases as 1/cos(θ)
        # Capped to prevent infinity at 90 degrees
        angle_factor = min(1.0 / max(math.cos(angle_of_incidence), 0.1), 3.0)
        thickness_loss *= angle_factor

    return thickness_loss


def calculate_fresnel_zone_radius(
    distance_m: float,
    frequency_ghz: float,
    zone: int = 1
) -> float:
    """
    Calculate Fresnel zone radius.

    For the first Fresnel zone:
    r = sqrt(n * λ * d1 * d2 / (d1 + d2))

    At the midpoint (d1 = d2 = d/2):
    r = sqrt(n * λ * d / 4) = 0.5 * sqrt(n * λ * d)

    Args:
        distance_m: Total path distance in meters
        frequency_ghz: Frequency in GHz
        zone: Fresnel zone number (1 = first zone)

    Returns:
        Fresnel zone radius at midpoint in meters
    """
    wavelength = SPEED_OF_LIGHT / (frequency_ghz * 1e9)
    radius = 0.5 * math.sqrt(zone * wavelength * distance_m)
    return radius


def calculate_fresnel_obstruction_loss(
    clearance_ratio: float
) -> float:
    """
    Calculate additional loss due to Fresnel zone obstruction.

    Args:
        clearance_ratio: Ratio of clearance to first Fresnel zone radius
                        (1.0 = edge of first zone, 0.6 = 60% clearance)

    Returns:
        Additional loss in dB
    """
    if clearance_ratio >= 1.0:
        return 0.0  # Full clearance, no additional loss
    elif clearance_ratio >= 0.6:
        # Partial obstruction (0.6 to 1.0): gradual loss
        return 6.0 * (1.0 - clearance_ratio) / 0.4
    elif clearance_ratio >= 0.0:
        # Significant obstruction (0 to 0.6): knife-edge diffraction
        return 6.0 + 10.0 * (0.6 - clearance_ratio) / 0.6
    else:
        # Complete obstruction
        return 20.0


@dataclass
class SignalGrid:
    """Grid of signal strength values across the floor plan."""
    width: int  # Grid width in cells
    height: int  # Grid height in cells
    grid: np.ndarray  # 2D array of signal strengths in dBm
    resolution: float  # Meters per grid cell


class RFEngine(Protocol):
    """Protocol for RF propagation engines."""

    def predict_signal(
        self,
        map_data: MapData,
        router_positions: List[Tuple[float, float]],
        routers: List[RouterSchema],
        scale: float,
        grid_resolution: int = 2
    ) -> SignalGrid:
        """Predict signal strength across the floor plan."""
        ...


class PhysicsEngine:
    """
    Physics-based RF propagation engine.

    Implements:
    - ITU-R P.1238-10 indoor propagation model
    - ITU-R P.2040-2 material penetration losses
    - Fresnel zone obstruction effects
    - Multi-wall cumulative loss
    - Angle of incidence for wall penetration
    """

    def __init__(
        self,
        frequency_ghz: float = 2.4,
        environment: str = "residential",
        include_fresnel: bool = True,
        include_reflections: bool = True,
        include_shadow_fading: bool = False,
        shadow_fading_std: float = 4.0
    ):
        self.frequency_ghz = frequency_ghz
        self.frequency_hz = frequency_ghz * 1e9
        self.wavelength = SPEED_OF_LIGHT / self.frequency_hz
        self.environment = environment
        self.include_fresnel = include_fresnel
        self.include_reflections = include_reflections
        self.include_shadow_fading = include_shadow_fading
        self.shadow_fading_std = shadow_fading_std

        logger.info(
            f"PhysicsEngine initialized: {frequency_ghz}GHz, "
            f"env={environment}, fresnel={include_fresnel}, reflections={include_reflections}"
        )

    def predict_signal(
        self,
        map_data: MapData,
        router_positions: List[Tuple[float, float]],
        routers: List[RouterSchema],
        scale: float,
        grid_resolution: int = 2
    ) -> SignalGrid:
        """
        Predict signal strength across entire floor plan using physics model.

        Args:
            map_data: Floor plan with walls
            router_positions: List of (x, y) positions in pixels
            routers: Router specifications
            scale: Meters per pixel
            grid_resolution: Size of grid cells in pixels

        Returns:
            SignalGrid with predicted signal strengths
        """
        width = map_data.dimensions.width
        height = map_data.dimensions.height

        # Create grid
        grid_width = width // grid_resolution
        grid_height = height // grid_resolution
        signal_grid = np.full((grid_height, grid_width), -100.0)

        # Pre-compute wall segments for faster lookup
        walls = map_data.walls

        # For each router
        for router_pos, router in zip(router_positions, routers):
            router_x, router_y = router_pos

            # Get router parameters
            tx_power_dbm = router.max_tx_power_dbm
            tx_antenna_gain = float(router.antenna_gain_dbi)

            # Typical receiver antenna gain (smartphone/laptop)
            rx_antenna_gain = 2.0

            # Calculate signal at each grid point
            for gy in range(grid_height):
                for gx in range(grid_width):
                    # Convert grid coords to pixel coords (center of cell)
                    px = (gx + 0.5) * grid_resolution
                    py = (gy + 0.5) * grid_resolution

                    # Calculate distance in meters
                    dx = (px - router_x) * scale
                    dy = (py - router_y) * scale
                    distance_m = math.sqrt(dx * dx + dy * dy)

                    # ITU-R P.1238 path loss
                    path_loss_db = calculate_itu_p1238_loss(
                        distance_m,
                        self.frequency_ghz,
                        self.environment
                    )

                    # Wall penetration losses
                    wall_loss_db = self._calculate_total_wall_loss(
                        router_x, router_y, px, py, walls, scale
                    )

                    # Fresnel zone obstruction (optional)
                    fresnel_loss_db = 0.0
                    if self.include_fresnel and distance_m > 1.0:
                        fresnel_loss_db = self._calculate_fresnel_loss(
                            router_x, router_y, px, py, walls, scale, distance_m
                        )

                    # Shadow fading (log-normal, optional)
                    shadow_fading_db = 0.0
                    if self.include_shadow_fading:
                        shadow_fading_db = np.random.normal(0, self.shadow_fading_std)

                    # Total received power (link budget) for direct path
                    # Pr = Pt + Gt + Gr - Lp - Lw - Lf - Ls
                    direct_signal_dbm = (
                        tx_power_dbm +
                        tx_antenna_gain +
                        rx_antenna_gain -
                        path_loss_db -
                        wall_loss_db -
                        fresnel_loss_db -
                        shadow_fading_db
                    )

                    # Add first-order reflections (single bounce off walls)
                    signal_dbm = direct_signal_dbm
                    if self.include_reflections and distance_m > 0.5:
                        reflection_contribution = self._calculate_reflection_contribution(
                            router_x, router_y, px, py, walls, scale,
                            tx_power_dbm, tx_antenna_gain, rx_antenna_gain
                        )
                        if reflection_contribution > -100:
                            # Combine direct and reflected signals (power addition)
                            signal_dbm = self._power_sum_db([direct_signal_dbm, reflection_contribution])

                    # Take maximum signal (best router at this location)
                    signal_grid[gy, gx] = max(signal_grid[gy, gx], signal_dbm)

        return SignalGrid(
            width=grid_width,
            height=grid_height,
            grid=signal_grid,
            resolution=scale * grid_resolution
        )

    def _calculate_total_wall_loss(
        self,
        tx_x: float, tx_y: float,
        rx_x: float, rx_y: float,
        walls: List[WallSegment],
        scale: float
    ) -> float:
        """
        Calculate cumulative wall penetration loss along signal path.

        Includes:
        - Multi-wall model (each wall adds loss)
        - Angle of incidence correction
        - Material and thickness-dependent loss
        """
        total_loss_db = 0.0

        # Signal path vector
        path_dx = rx_x - tx_x
        path_dy = rx_y - tx_y
        path_length = math.sqrt(path_dx * path_dx + path_dy * path_dy)

        if path_length < 1.0:
            return 0.0

        for wall in walls:
            # Check intersection
            intersection = self._line_segment_intersection(
                tx_x, tx_y, rx_x, rx_y,
                wall.start.x, wall.start.y,
                wall.end.x, wall.end.y
            )

            if intersection is not None:
                # Calculate angle of incidence
                wall_dx = wall.end.x - wall.start.x
                wall_dy = wall.end.y - wall.start.y
                wall_length = math.sqrt(wall_dx * wall_dx + wall_dy * wall_dy)

                if wall_length > 0:
                    # Wall normal vector (perpendicular to wall)
                    normal_x = -wall_dy / wall_length
                    normal_y = wall_dx / wall_length

                    # Path unit vector
                    path_ux = path_dx / path_length
                    path_uy = path_dy / path_length

                    # Angle of incidence (dot product gives cos(θ))
                    cos_angle = abs(path_ux * normal_x + path_uy * normal_y)
                    angle_of_incidence = math.acos(min(cos_angle, 1.0))
                else:
                    angle_of_incidence = 0.0

                # Calculate wall loss with angle correction
                wall_loss = calculate_wall_loss(
                    wall, self.frequency_ghz, angle_of_incidence
                )
                total_loss_db += wall_loss

        return total_loss_db

    def _calculate_fresnel_loss(
        self,
        tx_x: float, tx_y: float,
        rx_x: float, rx_y: float,
        walls: List[WallSegment],
        scale: float,
        distance_m: float
    ) -> float:
        """
        Calculate Fresnel zone obstruction loss.

        Checks if walls penetrate the first Fresnel zone ellipsoid
        and calculates additional diffraction loss.
        """
        # First Fresnel zone radius at midpoint
        fresnel_radius_m = calculate_fresnel_zone_radius(
            distance_m, self.frequency_ghz, zone=1
        )
        fresnel_radius_px = fresnel_radius_m / scale

        # Path midpoint
        mid_x = (tx_x + rx_x) / 2
        mid_y = (tx_y + rx_y) / 2

        # Check each wall for Fresnel zone penetration
        min_clearance_ratio = 1.0

        for wall in walls:
            # Distance from wall to path midpoint
            clearance_px = self._point_to_segment_distance(
                mid_x, mid_y,
                wall.start.x, wall.start.y,
                wall.end.x, wall.end.y
            )

            # Wall thickness adds to obstruction
            wall_thickness_px = (wall.thickness or 0.2) / scale
            effective_clearance = clearance_px - wall_thickness_px / 2

            # Clearance ratio (1.0 = exactly at Fresnel zone edge)
            if fresnel_radius_px > 0:
                clearance_ratio = effective_clearance / fresnel_radius_px
                min_clearance_ratio = min(min_clearance_ratio, clearance_ratio)

        return calculate_fresnel_obstruction_loss(min_clearance_ratio)

    def _line_segment_intersection(
        self,
        p1x: float, p1y: float,
        p2x: float, p2y: float,
        p3x: float, p3y: float,
        p4x: float, p4y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Find intersection point of two line segments.

        Returns intersection point (x, y) or None if no intersection.
        Uses parametric form of line equations.
        """
        # Direction vectors
        d1x = p2x - p1x
        d1y = p2y - p1y
        d2x = p4x - p3x
        d2y = p4y - p3y

        # Cross product of directions
        cross = d1x * d2y - d1y * d2x

        if abs(cross) < 1e-10:
            return None  # Parallel or coincident

        # Parameters for intersection
        t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / cross
        u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / cross

        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = p1x + t * d1x
            iy = p1y + t * d1y
            return (ix, iy)

        return None

    def _point_to_segment_distance(
        self,
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        # Segment vector
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-10:
            # Degenerate segment (point)
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # Project point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    def _calculate_reflection_contribution(
        self,
        tx_x: float, tx_y: float,
        rx_x: float, rx_y: float,
        walls: List[WallSegment],
        scale: float,
        tx_power_dbm: float,
        tx_antenna_gain: float,
        rx_antenna_gain: float
    ) -> float:
        """
        Calculate signal contribution from first-order wall reflections.

        Uses the image source method: for each wall, creates a virtual
        transmitter at the mirror image position and checks if the
        reflected path is valid.

        Returns:
            Combined reflected signal power in dBm, or -100 if no valid reflections
        """
        reflection_powers = []

        for wall in walls:
            # Get mirror image of TX across this wall
            mirror_tx = self._mirror_point_across_wall(tx_x, tx_y, wall)
            if mirror_tx is None:
                continue

            mirror_x, mirror_y = mirror_tx

            # Find reflection point on wall
            reflection_point = self._get_reflection_point(
                mirror_x, mirror_y, rx_x, rx_y, wall
            )
            if reflection_point is None:
                continue

            ref_x, ref_y = reflection_point

            # Calculate distances for reflected path
            d1_px = math.sqrt((tx_x - ref_x)**2 + (tx_y - ref_y)**2)
            d2_px = math.sqrt((ref_x - rx_x)**2 + (ref_y - rx_y)**2)
            total_distance_m = (d1_px + d2_px) * scale

            if total_distance_m < 0.5:
                continue

            # Path loss for reflected path
            path_loss_db = calculate_itu_p1238_loss(
                total_distance_m,
                self.frequency_ghz,
                self.environment
            )

            # Reflection loss based on material
            material = (wall.material or "default").lower()
            reflection_coeff = REFLECTION_COEFFICIENTS.get(material, 0.4)
            reflection_loss_db = -20 * math.log10(max(reflection_coeff, 0.01))

            # Check for wall obstructions on the reflected path
            # TX to reflection point
            wall_loss_1 = self._calculate_wall_loss_simple(
                tx_x, tx_y, ref_x, ref_y, walls, wall
            )
            # Reflection point to RX
            wall_loss_2 = self._calculate_wall_loss_simple(
                ref_x, ref_y, rx_x, rx_y, walls, wall
            )

            total_wall_loss = wall_loss_1 + wall_loss_2

            # Calculate reflected signal power
            reflected_power_dbm = (
                tx_power_dbm +
                tx_antenna_gain +
                rx_antenna_gain -
                path_loss_db -
                reflection_loss_db -
                total_wall_loss
            )

            # Only include significant reflections
            if reflected_power_dbm > -95:
                reflection_powers.append(reflected_power_dbm)

        # Combine all reflections (take best 3 to limit computation)
        if not reflection_powers:
            return -100.0

        # Sort and take top reflections
        reflection_powers.sort(reverse=True)
        top_reflections = reflection_powers[:3]

        return self._power_sum_db(top_reflections)

    def _mirror_point_across_wall(
        self,
        px: float, py: float,
        wall: WallSegment
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate mirror image of point across wall line.

        Returns mirror point coordinates, or None if point is on wrong side.
        """
        # Wall vector
        wall_dx = wall.end.x - wall.start.x
        wall_dy = wall.end.y - wall.start.y
        wall_len_sq = wall_dx * wall_dx + wall_dy * wall_dy

        if wall_len_sq < 1e-10:
            return None

        # Project point onto wall line
        t = ((px - wall.start.x) * wall_dx + (py - wall.start.y) * wall_dy) / wall_len_sq

        # Closest point on infinite wall line
        closest_x = wall.start.x + t * wall_dx
        closest_y = wall.start.y + t * wall_dy

        # Mirror point
        mirror_x = 2 * closest_x - px
        mirror_y = 2 * closest_y - py

        return (mirror_x, mirror_y)

    def _get_reflection_point(
        self,
        mirror_tx_x: float, mirror_tx_y: float,
        rx_x: float, rx_y: float,
        wall: WallSegment
    ) -> Optional[Tuple[float, float]]:
        """
        Find the reflection point on a wall segment.

        The reflection point is where the line from mirror_tx to rx
        intersects the wall segment.
        """
        intersection = self._line_segment_intersection(
            mirror_tx_x, mirror_tx_y, rx_x, rx_y,
            wall.start.x, wall.start.y,
            wall.end.x, wall.end.y
        )

        return intersection

    def _calculate_wall_loss_simple(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        walls: List[WallSegment],
        exclude_wall: Optional[WallSegment] = None
    ) -> float:
        """
        Calculate wall loss for a path segment, optionally excluding one wall.
        """
        total_loss = 0.0

        for wall in walls:
            # Skip the reflecting wall itself
            if exclude_wall is not None:
                if (wall.start.x == exclude_wall.start.x and
                    wall.start.y == exclude_wall.start.y and
                    wall.end.x == exclude_wall.end.x and
                    wall.end.y == exclude_wall.end.y):
                    continue

            intersection = self._line_segment_intersection(
                x1, y1, x2, y2,
                wall.start.x, wall.start.y,
                wall.end.x, wall.end.y
            )

            if intersection is not None:
                wall_loss = calculate_wall_loss(wall, self.frequency_ghz, 0.0)
                total_loss += wall_loss

        return total_loss

    def _power_sum_db(self, powers_dbm: List[float]) -> float:
        """
        Sum multiple power levels in dB domain.

        Converts to linear, sums, converts back to dB.
        """
        if not powers_dbm:
            return -100.0

        # Filter out very weak signals
        valid_powers = [p for p in powers_dbm if p > -100]
        if not valid_powers:
            return -100.0

        # Convert to linear (milliwatts)
        linear_powers = [10 ** (p / 10) for p in valid_powers]

        # Sum and convert back to dBm
        total_linear = sum(linear_powers)
        return 10 * math.log10(total_linear) if total_linear > 0 else -100.0


# Backwards compatibility alias
SimplifiedEngine = PhysicsEngine


class PyLayersEngine:
    """
    Full ray-tracing RF propagation engine using PyLayers.

    Falls back to PhysicsEngine if PyLayers is not available.
    """

    def __init__(self, frequency_ghz: float = 2.4, environment: str = "residential"):
        self.frequency_ghz = frequency_ghz
        self.environment = environment
        self._pylayers_available = self._check_pylayers()

    def _check_pylayers(self) -> bool:
        """Check if PyLayers is available."""
        try:
            import pylayers
            return True
        except ImportError:
            return False

    def predict_signal(
        self,
        map_data: MapData,
        router_positions: List[Tuple[float, float]],
        routers: List[RouterSchema],
        scale: float,
        grid_resolution: int = 2
    ) -> SignalGrid:
        """
        Predict signal strength using PyLayers ray-tracing.
        Falls back to PhysicsEngine if PyLayers is not available.
        """
        # Use physics engine (PyLayers integration TODO)
        physics = PhysicsEngine(self.frequency_ghz, self.environment)
        return physics.predict_signal(
            map_data, router_positions, routers, scale, grid_resolution
        )


def get_rf_engine(
    frequency_ghz: float = 2.4,
    environment: str = "residential"
) -> PhysicsEngine:
    """
    Factory function to get RF propagation engine.

    Args:
        frequency_ghz: Operating frequency
        environment: Environment type (residential, office, commercial)

    Returns:
        PhysicsEngine instance
    """
    return PhysicsEngine(frequency_ghz, environment)


def calculate_coverage_percentage(
    signal_grid: SignalGrid,
    threshold_dbm: float = -70.0
) -> float:
    """
    Calculate percentage of area with acceptable signal.

    Args:
        signal_grid: Grid of signal strengths
        threshold_dbm: Minimum acceptable signal strength

    Returns:
        Percentage of covered area (0-100)
    """
    total_cells = signal_grid.grid.size
    covered_cells = np.sum(signal_grid.grid >= threshold_dbm)
    return (covered_cells / total_cells) * 100.0


def calculate_signal_statistics(signal_grid: SignalGrid) -> dict:
    """Calculate various signal statistics for the grid."""
    grid = signal_grid.grid

    # Filter out uninitialized values
    valid_signals = grid[grid > -99]

    if len(valid_signals) == 0:
        return {
            "mean": -100.0,
            "median": -100.0,
            "std": 0.0,
            "min": -100.0,
            "max": -100.0,
            "percentile_10": -100.0,
            "percentile_90": -100.0
        }

    return {
        "mean": float(np.mean(valid_signals)),
        "median": float(np.median(valid_signals)),
        "std": float(np.std(valid_signals)),
        "min": float(np.min(valid_signals)),
        "max": float(np.max(valid_signals)),
        "percentile_10": float(np.percentile(valid_signals, 10)),
        "percentile_90": float(np.percentile(valid_signals, 90))
    }


# Utility functions for external use
def get_material_attenuation(material: str, frequency_ghz: float) -> float:
    """Get base wall attenuation for a material at given frequency."""
    freq_key = get_frequency_key(frequency_ghz)
    material_lower = (material or "default").lower()

    if material_lower in MATERIAL_PROPERTIES:
        base_loss, _ = MATERIAL_PROPERTIES[material_lower].get(freq_key, (6.0, 0.4))
    else:
        base_loss, _ = MATERIAL_PROPERTIES["default"].get(freq_key, (6.0, 0.4))

    return base_loss


def get_path_loss_exponent(environment: str, frequency_ghz: float) -> float:
    """Get ITU-R P.1238 distance power loss coefficient."""
    freq_key = get_frequency_key(frequency_ghz)
    env_lower = (environment or "default").lower()

    if env_lower in ITU_POWER_LOSS_COEFFICIENT:
        return ITU_POWER_LOSS_COEFFICIENT[env_lower].get(freq_key, 28.0) / 10.0
    return ITU_POWER_LOSS_COEFFICIENT["default"].get(freq_key, 28.0) / 10.0
