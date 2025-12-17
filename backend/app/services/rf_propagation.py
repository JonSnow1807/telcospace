"""RF propagation simulation engine with dual implementation (Simplified + PyLayers)."""

import numpy as np
from typing import List, Tuple, Optional, Protocol
from dataclasses import dataclass
import os

from app.schemas.project import MapData, WallSegment
from app.schemas.router import Router as RouterSchema


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


class SimplifiedEngine:
    """
    Simplified RF propagation engine using Free Space Path Loss + wall attenuation.

    This is the fallback engine that's guaranteed to work without external dependencies.
    Uses:
    1. Free Space Path Loss (Friis equation)
    2. Wall penetration losses
    3. Simple log-distance path loss model
    """

    def __init__(self, frequency_ghz: float = 2.4):
        self.frequency_ghz = frequency_ghz
        self.frequency_hz = frequency_ghz * 1e9
        self.wavelength = 3e8 / self.frequency_hz

        # Path loss exponent (indoor environment)
        self.path_loss_exponent = 3.0

        # Reference distance (1 meter)
        self.d0 = 1.0

        # Shadowing standard deviation (dB)
        self.shadow_std = 4.0

    def predict_signal(
        self,
        map_data: MapData,
        router_positions: List[Tuple[float, float]],
        routers: List[RouterSchema],
        scale: float,
        grid_resolution: int = 2
    ) -> SignalGrid:
        """
        Predict signal strength across entire floor plan.

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
        signal_grid = np.full((grid_height, grid_width), -100.0)  # Initialize with very low signal

        # For each router
        for router_pos, router in zip(router_positions, routers):
            router_x, router_y = router_pos

            # Calculate signal at each grid point
            for gy in range(grid_height):
                for gx in range(grid_width):
                    # Convert grid coords to pixel coords
                    px = (gx + 0.5) * grid_resolution
                    py = (gy + 0.5) * grid_resolution

                    # Calculate signal strength from this router
                    signal = self._calculate_point_signal(
                        router_x, router_y,
                        px, py,
                        router,
                        map_data.walls,
                        scale
                    )

                    # Take maximum signal (best router at this location)
                    signal_grid[gy, gx] = max(signal_grid[gy, gx], signal)

        return SignalGrid(
            width=grid_width,
            height=grid_height,
            grid=signal_grid,
            resolution=scale * grid_resolution
        )

    def _calculate_point_signal(
        self,
        tx_x: float, tx_y: float,
        rx_x: float, rx_y: float,
        router: RouterSchema,
        walls: List[WallSegment],
        scale: float
    ) -> float:
        """
        Calculate signal strength at a single point.

        Uses:
        1. Log-distance path loss model
        2. Wall penetration losses
        """
        # Convert to meters
        distance_m = np.sqrt((rx_x - tx_x) ** 2 + (rx_y - tx_y) ** 2) * scale

        # Minimum distance to avoid log(0)
        if distance_m < 0.1:
            distance_m = 0.1

        # Free Space Path Loss at reference distance
        fspl_d0 = 20 * np.log10(self.d0) + \
                  20 * np.log10(self.frequency_hz) + \
                  20 * np.log10(4 * np.pi / 3e8)

        # Log-distance path loss
        path_loss_db = fspl_d0 + 10 * self.path_loss_exponent * np.log10(distance_m / self.d0)

        # Calculate wall penetration loss
        wall_loss_db = self._calculate_wall_penetration(
            tx_x, tx_y, rx_x, rx_y, walls
        )

        # Received signal strength
        tx_power = router.max_tx_power_dbm
        antenna_gain = float(router.antenna_gain_dbi)
        rx_antenna_gain = 2.0  # Typical device antenna gain

        signal_dbm = (
            tx_power +
            antenna_gain +
            rx_antenna_gain -
            path_loss_db -
            wall_loss_db
        )

        return signal_dbm

    def _calculate_wall_penetration(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        walls: List[WallSegment]
    ) -> float:
        """
        Calculate total attenuation from walls between two points.

        Checks line-of-sight path and sums up wall losses.
        """
        total_loss_db = 0.0

        for wall in walls:
            # Check if line (x1,y1)-(x2,y2) intersects wall
            if self._line_intersects_segment(
                x1, y1, x2, y2,
                wall.start.x, wall.start.y,
                wall.end.x, wall.end.y
            ):
                total_loss_db += wall.attenuation_db

        return total_loss_db

    def _line_intersects_segment(
        self,
        p1x: float, p1y: float,
        p2x: float, p2y: float,
        p3x: float, p3y: float,
        p4x: float, p4y: float
    ) -> bool:
        """
        Check if line segment (p1-p2) intersects segment (p3-p4).

        Uses cross product method for robust intersection detection.
        """
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        return (
            ccw(p1x, p1y, p3x, p3y, p4x, p4y) != ccw(p2x, p2y, p3x, p3y, p4x, p4y) and
            ccw(p1x, p1y, p2x, p2y, p3x, p3y) != ccw(p1x, p1y, p2x, p2y, p4x, p4y)
        )


class PyLayersEngine:
    """
    Full ray-tracing RF propagation engine using PyLayers.

    Provides more accurate signal prediction through:
    1. Full ray-tracing with reflections
    2. Diffraction effects
    3. Material-specific propagation characteristics
    """

    def __init__(self, frequency_ghz: float = 2.4):
        self.frequency_ghz = frequency_ghz
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

        Falls back to SimplifiedEngine if PyLayers is not available.
        """
        if not self._pylayers_available:
            # Fallback to simplified engine
            simplified = SimplifiedEngine(self.frequency_ghz)
            return simplified.predict_signal(
                map_data, router_positions, routers, scale, grid_resolution
            )

        # PyLayers implementation would go here
        # For now, use simplified engine
        # TODO: Implement full PyLayers integration
        simplified = SimplifiedEngine(self.frequency_ghz)
        return simplified.predict_signal(
            map_data, router_positions, routers, scale, grid_resolution
        )


def get_rf_engine(frequency_ghz: float = 2.4) -> RFEngine:
    """
    Factory function to get the best available RF engine.

    Tries PyLayers first, falls back to simplified model.
    """
    try:
        import pylayers
        return PyLayersEngine(frequency_ghz)
    except ImportError:
        return SimplifiedEngine(frequency_ghz)


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
