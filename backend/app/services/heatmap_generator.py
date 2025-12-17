"""Heatmap visualization generator for signal coverage."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple
import os

from app.services.rf_propagation import SignalGrid


# Custom colormap: Red (weak) -> Yellow -> Green (strong)
SIGNAL_COLORMAP = LinearSegmentedColormap.from_list(
    'signal_strength',
    [
        (0.8, 0.0, 0.0),    # Red (weak signal)
        (1.0, 0.5, 0.0),    # Orange
        (1.0, 1.0, 0.0),    # Yellow
        (0.5, 1.0, 0.0),    # Light green
        (0.0, 0.8, 0.0),    # Green (strong signal)
    ]
)


def generate_heatmap_image(
    signal_grid: SignalGrid,
    output_path: str,
    background_image: Optional[str] = None,
    router_positions: Optional[List[Tuple[float, float]]] = None,
    grid_resolution: int = 2,
    vmin: float = -90.0,
    vmax: float = -30.0,
    alpha: float = 0.7,
    dpi: int = 150
) -> str:
    """
    Generate a heatmap visualization of signal strength.

    Args:
        signal_grid: Signal strength grid
        output_path: Where to save the image
        background_image: Optional path to floor plan for overlay
        router_positions: Optional list of router positions to mark
        grid_resolution: Grid cell size in pixels
        vmin: Minimum signal for colormap (dBm)
        vmax: Maximum signal for colormap (dBm)
        alpha: Heatmap transparency (0-1)
        dpi: Output image DPI

    Returns:
        Path to generated image
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate figure size to match signal grid dimensions
    fig_width = signal_grid.width / 100.0 * 4  # Scale for visibility
    fig_height = signal_grid.height / 100.0 * 4

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, fig_width), max(6, fig_height)))

    # Load and display background image if provided
    if background_image and os.path.exists(background_image):
        bg_img = plt.imread(background_image)
        ax.imshow(bg_img, extent=[0, signal_grid.width, signal_grid.height, 0], alpha=0.4)

    # Plot heatmap
    im = ax.imshow(
        signal_grid.grid,
        cmap=SIGNAL_COLORMAP,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        extent=[0, signal_grid.width, signal_grid.height, 0]
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Signal Strength (dBm)', rotation=270, labelpad=15)

    # Mark router positions
    if router_positions:
        for i, (rx, ry) in enumerate(router_positions):
            # Convert pixel position to grid position
            gx = rx / grid_resolution
            gy = ry / grid_resolution
            ax.plot(gx, gy, 'b^', markersize=15, markeredgecolor='white', markeredgewidth=2)
            ax.annotate(
                f'R{i+1}',
                (gx, gy),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='blue'
            )

    # Configure axes
    ax.set_xlim(0, signal_grid.width)
    ax.set_ylim(signal_grid.height, 0)  # Flip Y axis
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('WiFi Signal Coverage Heatmap')

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def generate_coverage_report(
    signal_grid: SignalGrid,
    threshold_dbm: float = -70.0
) -> dict:
    """
    Generate a coverage report with statistics.

    Args:
        signal_grid: Signal strength grid
        threshold_dbm: Minimum acceptable signal strength

    Returns:
        Dictionary with coverage statistics
    """
    grid = signal_grid.grid
    total_cells = grid.size

    # Calculate coverage at different thresholds
    excellent_threshold = -50.0  # Very strong
    good_threshold = -60.0       # Strong
    fair_threshold = -70.0       # Acceptable
    weak_threshold = -80.0       # Weak

    excellent_cells = np.sum(grid >= excellent_threshold)
    good_cells = np.sum((grid >= good_threshold) & (grid < excellent_threshold))
    fair_cells = np.sum((grid >= fair_threshold) & (grid < good_threshold))
    weak_cells = np.sum((grid >= weak_threshold) & (grid < fair_threshold))
    dead_cells = np.sum(grid < weak_threshold)

    # Signal statistics
    valid_signals = grid[grid > -99]

    report = {
        "total_area": total_cells,
        "coverage_breakdown": {
            "excellent": {
                "cells": int(excellent_cells),
                "percentage": float(excellent_cells / total_cells * 100),
                "threshold": f">= {excellent_threshold} dBm"
            },
            "good": {
                "cells": int(good_cells),
                "percentage": float(good_cells / total_cells * 100),
                "threshold": f"{good_threshold} to {excellent_threshold} dBm"
            },
            "fair": {
                "cells": int(fair_cells),
                "percentage": float(fair_cells / total_cells * 100),
                "threshold": f"{fair_threshold} to {good_threshold} dBm"
            },
            "weak": {
                "cells": int(weak_cells),
                "percentage": float(weak_cells / total_cells * 100),
                "threshold": f"{weak_threshold} to {fair_threshold} dBm"
            },
            "dead_zone": {
                "cells": int(dead_cells),
                "percentage": float(dead_cells / total_cells * 100),
                "threshold": f"< {weak_threshold} dBm"
            }
        },
        "total_coverage_percent": float((total_cells - dead_cells) / total_cells * 100),
        "acceptable_coverage_percent": float(np.sum(grid >= threshold_dbm) / total_cells * 100),
        "signal_statistics": {
            "mean": float(np.mean(valid_signals)) if len(valid_signals) > 0 else -100.0,
            "median": float(np.median(valid_signals)) if len(valid_signals) > 0 else -100.0,
            "std": float(np.std(valid_signals)) if len(valid_signals) > 0 else 0.0,
            "min": float(np.min(valid_signals)) if len(valid_signals) > 0 else -100.0,
            "max": float(np.max(valid_signals)) if len(valid_signals) > 0 else -100.0
        }
    }

    return report
