"""
Auto-detect geometry bounds from a FEMM .fem file (Approach 1)

This module parses the FEMM file directly to extract the point list
and computes useful geometry bounds:
- Axis-aligned bounding box (x_min/x_max, y_min/y_max)
- Maximum radial extent r_max
- Approximate angular coverage (theta_min/theta_max)
- Basic counts (num_points)

Usage (CLI):
    python auto_detect_geometry.py d:\\path\\to\\model.fem

Programmatic:
    from auto_detect_geometry import auto_detect_geometry_bounds
    bounds = auto_detect_geometry_bounds("TeslaModel3.fem")

Notes:
- This does NOT require FEMM/pyfemm; it operates on the file alone.
- It reads the [NumPoints] section and stops at the next bracketed section.
- It safely skips malformed lines.
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple

SectionHeaders = (
    "[NumPoints]",
    "[NumSegments]",
    "[NumArcSegments]",
    "[NumHoles]",
    "[NumBlockLabels]",
)


def _parse_points(fem_path: Path) -> List[Tuple[float, float]]:
    """Parse the [NumPoints] section and return list of (x, y) floats.

    The FEMM file uses tab-separated values for points.
    We read lines after [NumPoints] until another bracketed section starts.
    """
    points: List[Tuple[float, float]] = []

    with fem_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    in_points = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Section header detection
        if line.startswith("[") and line.endswith("]"):
            if line == "[NumPoints]":
                in_points = True
                # Some FEMM files include the count on the next line; we can ignore it
                continue
            else:
                # Any other header ends the points section
                if in_points:
                    break
                else:
                    continue

        if in_points:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((x, y))
                except ValueError:
                    # Skip non-numeric lines (e.g., metadata)
                    continue

    return points


def _compute_bounds(points: List[Tuple[float, float]]) -> Dict[str, float]:
    """Compute bounding box, radial and angular extents from points."""
    if not points:
        raise ValueError("No points parsed from FEM file")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # Radial
    rs = [math.hypot(x, y) for x, y in points]
    r_max = max(rs)

    # Angular coverage (mechanical angle in degrees)
    thetas = []
    for x, y in points:
        if abs(x) < 1e-12 and abs(y) < 1e-12:
            continue
        th = math.degrees(math.atan2(y, x))
        if th < 0:
            th += 360.0
        thetas.append(th)

    if thetas:
        theta_min = min(thetas)
        theta_max = max(thetas)
    else:
        theta_min = 0.0
        theta_max = 60.0  # reasonable default for sector models

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "r_max": r_max,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "num_points": float(len(points)),
    }


def auto_detect_geometry_bounds(fem_file_path: str | Path) -> Dict[str, float]:
    """High-level API: parse FEM file and return geometry bounds.

    Returns a dict with keys:
      x_min, x_max, y_min, y_max, r_max, theta_min, theta_max, num_points
    """
    fem_path = Path(fem_file_path)
    if not fem_path.exists():
        raise FileNotFoundError(f"FEM file not found: {fem_path}")

    points = _parse_points(fem_path)
    bounds = _compute_bounds(points)
    return bounds


def _format_bounds(bounds: Dict[str, float]) -> str:
    return (
        f"Bounds:\n"
        f"  X: [{bounds['x_min']:.3f}, {bounds['x_max']:.3f}] mm\n"
        f"  Y: [{bounds['y_min']:.3f}, {bounds['y_max']:.3f}] mm\n"
        f"  r_max: {bounds['r_max']:.3f} mm\n"
        f"  theta: {bounds['theta_min']:.3f}° → {bounds['theta_max']:.3f}°\n"
        f"  points: {int(bounds['num_points'])}\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_detect_geometry.py <path-to-.fem>")
        sys.exit(1)

    fem_path = Path(sys.argv[1])
    try:
        bounds = auto_detect_geometry_bounds(fem_path)
        print(_format_bounds(bounds))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)
