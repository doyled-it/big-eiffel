"""American Family Field wall geometry and the definition of "clears".

Two notions of clearing are kept separate throughout the project:

1. Raw-distance clear: the ball's landing (carry) distance reaches the fence
   distance. This is the headline "1-in-8" style number.
2. Full clear: the ball also stays fair and is still above the wall height
   when it reaches the wall. Because an 8 ft wall sits 8 ft off the ground, a
   ball that merely lands at the wall base is at zero height there and does
   not clear. We convert the wall height into an extra distance using the
   trajectory's descent angle, which the physics model provides.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config as C

_anchor_x = np.array(sorted(C.PARK_WALL_ANCHORS))
_anchor_y = np.array([C.PARK_WALL_ANCHORS[k] for k in _anchor_x])
_wall_interp = PchipInterpolator(_anchor_x, _anchor_y, extrapolate=False)


def wall_distance(spray_deg: float | np.ndarray) -> np.ndarray:
    """Wall distance (ft) at a given spray angle, interpolated from anchors.

    Arguments:
        spray_deg: Spray angle in degrees (negative toward left field).

    Returns:
        Wall distance in feet. Outside the fair range the nearest foul-line
        distance is used.
    """
    spray = np.asarray(spray_deg, dtype=float)
    clamped = np.clip(spray, _anchor_x[0], _anchor_x[-1])
    return _wall_interp(clamped)


def is_fair(spray_deg: float | np.ndarray) -> np.ndarray:
    """Whether a spray angle is in fair territory.

    Arguments:
        spray_deg: Spray angle in degrees.

    Returns:
        Boolean array, True where the ball stays fair.
    """
    spray = np.asarray(spray_deg, dtype=float)
    return (spray >= C.FAIR_MIN_DEG) & (spray <= C.FAIR_MAX_DEG)


def wall_height_distance_penalty(descent_angle_deg: float, wall_height_ft: float = C.WALL_HEIGHT_FT) -> float:
    """Extra carry distance needed to still be above the wall at the fence.

    A ball descending at ``descent_angle_deg`` must land this many feet beyond
    the wall base to pass over a wall of height ``wall_height_ft``.

    Arguments:
        descent_angle_deg: Trajectory descent angle at the fence, in degrees
            below horizontal (positive).
        wall_height_ft: Wall height in feet.

    Returns:
        Extra distance in feet.
    """
    return wall_height_ft / np.tan(np.radians(descent_angle_deg))


def effective_fence(
    base_distance_ft: float, descent_angle_deg: float | None, wall_height_ft: float = C.WALL_HEIGHT_FT
) -> float:
    """Effective landing distance a ball must reach to clear a wall.

    Arguments:
        base_distance_ft: Wall distance along the ground.
        descent_angle_deg: Descent angle at the fence. If None, the wall
            height is ignored (raw-distance clear).
        wall_height_ft: Wall height in feet.

    Returns:
        Effective landing distance in feet.
    """
    if descent_angle_deg is None:
        return base_distance_ft
    return base_distance_ft + wall_height_distance_penalty(descent_angle_deg, wall_height_ft)


def clears_park(
    carry_ft: float | np.ndarray,
    spray_deg: float | np.ndarray,
    descent_angle_deg: float | None = None,
) -> np.ndarray:
    """Whether a batted ball clears the American Family Field wall.

    Arguments:
        carry_ft: Landing (carry) distance in feet.
        spray_deg: Spray angle in degrees.
        descent_angle_deg: Descent angle at the fence; if given, the 8 ft wall
            height is enforced, otherwise only the ground distance is checked.

    Returns:
        Boolean array, True where the ball clears and stays fair.
    """
    carry = np.asarray(carry_ft, dtype=float)
    base = wall_distance(spray_deg)
    needed = base if descent_angle_deg is None else base + wall_height_distance_penalty(descent_angle_deg)
    return is_fair(spray_deg) & (carry >= needed)


def clears_generic(
    carry_ft: float | np.ndarray,
    spray_deg: float | np.ndarray | None = None,
    fence_ft: float = C.FENCE_FT,
    descent_angle_deg: float | None = None,
) -> np.ndarray:
    """Whether a batted ball clears a generic fence of a fixed distance.

    Arguments:
        carry_ft: Landing (carry) distance in feet.
        spray_deg: Spray angle in degrees; if given, fairness is enforced.
        fence_ft: Fence distance in feet.
        descent_angle_deg: Descent angle; if given, an 8 ft wall is enforced.

    Returns:
        Boolean array, True where the ball clears.
    """
    carry = np.asarray(carry_ft, dtype=float)
    needed = fence_ft if descent_angle_deg is None else fence_ft + wall_height_distance_penalty(descent_angle_deg)
    cleared = carry >= needed
    if spray_deg is not None:
        cleared = cleared & is_fair(spray_deg)
    return cleared
