"""Hard protocol identity checks for the formal large-random corpus."""

import math


PROTOCOL_ID = "large_random_vhacd_high_v1"
COLLISION_MODE = "vhacd_high_v1"
VISUAL_COLLISION_MODE = "vhacd_visual_high_v2"
VHACD_RESOLUTION = 1_000_000
VHACD_MAX_CONVEX_HULLS = 128
VHACD_MAX_VERTICES = 64


def _close(value, expected, tolerance=1e-9):
    try:
        return math.isclose(float(value), expected, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _int_equal(value, expected):
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def is_formal_protocol_manifest(manifest):
    """Reject legacy single-hull attempts while preserving them on disk."""
    high_v1 = bool(
        manifest.get("object_collision_mode") == COLLISION_MODE
        and manifest.get("object_vhacd_high_v1") is True
        and manifest.get("object_vhacd_visual_high_v2") is not True
    )
    visual_high_v2 = bool(
        manifest.get("object_collision_mode") == VISUAL_COLLISION_MODE
        and manifest.get("object_vhacd_visual_high_v2") is True
        and manifest.get("object_vhacd_high_v1") is not True
    )
    collision_ok = bool(
        (high_v1 or visual_high_v2)
        and manifest.get("object_vhacd") is True
        and manifest.get("object_multicollision") is not True
        and _int_equal(
            manifest.get("object_vhacd_resolution"), VHACD_RESOLUTION
        )
        and _int_equal(
            manifest.get("object_vhacd_max_convex_hulls"),
            VHACD_MAX_CONVEX_HULLS,
        )
        and _int_equal(
            manifest.get("object_vhacd_max_vertices"),
            VHACD_MAX_VERTICES,
        )
    )
    lift = manifest.get(
        "lift_command_height_m", manifest.get("lift_height")
    )
    return bool(
        collision_ok
        and _close(lift, 0.10)
        and _close(manifest.get("contact_offset_m"), 0.001)
        and _close(manifest.get("gravity" , 9.8), 9.8)
        and _close(manifest.get("max_gravity_displacement_m"), 0.0125)
        and _close(manifest.get("max_direction_displacement_m"), 0.015)
    )


def assert_formal_protocol_manifest(manifest, source="manifest"):
    if not is_formal_protocol_manifest(manifest):
        raise RuntimeError(
            f"{source} does not satisfy {PROTOCOL_ID}: {manifest}"
        )
