"""Alignment DOF and target-layer conventions shared by every solver."""

from __future__ import annotations

from alignment_artifact_layout import ACTIVE_DOF as CONFIG_ACTIVE_DOF, TARGET_LAYERS

EPS_COVARIANCE = 1e-8
DEFAULT_TARGET_LAYERS = TARGET_LAYERS

# Shared by both track modes.  tz is excluded (never injected and poorly
# constrained by a planar residual); rz must stay in -- it is the only
# rotation the current misalignment actually produces.
DEFAULT_ACTIVE_DOF = CONFIG_ACTIVE_DOF


def parse_active_dof(active_dof):
    """Parse active alignment DOF from CLI text or an iterable of integers."""
    names = {
        "tx": 0, "ty": 1, "tz": 2,
        "rx": 3, "alpha": 3,
        "ry": 4, "beta": 4,
        "rz": 5, "gamma": 5,
    }
    if active_dof is None or active_dof == "":
        return DEFAULT_ACTIVE_DOF
    if isinstance(active_dof, str):
        raw_items = [item.strip().lower() for item in active_dof.split(",") if item.strip()]
    else:
        raw_items = list(active_dof)

    parsed = []
    for item in raw_items:
        if isinstance(item, str) and item in names:
            value = names[item]
        else:
            value = int(item)
        if value < 0 or value > 5:
            raise ValueError(f"Active DOF {value} is outside the valid range [0, 5].")
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("At least one active DOF must be selected.")
    return tuple(parsed)


def parse_target_layers(target_layers):
    """Parse target layer spec like '17:8,24:4' into ((17, 8), (24, 4))."""
    if target_layers is None or target_layers == "":
        return DEFAULT_TARGET_LAYERS
    if not isinstance(target_layers, str):
        return tuple((int(vol), int(lay)) for vol, lay in target_layers)

    parsed = []
    for item in target_layers.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Target layer '{item}' must have format volume:layer.")
        parsed.append((int(parts[0]), int(parts[1])))
    if not parsed:
        raise ValueError("At least one target layer must be selected.")
    return tuple(parsed)


def resolve_active_dof(config):
    """Both track modes share one default; parse_active_dof handles None and ''."""
    return parse_active_dof(config.active_dof)
