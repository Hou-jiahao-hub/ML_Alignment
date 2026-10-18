"""ACTS-independent differentiable six-parameter helix mathematics.

The physical parameter order is ``(x0, y0, z0, psi0, tan_lambda, rho)``.
Path length is transverse arc length in millimetres and ``rho`` is signed
curvature in inverse millimetres.

Import facade over ``mlalign.helix``: ``trajectory`` (evaluation and plane
intersection), ``gauge`` (optimizer representations), ``gauss_newton`` (shared
row-wise LM), ``measurement_fit`` and ``point_fit``.  Monkeypatching a name
here does not reach the ``mlalign.helix`` code that calls it.
"""

from __future__ import annotations

import torch  # noqa: F401

try:  # Imported as part of the ``scripts`` package.
    from .mlalign.helix.gauge import (  # noqa: F401
        assemble_track_parameters,
        CURVATURE_SCALE,
        pack_gauge_fixed_track_parameters,
        reference_gauge_frame,
        split_track_parameters,
        unpack_gauge_fixed_track_parameters,
    )
    from .mlalign.helix.gauss_newton import (  # noqa: F401
        _merge_gauss_newton_diagnostics,
        _stationary_newton_decrement,
        damped_gauss_newton,
        GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE,
        GAUSS_NEWTON_INITIAL_DAMPING,
        GAUSS_NEWTON_MAX_ITERS,
        GAUSS_NEWTON_MIN_DAMPING,
        GAUSS_NEWTON_TOLERANCE,
        RowFunction,
    )
    from .mlalign.helix.measurement_fit import (  # noqa: F401
        _BATCHED_MEASUREMENT_RESIDUAL_JACOBIAN,
        _fit_helix_measurements_gauss_newton_block,
        _measurement_standardized_residual_tensors,
        _require_well_constrained_measurement_rows,
        _single_measurement_residual,
        fit_helix_measurements_gauss_newton,
        MEASUREMENT_FIT_CHUNK_ROWS,
        measurement_local_prediction,
        measurement_newton_decrement,
        MEASUREMENT_NEWTON_DECREMENT_STATIONARY,
        measurement_residual_jacobian,
        measurement_standardized_residual,
        measurement_stationarity_report,
        MEASUREMENT_VARIANCE_FLOOR,
        MeasurementFitInputs,
    )
    from .mlalign.helix.point_fit import (  # noqa: F401
        _BATCHED_POINT_FIT_JACOBIAN,
        _fit_helix_points_gauss_newton_block,
        _single_point_fit_effective_mask,
        _single_point_fit_residual,
        _single_point_fit_residual_and_mask,
        fit_helix_points_gauss_newton,
        POINT_FIT_MAX_NON_STATIONARY_ROW_FRACTION,
        require_stationary_helix_point_fit,
    )
    from .mlalign.helix.trajectory import (  # noqa: F401
        _sinc,
        accept_newton_candidate,
        closest_helix_points,
        CLOSEST_POINT_CURVATURE_EPS,
        helix_point,
        helix_tangent,
        intersect_plane,
        PLANE_PARALLEL_EPS,
        PLANE_RESIDUAL_TOLERANCE,
        S_MAX,
    )
except ImportError:  # Direct script execution.
    from mlalign.helix.gauge import (  # noqa: F401
        assemble_track_parameters,
        CURVATURE_SCALE,
        pack_gauge_fixed_track_parameters,
        reference_gauge_frame,
        split_track_parameters,
        unpack_gauge_fixed_track_parameters,
    )
    from mlalign.helix.gauss_newton import (  # noqa: F401
        _merge_gauss_newton_diagnostics,
        _stationary_newton_decrement,
        damped_gauss_newton,
        GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE,
        GAUSS_NEWTON_INITIAL_DAMPING,
        GAUSS_NEWTON_MAX_ITERS,
        GAUSS_NEWTON_MIN_DAMPING,
        GAUSS_NEWTON_TOLERANCE,
        RowFunction,
    )
    from mlalign.helix.measurement_fit import (  # noqa: F401
        _BATCHED_MEASUREMENT_RESIDUAL_JACOBIAN,
        _fit_helix_measurements_gauss_newton_block,
        _measurement_standardized_residual_tensors,
        _require_well_constrained_measurement_rows,
        _single_measurement_residual,
        fit_helix_measurements_gauss_newton,
        MEASUREMENT_FIT_CHUNK_ROWS,
        measurement_local_prediction,
        measurement_newton_decrement,
        MEASUREMENT_NEWTON_DECREMENT_STATIONARY,
        measurement_residual_jacobian,
        measurement_standardized_residual,
        measurement_stationarity_report,
        MEASUREMENT_VARIANCE_FLOOR,
        MeasurementFitInputs,
    )
    from mlalign.helix.point_fit import (  # noqa: F401
        _BATCHED_POINT_FIT_JACOBIAN,
        _fit_helix_points_gauss_newton_block,
        _single_point_fit_effective_mask,
        _single_point_fit_residual,
        _single_point_fit_residual_and_mask,
        fit_helix_points_gauss_newton,
        POINT_FIT_MAX_NON_STATIONARY_ROW_FRACTION,
        require_stationary_helix_point_fit,
    )
    from mlalign.helix.trajectory import (  # noqa: F401
        _sinc,
        accept_newton_candidate,
        closest_helix_points,
        CLOSEST_POINT_CURVATURE_EPS,
        helix_point,
        helix_tangent,
        intersect_plane,
        PLANE_PARALLEL_EPS,
        PLANE_RESIDUAL_TOLERANCE,
        S_MAX,
    )


_BROKEN_HELIX_COMPAT_NAMES = {
    "HIGHLAND_ENERGY_GEV",
    "HIGHLAND_LOG_COEFFICIENT",
    "KINK_THETA_EPS",
    "MUON_MASS_GEV",
    "assign_hit_segments",
    "broken_helix_local_path",
    "broken_helix_segments",
    "broken_helix_track_parameters",
    "highland_kink_sigma",
    "node_arc_lengths",
    "segment_origin_path",
}


def __getattr__(name: str):
    """Lazily preserve historical imports of experimental broken-helix symbols."""
    if name not in _BROKEN_HELIX_COMPAT_NAMES:
        raise AttributeError(name)
    try:
        from . import broken_helix_model
    except ImportError:  # Direct script execution.
        import broken_helix_model
    value = getattr(broken_helix_model, name)
    globals()[name] = value
    return value
