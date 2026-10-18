"""Box bounds on the geometry corrections, shared by block-LM and schur-lm."""

from __future__ import annotations

JOINT_CLAMP_T = 1.5
JOINT_CLAMP_R = 0.05

#: Tolerance for "sitting on a box constraint".  ``clamp_`` makes the equality
#: exact, so this only absorbs the round trip through the candidate row.
BLOCK_LM_BOUND_ATOL = 1.0e-12


def _block_lm_free_dof_mask(values, gradient, lower, upper, atol=BLOCK_LM_BOUND_ATOL):
    """Active-set mask for the box constraints: which DOFs may still move.

    From the projected gradient ``P = x - clip(x - g, l, u)`` with a descent
    direction of ``-g``: at the upper bound ``g < 0`` wants to push further up
    and is infeasible, at the lower bound ``g > 0`` wants to push further down.
    Those DOFs are frozen; everything else, an interior point or a bound whose
    gradient points back inside, stays free.  ``g == 0`` never freezes -- it
    contributes nothing to the decrement and is better left visible to the rank
    check.

    ``values``/``lower``/``upper`` are physical, ``gradient`` is normalized.
    Mixing them is safe because only the sign of the gradient is read, and
    ``physical_scales`` is strictly positive.
    """
    at_upper = values >= upper - atol
    at_lower = values <= lower + atol
    return ~((at_upper & (gradient < 0.0)) | (at_lower & (gradient > 0.0)))
