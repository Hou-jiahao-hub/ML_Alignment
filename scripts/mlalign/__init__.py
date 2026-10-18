"""ACTS-independent alignment core shared by the ``ml_alignment_autograd`` CLI.

Stage modules follow the three key optimisations: ``stage_surface`` (frozen KF
tracks, Adam), ``stage_ntr_helix`` (non-target-refitted helices with fixed-track
block-LM) and ``joint_alignment_schur`` (joint Schur-reduced LM, kept at the
top level).  Everything else here is shared by at least one of them.
"""
