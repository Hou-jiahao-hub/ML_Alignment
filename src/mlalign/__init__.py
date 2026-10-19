"""Joint track-geometry Schur-LM alignment of tracking-detector surfaces.

Reads a geometry JSON and the ACTS ``trackstates`` it was reconstructed with,
refits every track as a six-parameter helix and solves geometry and tracks
jointly with a Schur-reduced Levenberg-Marquardt.  ACTS-independent; runs on
the CPU in float64.  Defaults live in ``mlalign.config``.

Monkeypatch a name in its defining module (e.g. ``mlalign.pipeline``), never
through the re-exports below.
"""

# uproot/awkward must load before torch: torch otherwise pulls in the system
# libstdc++, which lacks the CXXABI version awkward_cpp needs.
import awkward  # noqa: F401
import uproot  # noqa: F401

from .config import AlignmentConfig
from .pipeline import run_alignment
from .reporting import write_alignment_summary

__version__ = "0.2.0"

__all__ = ["AlignmentConfig", "run_alignment", "write_alignment_summary", "__version__"]
