import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import acts  # noqa: F401
except ModuleNotFoundError:
    acts = None


@unittest.skipUnless(acts is not None, "ACTS Python bindings are unavailable")
class LocalRotationTest(unittest.TestCase):
    def test_detector_factory_uses_configured_odd_root(self):
        import alignment_odd

        configured_root = Path("/tmp/configured-odd").resolve()
        sentinel = object()
        with (
            patch.object(
                alignment_odd,
                "PROJECT_CONFIG",
                type("Config", (), {"odd_root": configured_root})(),
            ),
            patch(
                "acts.examples.odd.getOpenDataDetector", return_value=sentinel
            ) as factory,
        ):
            detector = alignment_odd.get_configured_open_data_detector(
                misaligned=True
            )

        self.assertIs(detector, sentinel)
        factory.assert_called_once_with(
            misaligned=True,
            odd_dir=configured_root,
        )

    def test_rz_is_applied_about_the_local_z_axis(self):
        import alignment_odd

        tilt = 0.6
        angle = 0.1
        cos_tilt = math.cos(tilt)
        sin_tilt = math.sin(tilt)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        nominal_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, cos_tilt, -sin_tilt],
            [0.0, sin_tilt, cos_tilt],
        ]
        expected_rotation = [
            [cos_angle, -sin_angle, 0.0],
            [cos_tilt * sin_angle, cos_tilt * cos_angle, -sin_tilt],
            [sin_tilt * sin_angle, sin_tilt * cos_angle, cos_tilt],
        ]
        element = {
            "ID": "vol=1|lay=2|sen=3",
            "Translation": [0.0, 0.0, 0.0],
            "RotationMatrix": nominal_rotation,
        }

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.object(alignment_odd.random, "random", return_value=0.75),
            patch.object(alignment_odd.random, "uniform", return_value=angle),
            redirect_stdout(io.StringIO()),
        ):
            _, _, records = alignment_odd.setupMisalignment(
                [element],
                Path(output_dir),
                target_volume=-1,
                tx=0,
                ty=0,
                tz=0,
                rx=0,
                ry=0,
                rz=1,
                rotation_mag_rad=angle,
            )

        np.testing.assert_allclose(
            records[0]["RotationMatrix"], expected_rotation, rtol=0.0, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
