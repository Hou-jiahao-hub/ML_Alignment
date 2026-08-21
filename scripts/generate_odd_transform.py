import json
import numpy as np
import math
import acts
from acts.examples.odd import getOpenDataDetector
import argparse

from alignment_artifact_layout import PROJECT_CONFIG, data_path
from project_config import absolute_project_path, add_config_argument

def extract_euler_angles(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return [x, y, z]

def geoid_to_str(geo_id):
    parts = [
        f"vol={geo_id.volume}",
        f"lay={geo_id.layer}",
        f"sen={geo_id.sensitive}"
    ]
    if geo_id.extra != 0:
        parts.append(f"ext={geo_id.extra}")
    return "|".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Generate ODD transform data")
    add_config_argument(parser)
    parser.add_argument("--material-config", type=absolute_project_path, default=None,
                        help="Custom path to ODD material map file")
    args = parser.parse_args()

    geoDir = PROJECT_CONFIG.odd_root

    oddMaterialMap = (
        args.material_config
        if args.material_config
        else PROJECT_CONFIG.odd_material_map
        )

    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

    print("loading Open Data Detector...")
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()

    gctx = acts.GeometryContext()

    transform_data = []

    sensitive_surface = []
    def surface_visitor(surface):
        if surface.isSensitive:
            sensitive_surface.append(surface)

    trackingGeometry.visitSurfaces(surface_visitor)
    print(f"finding {len(sensitive_surface)} sensitive surfaces, the transformation matrix is being extracted")

    for count, surface in enumerate(sensitive_surface, start = 1):
        geo_id = surface.geometryId
        transform = surface.localToGlobalTransform(gctx)

        translation = transform.translation
        rot_matrix = transform.rotation

        translation_list = []
        for i in range(3):
            try:
                val = translation[i, 0]
            except (IndexError, TypeError):
                val = translation[i]
            translation_list.append(float(val))

        rotation_list = [[float(rot_matrix[i, j]) for j in range(3)] for i in range(3)]

        translation_np = np.array([translation_list])
        rotation_np = np.array(rotation_list)
        
        full_matrix = np.eye(4)
        full_matrix[0:3, 0:3] = rotation_np
        full_matrix[0:3, 3] = translation_np

        rotation = extract_euler_angles(rotation_np)

        transform_data.append({
                "Count": count,
                "ID": geoid_to_str(geo_id),
                "Translation": translation_list,
                "Rotation": rotation,
                "RotationMatrix": rotation_np.tolist(),
                "FullMatrix": full_matrix.tolist()
                })
    
    output_filename = data_path("odd_transforms.json")
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w")as f:
        json.dump(transform_data, f, indent=2)

    print(f"odd_transform saved to: {output_filename}")

if "__main__" == __name__:
    main()
