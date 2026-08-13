import json
import sys

import numpy as np
import torch
import trimesh
from yourdfpy import URDF

from xhand_visual_mesh_gate import (
    URDF_PATH,
    _hand_surface_points,
    materialize_actual_closure_state,
)


stem = sys.argv[1]
directory = f"migration_4090/min_validation_v2/{stem}"
sample = torch.load(
    f"{directory}/sphere_candidates.pt", map_location="cpu", weights_only=False
)["samples"][0]
report = json.loads(open(f"{directory}/both.json").read())
actual = materialize_actual_closure_state(sample, report)
robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
values = dict(zip(actual["joint_names"], actual["full_body_q"]))
robot.update_cfg(np.asarray([values[name] for name in robot.actuated_joint_names]))
obj = trimesh.load_mesh(actual["object_mesh_path"], force="mesh", process=False)
obj.apply_transform(np.asarray(actual["object_pose_world"]))
offset = np.asarray(actual.get("isaac_robot_root_alignment_offset_xyz", [0, 0, 0]))
print("object_bounds", obj.bounds.tolist())
for side in ("left", "right"):
    points = _hand_surface_points(robot, side) + offset
    closest, distances, _ = trimesh.proximity.closest_point(obj, points)
    i = int(np.argmin(distances))
    print(side, "min_mm", float(distances[i] * 1000), "near20", int(np.sum(distances < 0.02)), "near5", int(np.sum(distances < 0.005)))
    print("hand", points[i].tolist(), "object", closest[i].tolist())
