# Mirrored Allegro right-hand asset

This asset is a deterministic mirror of the repository's modified Allegro
left-hand URDF. It is intended for symmetric left/right simulation, not as a
manufacturer-certified replacement for hardware calibration files.

The physical hand is reflected through the palm-local `y=0` plane. The six
virtual base joints are deliberately not reflected, so both assets retain the
same world `xyz/rpy` generalized-coordinate convention. Revolute axes are
transformed as pseudovectors, origins and inertias are mirrored, OBJ vertices
are reflected, and triangle winding is reversed.

Registered names:

- `allegro`: legacy alias for the left hand;
- `allegro_left`: explicit left hand;
- `allegro_right`: mirrored right hand.

Rebuild and validate from the repository root:

```bash
conda run -n tro python scripts/build_allegro_right.py
conda run -n tro python \
  scripts/validate_allegro_mirror.py --samples 100 --tolerance 1e-8
"$ISAAC_PYTHON" validation/validate_allegro_assets_isaac.py --gpu 0
```

The offline validation checks URDF structure, joint limits, 100 random forward
kinematics samples, visual and collision geometry, inertial frames and tensors,
mass, and robot point clouds. The Isaac validation loads both assets, compares
their DoF/body/shape metadata, and advances a two-actor simulation.
