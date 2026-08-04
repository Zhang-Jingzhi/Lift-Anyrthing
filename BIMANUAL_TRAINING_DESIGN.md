# Bimanual training design

The current pipeline uses the official Allegro left-hand asset and a mirrored
right-hand asset. The right asset passed exact kinematic/mesh/inertial mirror
checks and an Isaac load/DOF/body/limit validation. Both validation reports are
stored beside the tracked asset under `runtime_assets/`.

## Representation

- Each hand has a 22-dimensional configuration.
- A training target stores the two configurations as a 44-dimensional pair.
- Forward kinematics produces 21 SE(3) link nodes per hand and 42 robot nodes
  in the diffusion graph.
- The two 21-link blocks share the Allegro embodiment embedding, while fixed
  role anchors preserve the left/right assignment.

## Sampling and symmetry

- An epoch schedules the six development objects equally before sampling a
  strict grasp from the chosen object.
- The strict left/right configuration keeps role swapping disabled by default.
- Optional swap-equivariance remains available for symmetric ablations.

## Optimization terms

- Translation and rotation diffusion losses: original TRO-Grasp objective.
- Hand-object penetration: differentiable nearest-surface point-to-plane
  penalty using object normals.
- Dual contact: each hand is penalized when its closest surface point is more
  than 5 mm from the object.
- Inter-hand collision: the closest cross-hand point pair is penalized below
  the configured clearance.
- Lateral opposition and relative-root terms discourage one-sided or
  top/bottom grasp layouts.

Geometry losses operate on a clean-pose estimate reconstructed from the noisy
diffusion state and predicted noise.

## Hard post-generation gates

Joint limits are not represented directly by the link-SE(3) denoiser. They
are therefore enforced after inverse kinematics, together with exact mesh
penetration, bilateral realized contact, exact hand-hand clearance, tabletop
approach/closure clearance, gravity-and-lift success, and six independent
Isaac disturbances. Left-only and right-only gravity ablations reject samples
that do not genuinely need cooperation. The local cooperative wrench audit and
repeat rollout verification provide further quality gates.
