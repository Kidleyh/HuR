# Human Temporal V2 (GVHMR)

This optional stage runs the official GVHMR model in-process on each HuR
logical person's observed frames and existing person boxes. It does not run a
new person detector, does not alter logical tracks, and does not participate in
the current reward.

The adapter uses the upstream `DemoPL.predict()` path with static-camera
inputs. HuR supplies the observed RGB frames and person boxes to the upstream
ViTPose and HMR2 feature preprocessors. `smpl_params_global` is retained in
memory, and the official GVHMR endecoder produces 22 global 3D joints.

For every logical person, `person.temporal.human_3d` contains root-relative
joint velocity, acceleration, and jerk plus global root velocity and
acceleration. Frame intervals are computed from the original frame indices and
video FPS. Each per-frame joint value is the P90 across finite body joints;
the result also contains mean, P90, maximum, frame metrics, and the five worst
acceleration and jerk frames. `score` remains `null`.

GVHMR, HMR2, ViTPose, and body-model weights must be installed locally. The
module uses lazy imports and never downloads resources. See
`docs/RUN_COMMANDS.md` for commands and expected paths.

Current limitations:

- GVHMR treats a logical person's observed frames as one sequence; it does not
  fill missing HuR observations.
- Metrics may expose noise from monocular 3D recovery as well as true motion.
- No contact, balance, torque, GRF, foot-slip, MuJoCo, or temporal reward is
  implemented.
