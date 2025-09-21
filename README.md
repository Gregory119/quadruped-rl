# Pedipulation
## Rewards
### Termination Conditions (Hard Limits)
If it is impossible for the robot to continue interacting with the environment,
then an episode should terminate with a zero reward. This means the accumulated
reward will not continue to grow in the future, so the policy should avoid
it. The nice thing is that the accumulated reward remains positive. This
includes the following condition(s):
- robot falls over (detect with gravity vector)
- hard torque limits

### Soft Limit Violations
These are undersireable limit violations, but it should be possible for the
robot to recover, in which case the policy should be encouraged to do so by not
terminating but reducing the return. These limits are more conservative than the
hard limits so that they occur before hard limits, helping avoid the hard
limits. These limits include:
 - torque limits
 - any self-collision
 - any collision between ground and robot (excluding feet)
 
Discount the entire step reward by a scale factor < 1 if any soft limit is
violated. For each instantaneous event this scale factor reduces by an amount
depending on the type and occurence count of an event. For violations that can
occur over a duration the contributing scale factor reduces as it gets closer to
the hard limit.
 
### Negative Rewards
Using negative rewards will affect all state actions. These are primarily used
to affect the way in which motion occurs accounting for safety and stability.
- negative reward based on total robot energy. This will act as velocity damping
  weighted by masses as well as a safety precaution. This will reduce the max
  slope of displacement vs time.
- negative reward based on robot power (this is like damping the motion,
  accounting for velocity, acceleration, and inertia). For example, kinetic
  energy of point mass is 0.5*m*v^2 so power (rate of change of energy) is
  m*v*a. So inertia acts like a weighting (motion of higher mass links is more
  important). Lower acceleration is allowed for higher velocities, but higher
  acceleration is allowed for lower velocities. This could cause the motion to
  follow a smooth S curve. This makes intuitive sense because its more important
  in terms of safety that a heavy large link isn't oscillating, than a small
  light link.

## state variables
In two of the Marc Hutter papers the pose of the robot isn't used in the
state. The gravity vector represented in the base frame acts as orientation
information independent of the global frame. There seems to be a pattern of
intentionally only using variables that can be represented agnostically to some
global frame. If dependence on a global frame existed, then this global frame
could be anywhere in a large space, resulting in less similar state variables
even if relative variables between robot and object are similar, causing the
policy to have to learn these similarities => larger learning time.

### robot global position
- gravity vector in robot base frame: acts as robot orientation for stability
- height of robot base from flat ground: help with knowing if robot feet are in
  contact with ground
 
### robot velocity
- robot base linear velocity w.r.t global frame, represented in base frame
- robot base angular velocity w.r.t global frame, represented in the base frame
   
### desired foot position
- Desired foot position relative to robot base frame. Before given to policy,
  the desired foot position can be w.r.t a global frame and then transformed to
  the robot base frame before being given to the pedipulate policy.
  
### joints
- joint positions and velocities

## action
- absolute joint positions

# Reorient Policy
## state variables

### relative velocity between object and robot
- linear velocity of object w.r.t robot frame, represented in the robot frame
- angular velocity of object w.r.t robot frame, represented in the robot frame

### relative pose between object and robot
- position of object frame relative to robot base frame, represented in the robot
  base frame
- rotation matrix (or quarternion) of object frame w.r.t robot base frame

# Frames
- `trunk`: body frame referred to as the `base` frame, attached to the main body
  of the robot
- `FR`: `site` at the center of the front right foot (get its frame)

# TODO
- read about reward shaping for safety and hard/soft limit violations

# Questions
- why is previous action used in state? (research this) maybe action rate
  regularization?

- MAYBE: position of right front foot relative to object, represented in the
  robot base frame
