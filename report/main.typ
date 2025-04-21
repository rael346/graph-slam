#import "@preview/charged-ieee:0.1.3": ieee

#set text(font: "New Computer Modern")

#show: ieee.with(
  title: [Yet Another Implementation of Graph-based SLAM],
  abstract: [],
  authors: (
    (
      name: "Duy Tran",
      department: [Khoury College of Computer Science],
      organization: [Roux Insititute at Northeastern University],
      location: [Portland, Maine],
      email: "tran.duy3@northeastern.edu",
    ),
  ),
  index-terms: (
    "Robotics",
    "SLAM",
    "Factor Graph",
    "Backend",
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Graph-based SLAM is currently the state-of-the-art SLAM
backend technique for optimizing pose graphs. This paper
will discuss in further detail an implementation of the
main algorithm introduced in @grisettti2010 for 2D graphs
and tested on Luca Carlone's 2D Pose Graph Optimization Dataset at
#link("https://lucacarlone.mit.edu/datasets/")[lucacarlone.mit.edu/datasets]
@carlone2014. The implementation can be found at
#link("https://github.com/rael346/graph-slam")[github.com/rael346/graph-slam]

= Related Works <sec:related>

There have been many implementations of graph-based SLAM algorithm over the years in C++
like Ceres @ceres, which is a general non-linear optimization library, and g2o @g2o, which
is a non-linear optimization library with a focus on SLAM-related optimization problem.

This project, similar to g2o, will implement the non-linear optimization algorithm with a focus
on solving 2D pose graphs discussed in @grisettti2010. The project was also inspired by the work
of Irion @python-graphslam, implementing the algorithm entirely in Python.

= Background <sec:background>

The crux of the algorithm is not in the optimization algorithm, which was discussed in detail in
Algorithm 2 of @grisettti2010, but the manifold representation of a robot pose. Therefore, this
section will go into detail the manifold representation and the math behind it.

== Special Euclidean Group SE(2)

A robot pose in space is described by its position and orientation with respect to an origin and axes.
For 2D space, the position is a set of $(x, y) in RR^2$ coordinates and a rotation angle $theta in [-pi, pi]$.
This representation can also be understood as a transformation from the origin using the transformation matrix
$
  T = mat(
        R, t;
        0, 1;
      )
$ <eq:se2_full>

with $R$ being the rotation matrix of the transformation
$
  R = mat(
        cos(theta), -sin(theta);
        sin(theta), cos(theta);
      )
$ <se2_rotation>

and $t = (x, y)$ being the translation of the transformation.

Thus we have a compact representation $(x, y, theta)$ and the full representation in matrix form from equation
@eq:se2_full. The compact representation is mostly used to save memory since it requires only 3 floats
compared to the full representation which requires 9 floats.

The full representation also belongs to a special group called Special Euclidean Group, which
is mathematically defined as
$
  "SE"(2) = { T = mat(
          R(theta), t;
          0, 1;
        )
    | R in "SO"(2), t in RR^2 }
$ <eq:se2>

$
  "SO"(2) = { R in RR^(2 times 2) | R R^T = I, det(R) = 1}
$ <eq:so2>

== SE(2) Operations

This section will layout the main operations for poses in SE(2) form required to implement the
optimization algorithm, which was discussed in length in @blancoclaraco2022. The difference is
that @blancoclaraco2022 developed the full mathematical framwork for Special Euclidean Group
while this section only discuss the relevant operations for the algorithm. Furthermore, this
section serves as a somewhat more comprehensive tutorial to the math dicussed in @grisettti2010
for the 2D case, since @grisettti2010 only show the resulting equation without much elaboration.

For the rest of the section, we will refer to $p = (x, y, theta)$ as the compact representation
of the robot pose and $T$ as the full representation described in equation @eq:se2_full.

=== Inverse

$
  T^(-1)
  stretch(=)^"full" mat(
    R(theta)^top, -R(theta)^top t;
    0, 1;
  )
  stretch(=)^"compact" mat(
    -R(theta)^top t;
    -theta;
  )
$

This can be verified by $T T^(-1) = I$. Note that $R(theta)^top = R(-theta)$.
While this operation wasn't used directly in the implementation, it is a useful one
for later equation derivation.

=== Composition

$
  p_1 plus.circle p_2 =& T_1 T_2\
  stretch(=)^"full"& mat(
    R(theta_1 + theta_2), R(theta_1)t_2 + t_1;
    0, 1;
  )\
  stretch(=)^"compact"& mat(
    R(theta_1)t_2 + t_1;
    theta_1 + theta_2;
  )
$

This operation also wasn't used directly in the implementation, but is the
basis for the next operation.

=== Inverse Composition

$
  p_1 minus.circle p_2 =& T_2^(-1) T_1\
  stretch(=)^"full"& mat(
    R(theta_1 - theta_2), R(theta_2)^top (t_1 - t_2);
    0, 1;
  )\
  stretch(=)^"compact"& mat(
    R(theta_2)^top (t_1 - t_2);
    theta_1 - theta_2;
  )
$ <eq:inv_comp>

This operation is the crux of the optimization problem, since it is used in
the error calculation. Note that the original tutorial @grisettti2010 didn't
use this operation, but instead use the inverse and composition of poses. The
inverse composition is the same mathematically, but is represented more succinctly
as a single operation. @blancoclaraco2022 used this operation extensively to describe
the error calculation.

== Error Function

In graph-based SLAM, the vertices are the poses of the robot at different timestamp,
whereas the edges represent the diference between the two poses $p_i$ and $p_j$. This
difference can also be understood as the pose of the robot at time $j$ relative to the
pose at time $i$.

The edges also contain the measured difference $z_(i j)$ gotten from the frontend of the SLAM
system (usually processed from the raw sensor measurements). Thus the difference between the
expected difference and the measured difference is defined as

$
  e_(i j) = (p_j minus.circle p_i) minus.circle z_(i j)
$ <eq:error>

Note that since the poses inverse composition is not commutative, the ordering of the poses
here is actually important. This equation also corresponds with equation 30 in @grisettti2010,
which describes the error function in its full derivation.

== Jacobians

This is another important section since the optimization algorithm relies heavily on the
Jacobians calculated from the pose operations.

=== Jacobians of Inverse Composition

With $Delta t = R(theta_2)^top (t_1 - t_2)$ and $Delta theta = theta_1 - theta_2$ from
equation @eq:inv_comp we have

$
  diff(p_1 minus.circle p_2) / (diff p_1)
  = mat(
      diff(Delta t)/(diff t_1), diff(Delta t)/diff(theta_1);
      diff(Delta theta)/(diff t_1), diff(Delta theta)/diff(theta_1);
    )
  = mat(
      R(theta_2)^top, 0;
      0, 1;
    )
$

$
  diff(p_1 minus.circle p_2) / (diff p_2)
  &= mat(
      diff(Delta t)/(diff t_2), diff(Delta t)/diff(theta_2);
      diff(Delta theta)/(diff t_2), diff(Delta theta)/diff(theta_2);
    )\
  &= mat(
      -R(theta_2)^top, diff(R(theta)^T)/diff(theta_2) (t_1 - t_2);
      0, -1;
    )
$

=== Jacobians of Error Function

Deriving the Jacobians of the error function @eq:error is simply using the chain rule

$
  A_(i j)
  = (diff e_(i j)) / (diff p_i)
  = (diff e_(i j)) / diff(p_j minus.circle p_i) diff(p_j minus.circle p_i) / (diff p_i)
$

$
  B_(i j)
  = (diff e_(i j)) / (diff p_j)
  = (diff e_(i j)) / diff(p_j minus.circle p_i) diff(p_j minus.circle p_i) / (diff p_j)
$


=== Jacobians of Manifold

$
  M_i
  = (diff p_i plus.square Delta p_i) / (diff Delta p_i)
  = I
$

$
  M_j
  = (diff p_j plus.square Delta p_j) / (diff Delta p_j)
  = I
$

*Author Note*: this is the only part I couldn't understand since the original tutorial
@grisettti2010 didn't show any derivation. @blancoclaraco2022 didn't use this Jacobians
at all so it is hard to verify the math here. It probably has something to do with the
Lie Algebra and the Lie Group (SE(2) is a Lie Group) but I didn't have enough time to
flesh this out.

= Implementation Details <sec:details>

== Dataset Format

The dataset use the g2o format, the description of which can be found on Luca Carlone's
dataset website. Essentially, each line of the file is either
- The poses of the graph in SE(2) and has the format
  `VERTEX_SE2 id x y theta`
- The edge of the graph and has the format
  `EDGE_SE2 IDout IDin dx dy dtheta i11 i12 i13 i22 i23 i33`
  - `IDout` and `IDin` are the IDs of the poses that the edges connected
  - `dx dy dtheta` is the measured difference between the two poses, aka $z_(i j)$
  - `i11 i12 i13 i22 i23 i33` is the 3x3 information matrix, which is triangular

== Objects

After defining the core operations and Jacobians above, the implementation is fairly straight
forward.
- A `SE2` object that implements the above operations and Jacobians
- An `Edge` object that represents the edge of the graphs and stores the infomation matrix and
  the meaasured difference
- A `Graph` object that parses the dataset in g2o format and stores the list of poses in `SE2`
  form as well as the edges of the graph, and implements Algorithm 2 in the original tutorial
  @grisettti2010

== Libraries

This was implemented using Python 13 using Numpy for the matrix operations, Scipy for sparses matrix
solver, and Matplotlib for visualizing the pose graph before and after optimization.

#pagebreak()

= Results <sec:results>

#columns(2)[
  #figure(
    image("./results/intel_before.png", width: 100%),
    caption: [Intel Research Lab Before Optimization],
  )
  #colbreak()
  #figure(
    image("./results/intel_after.png", width: 100%),
    caption: [Intel Research Lab After Optimization],
  )
]

#columns(2)[
  #figure(
    image("./results/M3500_before.png", width: 100%),
    caption: [M3500 Before Optimization],
  )
  #colbreak()
  #figure(
    image("./results/M3500_after.png", width: 100%),
    caption: [M3500 After Optimization],
  )
]

For the first three datasets (Intel, MIT, M3500), which contains relatively less noise,
the algorithm converges to the global minima.

#columns(2)[
  #figure(
    image("./results/M3500a_before.png", width: 100%),
    caption: [M3500a Before Optimization],
  )
  #colbreak()
  #figure(
    image("./results/M3500a_after.png", width: 100%),
    caption: [M3500a After Optimization],
  ) <fig:m3500a_after>
]

#columns(2)[
  #figure(
    image("./results/M3500b_before.png", width: 100%),
    caption: [M3500b Before Optimization],
  )
  #colbreak()
  #figure(
    image("./results/M3500b_after.png", width: 100%),
    caption: [M3500b After Optimization],
  ) <fig:m3500b_after>
]

#columns(2)[
  #figure(
    image("./results/M3500c_before.png", width: 100%),
    caption: [M3500c Before Optimization],
  )
  #colbreak()
  #figure(
    image("./results/M3500c_after.png", width: 100%),
    caption: [M3500c After Optimization],
  ) <fig:m3500c_after>
]

For the variation of M3500 with extra noises to the relative orientation,
the algorithm struggled to find the global minima and either converges to
a local minima (@fig:m3500a_after) or not converges at all (@fig:m3500b_after).

All datasets were run with 20 max iteration, except for M3500b, which it still
fails to converges. Interestingly, M3500c manages to converges (@fig:m3500c_after)
to a local minima even though it has more noises than M3500b.

= Discussion

== Future Works

Similar to gradient descent, the non-linear optimization approach presented here
is still subjected to local minima convergence. Possible solutions to this problem
includes apply learning rate during the pose update step so that the algorithm
can slowly converges to the minima instead of bouncing around it; using an optimizer
like Adam or AdamW to "nudge" the algorithm to a global minima. However, such solutions
might not be necessary when taking into account the whole SLAM system. Within a SLAM
pipeline, pose graph optimization is usually only triggered when the robot completes
a loop (this process is called loop closure). Therefore, the pose graph is usually being
optimized incrementally instead of all at once as implemented here, which can reduce
the overall noise and graph complexity.

Another direction is implementing SE(3) poses, which represents the robot pose
in 3-dimension space. However, the 3D case is significantly more complex than the 2D
case because of the rotation being quarternions instead of a single angle $theta$.

Testing the algorithm by incorporating it with a frontend like Visual SLAM could be
an interesting next step, since it show whether the algorithm is robust enough in
real-time SLAM.

== Implementation Challenges

One of the biggest challenges implementing this algorithm is verifying if the math is
correct. The original tutorial @grisettti2010 didn't show the full derivation of the pose operators,
error function and jacobians, whereas @blancoclaraco2022 uses a completely different
notation system and usually showed the full form of the result element-by-element matrices.
This is fairly error prone since the full matrices are usually large, so a single plus/minus
can derail the whole algorithm. This is the main reason why I dedicated a whole section about
SE(2) above, since it significantly simplify the math by using the block matrices $R$ and $t$
instead of going element-by-element. In addition, having a reference implementation like Irion's @python-graphslam was very helpful in debuging the algorithm.
