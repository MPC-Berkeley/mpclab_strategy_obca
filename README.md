# mpclab_strategy_obca
Collision Avoidance in Tightly-Constrained Environments without Coordination: a Hierarchical Control Approach

[Webpage](http://bit.ly/data-sg-control),  [arXiv](https://arxiv.org/abs/2011.00413)

Authors:
- Xu Shen, xu_shen@berkeley.edu
- Edward Zhu, edward_zhu@berkeley.edu
- Yvonne R. Stürz, y.stuerz@berkeley.edu

A ROS and python package containing the code for online receding horizon OBCA control in tightly-constrained environments leveraging data-driven strategy prediction.

This package is intended for use with the [`BARC_research`](https://github.com/MPC-Berkeley/BARC_research.git) simulation and experiment codebase.

To use the controllers in this package, navigate to the root directory of this repository and run `catkin_make`. The `setup.bash` file from this package should be sourced after the one from `BARC_research` using the following command from the root directory
```
source ./devel/setup.bash --extend
```
The `--extend` flag stops overwriting from sourcing the `setup.bash` from this package.

Dependencies:
- APT: `libgmp3-dev` (required for `pypoman`)
- Python: `numpy`, `scipy`, `torch`, `pypoman`, `pyclipper`, `casadi==3.5.1`, [`forcespro`](https://forces.embotech.com/Documentation/installation/python.html)
