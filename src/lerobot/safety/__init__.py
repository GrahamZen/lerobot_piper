"""
lerobot.safety
==============

Safety filter for real-robot inference and simulation.

Provides a QP-based action projection that keeps critical end-effector links
inside a configurable 3-D box constraint during policy execution.

Typical usage (real robot / ACT policy)
---------------------------------------

    from lerobot.safety import (
        BoxConstraintConfig,
        RobotKinematicsInfo,
        enforce_safe_action,
        load_constraint_config,
        make_box_handles,
    )

    box  = load_constraint_config("tools/safety/constraint_config.json")
    kin  = RobotKinematicsInfo(urdf_path)
    hdls = make_box_handles(box)

    # Inside the inference loop:
    q_safe = enforce_safe_action(q_from_policy, kin, hdls)
"""

from lerobot.safety.safety_filter import (
    BoxConstraintConfig,
    RobotKinematicsInfo,
    approximate_gradient,
    enforce_safe_action,
    load_constraint_config,
    make_box_handles,
)

__all__ = [
    "BoxConstraintConfig",
    "RobotKinematicsInfo",
    "approximate_gradient",
    "enforce_safe_action",
    "load_constraint_config",
    "make_box_handles",
]
