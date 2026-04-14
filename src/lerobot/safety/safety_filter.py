"""
Safety filter based on Pinocchio kinematics and QP optimization.

Designed for both simulation (unit tests) and real-robot deployment.
No MuJoCo dependency.

Key design
----------
* Works in **logical DOF** space (e.g. 7 for single arm, 14 for dual arm),
  which is what the real robot policy produces.  Mimic/coupled joints (e.g.
  the symmetric gripper fingers) are handled internally.

* RobotKinematicsInfo auto-detects mimic joints from the URDF and exposes
  expand() / contract() helpers for the logical ↔ full-Pinocchio conversion.

* enforce_safe_action() accepts and returns logical-DOF arrays, so it can be
  dropped directly into the real-robot action pipeline with zero changes.

Typical usage
-------------
Single arm (7-DOF policy output):

    kin = RobotKinematicsInfo("piper_description.urdf")
    q_safe = enforce_safe_action(q_7, kin, lambda p: p[2] - 0.05)

Dual arm (14-DOF policy output, [left_7, right_7]):

    kin = RobotKinematicsInfo("piper_dual_description.urdf")
    q_safe = enforce_safe_action(q_14, kin, lambda p: p[2] - 0.05)
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin
import scipy.sparse as sp
from qpsolvers import solve_qp

# Patterns used to auto-detect critical links when none are specified.
# Matches both plain names ("link4") and prefixed ones ("left_link4").
_CRITICAL_PATTERNS: tuple[str, ...] = ("link4", "link6", "gripper_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_mimic_joints_from_urdf(
    urdf_path: str,
) -> list[tuple[str, str, float]]:
    """
    Parse a URDF and return [(mimic_joint_name, primary_joint_name, multiplier)].

    Detection strategy (in order):
    1. Explicit <mimic> tags in the URDF.
    2. Prismatic joint pairs that share the same parent link (symmetric
       gripper fingers — the fallback used by Piper URDFs that omit <mimic>).
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(urdf_path)).getroot()

    # Strategy 1: explicit <mimic> tags
    pairs: list[tuple[str, str, float]] = []
    for joint in root.iter("joint"):
        mimic_el = joint.find("mimic")
        if mimic_el is not None:
            pairs.append((
                joint.get("name", ""),
                mimic_el.get("joint", ""),
                float(mimic_el.get("multiplier", "1")),
            ))

    if pairs:
        return pairs

    # Strategy 2: same-parent prismatic pairs
    parent_to_prismatic: dict[str, list] = defaultdict(list)
    for joint in root.iter("joint"):
        if joint.get("type") == "prismatic":
            p_el = joint.find("parent")
            if p_el is not None:
                parent_to_prismatic[p_el.get("link", "")].append(joint)

    for p_joints in parent_to_prismatic.values():
        if len(p_joints) < 2:
            continue
        primary = p_joints[0]
        p_ax_el = primary.find("axis")
        p_axis_z = float(
            (p_ax_el.get("xyz") if p_ax_el is not None else "0 0 1").split()[-1]
        )
        for companion in p_joints[1:]:
            c_ax_el = companion.find("axis")
            c_axis_z = float(
                (c_ax_el.get("xyz") if c_ax_el is not None else "0 0 1").split()[-1]
            )
            mult = (c_axis_z / p_axis_z) if p_axis_z != 0 else -1.0
            pairs.append((companion.get("name", ""), primary.get("name", ""), mult))

    return pairs


def approximate_gradient(
    func: callable,
    pos: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference gradient of a scalar 3D function."""
    grad = np.zeros(3)
    for i in range(3):
        p_plus = pos.copy(); p_plus[i] += eps
        p_minus = pos.copy(); p_minus[i] -= eps
        grad[i] = (func(p_plus) - func(p_minus)) / (2.0 * eps)
    return grad


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RobotKinematicsInfo:
    """
    Pinocchio-based kinematics backend.

    Handles single-arm and dual-arm URDFs transparently.  Mimic joints are
    auto-detected so that the public API works in **logical DOF** space
    (7 for single arm, 14 for dual arm), matching the real-robot action format.

    Parameters
    ----------
    urdf_path:
        Absolute path to the URDF.  Only the kinematic tree is parsed;
        mesh files and package:// URIs are not required.
    critical_link_names:
        Links whose world-frame Z position is checked against the safety
        constraint.  If None, all frames matching the patterns in
        _CRITICAL_PATTERNS ("link4", "link6", "gripper_base") are used.
    """

    def __init__(
        self,
        urdf_path: str,
        critical_link_names: list[str] | None = None,
        critical_link_indices: list[int] | None = None,
    ):
        self._urdf_path = str(urdf_path)
        self.model = pin.buildModelFromUrdf(self._urdf_path)
        self.data = self.model.createData()

        self.q_min = self.model.lowerPositionLimit   # full Pinocchio nq
        self.q_max = self.model.upperPositionLimit

        # ── Critical links ───────────────────────────────────────────────
        # Priority: indices > names > auto-detect
        if critical_link_indices is not None:
            import xml.etree.ElementTree as ET
            urdf_links = [
                el.get("name")
                for el in ET.parse(self._urdf_path).getroot().iter("link")
            ]
            critical_link_names = [urdf_links[i] for i in critical_link_indices]

        if critical_link_names is not None:
            self.critical_link_names = critical_link_names
        else:
            self.critical_link_names = self._auto_critical_links()

        self.critical_frame_ids: list[int] = []
        for name in self.critical_link_names:
            fid = self.model.getFrameId(name)
            if fid < len(self.model.frames):
                self.critical_frame_ids.append(fid)
            else:
                print(f"[SafetyFilter] WARNING: frame '{name}' not found, skipping.")

        # ── Mimic joints ─────────────────────────────────────────────────
        # mimic_pairs: [(mimic_q_idx, primary_q_idx, multiplier)]
        self._mimic_pairs: list[tuple[int, int, float]] = []
        raw = _detect_mimic_joints_from_urdf(self._urdf_path)
        for mimic_name, primary_name, mult in raw:
            try:
                mimic_jid   = self.model.getJointId(mimic_name)
                primary_jid = self.model.getJointId(primary_name)
                mq = int(self.model.joints[mimic_jid].idx_q)
                pq = int(self.model.joints[primary_jid].idx_q)
                self._mimic_pairs.append((mq, pq, mult))
            except Exception:
                pass

        mimic_idxs = {mq for mq, _, _ in self._mimic_pairs}
        self._independent_idxs: list[int] = [
            i for i in range(self.model.nq) if i not in mimic_idxs
        ]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def nq(self) -> int:
        """Full Pinocchio DOF (includes mimic joints)."""
        return self.model.nq

    @property
    def logical_dof(self) -> int:
        """
        Independent (logical) DOF — what the policy / real robot produces.

        Single arm : 7  (joint1-6 + gripper)
        Dual arm   : 14 (7 per arm)
        """
        return len(self._independent_idxs)

    # ── Logical ↔ Full conversion ─────────────────────────────────────────

    def expand(self, q_logical: np.ndarray) -> np.ndarray:
        """
        Logical DOF → full Pinocchio nq.

        Fills mimic joints using their primary-joint value and multiplier.
        """
        assert len(q_logical) == self.logical_dof, (
            f"Expected {self.logical_dof} logical DOF, got {len(q_logical)}"
        )
        q_full = np.zeros(self.nq)
        for logical_i, phys_i in enumerate(self._independent_idxs):
            q_full[phys_i] = q_logical[logical_i]
        for mq, pq, mult in self._mimic_pairs:
            q_full[mq] = mult * q_full[pq]
        return q_full

    def contract(self, q_full: np.ndarray) -> np.ndarray:
        """Full Pinocchio nq → logical DOF (drops mimic joints)."""
        return q_full[self._independent_idxs]

    # ── Kinematics ───────────────────────────────────────────────────────

    def compute_kinematics_and_jacobians(self, q_full: np.ndarray) -> None:
        """Forward kinematics + joint Jacobians in one pass (fast)."""
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_full)

    def get_frame_position(self, frame_id: int) -> np.ndarray:
        """World-frame 3D position (call after compute_kinematics_and_jacobians)."""
        return self.data.oMf[frame_id].translation.copy()

    def get_frame_jacobian_logical(self, frame_id: int) -> np.ndarray:
        """
        Translational Jacobian w.r.t. **logical** joints (shape: 3 × logical_dof).

        For the critical links (link4/6/gripper_base), gripper finger joints
        are kinematically downstream and do not affect these positions, so
        their Jacobian columns are zero — the logical Jacobian is identical
        to taking the independent columns of the full Jacobian.
        """
        J6 = pin.getFrameJacobian(
            self.model, self.data, frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_trans = J6[:3, :]                          # (3, nq)
        return J_trans[:, self._independent_idxs]    # (3, logical_dof)

    # ── Private ──────────────────────────────────────────────────────────

    def _auto_critical_links(self) -> list[str]:
        found: list[str] = []
        for frame in self.model.frames:
            name = frame.name
            # Skip joint frames (contain "joint" and end with a pattern) —
            # we only want body/link frames.
            if "joint" in name:
                continue
            for pat in _CRITICAL_PATTERNS:
                if name == pat or name.endswith("_" + pat):
                    found.append(name)
                    break
        return found


# ---------------------------------------------------------------------------
# Safety filter
# ---------------------------------------------------------------------------

def enforce_safe_action(
    q_logical: np.ndarray,
    urdf_info: RobotKinematicsInfo,
    safety_handles: "callable | list[callable]",
    solver: str = "osqp",
    constraint_margin: float = 0.05,
    regularization: float = 1e-4,
) -> np.ndarray:
    """
    Project q_logical onto the safe set defined by one or more safety handles.

    Accepts and returns **logical DOF** (7 for single arm, 14 for dual arm),
    matching the real-robot action format directly.

    The safe set is:  { q : h_k(p_i(q)) >= 0  for all critical links i,
                                                 for all handles k }

    If q_logical is already safe, it is returned unchanged with zero overhead.
    Otherwise a minimum-norm QP correction is computed in logical-DOF space:

        min   0.5 ‖Δq‖²  +  0.5·reg·‖Δq‖²    (Tikhonov regularization)
        s.t.  (∇h_k · J_i) · Δq  >=  -h_k_i   (active constraints only)
              q_min_L - q_L  <=  Δq  <=  q_max_L - q_L   (joint limits)

    Only constraints with  h_val < constraint_margin  are included in the QP.
    This avoids ill-conditioning from the many inactive box-face constraints
    that would otherwise dominate when the Jacobian is near-singular.

    Parameters
    ----------
    q_logical : (logical_dof,) array
        Nominal joint configuration from the policy.
    urdf_info : RobotKinematicsInfo
    safety_handles : callable or list of callables
        Each callable has signature  h(pos: np.ndarray) -> float.
        Positive means safe, negative means violation.
        A single callable is treated as a list of length 1.
    solver : str
        QP solver name for qpsolvers (default: "osqp").
    constraint_margin : float
        Only constraints with h < margin are added to the QP.
        Keeps the problem small and well-conditioned near singular configs.
    regularization : float
        Tikhonov regularization added to the diagonal of P (P = (1+reg)·I).
        Improves conditioning when the constraint Jacobian is near-singular.

    Returns
    -------
    q_safe : (logical_dof,) array — closest safe configuration.
    """
    handles: list[callable] = (
        safety_handles if isinstance(safety_handles, list) else [safety_handles]
    )

    # Joint limits in logical space
    q_min_l = urdf_info.q_min[urdf_info._independent_idxs]
    q_max_l = urdf_info.q_max[urdf_info._independent_idxs]

    # Pre-clamp input to joint limits.
    # If the policy outputs an out-of-range value, the QP bounds become
    # lb = q_min - q_logical < ub = q_max - q_logical < 0 for every
    # over-limit joint, which conflicts with any safety constraint that
    # pushes delta_q positive → primal infeasible.  Clamping first
    # guarantees lb <= 0 <= ub so the trivial delta_q=0 is always feasible.
    q_logical = np.clip(q_logical, q_min_l, q_max_l)

    q_full = urdf_info.expand(q_logical)
    urdf_info.compute_kinematics_and_jacobians(q_full)

    n = urdf_info.logical_dof
    G_rows: list[np.ndarray] = []
    h_rows: list[float] = []
    any_violation = False

    for frame_id in urdf_info.critical_frame_ids:
        pos = urdf_info.get_frame_position(frame_id)
        J_v = urdf_info.get_frame_jacobian_logical(frame_id)  # (3, n)

        for handle in handles:
            h_val = float(handle(pos))

            if h_val < 0:
                any_violation = True

            # Only include constraints that are active or nearly active.
            # Inactive constraints (h >> 0) add redundant rows that worsen
            # conditioning when the Jacobian is near-singular.
            if h_val >= constraint_margin:
                continue

            grad_h = approximate_gradient(handle, pos)  # (3,)

            # Constraint: (grad_h @ J_v) @ δq >= -h_val
            # qpsolvers G·x <= h form → negate
            G_rows.append(-(grad_h @ J_v))
            h_rows.append(h_val)

    if not any_violation:
        return q_logical   # zero-overhead pass-through

    P    = sp.eye(n, format="csc") * (1.0 + regularization)
    q_qp = np.zeros(n)
    G    = sp.csc_matrix(np.array(G_rows))
    h    = np.array(h_rows)
    lb   = q_min_l - q_logical
    ub   = q_max_l - q_logical

    delta_q = solve_qp(P, q_qp, G, h, lb=lb, ub=ub, solver=solver)

    if delta_q is not None:
        # Post-clip: guard against OSQP numerical tolerance (~1e-3 rad)
        # slightly violating the box bounds.
        return np.clip(q_logical + delta_q, q_min_l, q_max_l)

    raise RuntimeError(
        "[SafetyFilter] QP infeasible — cannot find a safe action. "
        "Robot execution stopped to prevent unsafe motion."
    )


# ---------------------------------------------------------------------------
# Box constraint config + helpers
# ---------------------------------------------------------------------------

@dataclass
class BoxConstraintConfig:
    """
    Axis-aligned box workspace constraint.  All values in metres (world frame).

    Fields
    ------
    x_min, x_max, y_min, y_max, z_min, z_max:
        Box boundaries in the world frame.
    mode : "inclusion" | "exclusion"
        "inclusion" (default) — links must stay INSIDE the box.
            Uses 6 independent handles, one per face.
        "exclusion" — links must stay OUTSIDE the box (forbidden zone).
            Uses a single handle: h(pos) = -min(face distances).
            A link is safe when it is outside on at least one axis.
    critical_links:
        Indices into the URDF link list whose positions are checked.
        Empty list (default) → auto-detect from patterns in _CRITICAL_PATTERNS.
    """
    x_min: float = -0.8
    x_max: float =  0.8
    y_min: float = -0.8
    y_max: float =  0.8
    z_min: float =  0.18
    z_max: float =  1.2
    mode: str = "inclusion"
    critical_links: list[int] = field(default_factory=list)


def load_constraint_config(config_path: str) -> BoxConstraintConfig:
    """
    Load a BoxConstraintConfig from a JSON file using draccus.

    The JSON file must contain exactly the six fields defined in
    BoxConstraintConfig (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    import io
    import draccus
    from draccus import config_type

    with open(config_path) as f:
        raw = f.read()

    with config_type("json"):
        return draccus.load(BoxConstraintConfig, io.StringIO(raw))


def make_box_handles(cfg: BoxConstraintConfig) -> list:
    """
    Build safety handles from a BoxConstraintConfig.

    inclusion mode (default)
        Returns 6 handles, one per face.  h(pos) >= 0 iff the link is on
        the safe (inside) side of that face:
            h_x_min = pos[0] - x_min,  h_x_max = x_max - pos[0],  etc.

    exclusion mode
        Returns 1 handle.  h(pos) >= 0 iff the link is OUTSIDE the box
        on at least one axis:
            h(pos) = -min(pos[0]-x_min, x_max-pos[0],
                          pos[1]-y_min, y_max-pos[1],
                          pos[2]-z_min, z_max-pos[2])
        When inside the box every face distance is positive, so min > 0
        and h < 0 (violation).  The finite-difference gradient of this
        handle automatically points toward the nearest face, giving the
        minimum-displacement escape direction.

    Raises
    ------
    ValueError if cfg.mode is not "inclusion" or "exclusion".
    """
    x_min, x_max = cfg.x_min, cfg.x_max
    y_min, y_max = cfg.y_min, cfg.y_max
    z_min, z_max = cfg.z_min, cfg.z_max

    if cfg.mode == "inclusion":
        def h_x_min(pos): return float(pos[0]) - x_min
        def h_x_max(pos): return x_max - float(pos[0])
        def h_y_min(pos): return float(pos[1]) - y_min
        def h_y_max(pos): return y_max - float(pos[1])
        def h_z_min(pos): return float(pos[2]) - z_min
        def h_z_max(pos): return z_max - float(pos[2])
        return [h_x_min, h_x_max, h_y_min, h_y_max, h_z_min, h_z_max]

    if cfg.mode == "exclusion":
        def h_exclusion(pos):
            # Positive = outside box (safe), negative = inside box (violation).
            face_dists = [
                float(pos[0]) - x_min,
                x_max - float(pos[0]),
                float(pos[1]) - y_min,
                y_max - float(pos[1]),
                float(pos[2]) - z_min,
                z_max - float(pos[2]),
            ]
            return -min(face_dists)
        return [h_exclusion]

    raise ValueError(f"BoxConstraintConfig.mode must be 'inclusion' or 'exclusion', got {cfg.mode!r}")
