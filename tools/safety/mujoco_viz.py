#!/usr/bin/env python3
"""
MuJoCo real-time visualization for Piper robot models.

Supports any MJCF XML or URDF under assets/piper_urdf/, including:
  - piper_description     (8 DOF: 6 arm + 2 gripper fingers)
  - piper_h_description   (6 DOF)
  - piper_l_description   (6 DOF)
  - dual-arm merged URDF  (14 DOF, user-generated)

URDF files using package:// mesh references are resolved automatically.

--- Programmatic API ---

    from tools.safety.mujoco_viz import PiperMuJoCoViz

    viz = PiperMuJoCoViz("assets/piper_urdf/piper_description/mujoco_model/piper_description.xml")
    viz.start()
    viz.set_qpos([0.1, 0.2, -0.5, 0.0, 0.0, 0.0, 0.02, -0.02])
    time.sleep(5)
    viz.close()

--- Standalone stdin mode ---

    uv run python tools/safety/mujoco_viz.py \\
        --model assets/piper_urdf/piper_description/mujoco_model/piper_description.xml

    # then send JSON arrays line by line:
    [0.1, 0.2, -0.5, 0, 0, 0, 0.02, -0.02]

--- Socket mode ---

    uv run python tools/safety/mujoco_viz.py \\
        --model assets/piper_urdf/piper_description/mujoco_model/piper_description.xml \\
        --port 9999

    # send JSON arrays over TCP (newline-delimited):
    echo '[0.1, 0.2, -0.5, 0, 0, 0]' | nc localhost 9999
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path


def _resolve_model(model_path: str | Path) -> str:
    """
    Return a path to a loadable MJCF/XML file.

    - .xml files are returned as-is (assumed to be valid MJCF).
    - .urdf files are preprocessed:
        * package:// mesh URIs → basename only (temp file placed in meshes dir)
        * Links with non-positive-definite inertia tensors have their inertial
          block removed (MuJoCo will then compute inertia from geometry).
        * A <mujoco><compiler inertiafromgeom="auto"/></mujoco> element is
          injected so remaining geometry-only links still get inertia.
    """
    import xml.etree.ElementTree as ET

    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() == ".xml":
        return str(path)

    if path.suffix.lower() != ".urdf":
        raise ValueError(f"Unsupported model format: {path.suffix}. Use .xml (MJCF) or .urdf.")

    # Infer meshes directory: <pkg_root>/meshes/  (standard ROS layout)
    pkg_root = path.parent.parent  # e.g. piper_urdf/piper_description
    meshes_dir = pkg_root / "meshes"
    if not meshes_dir.is_dir():
        # Fallback: place next to URDF
        meshes_dir = path.parent

    tree = ET.parse(str(path))
    root = tree.getroot()

    # 1. Replace package:// mesh URIs with just the file basename.
    #
    #    MuJoCo's URDF parser strips directory parts and resolves bare
    #    filenames relative to the temp URDF's directory.  Strategy:
    #      a) For each mesh URI, search meshes_dir tree for a matching file
    #         (case-insensitive, any subdir).
    #      b) Prefer STL/OBJ files (MuJoCo-supported); skip DAE (Collada).
    #      c) Write the temp URDF into the directory of the first supported
    #         mesh found.  Update the filename to the actual on-disk name.
    #      d) If only DAE meshes exist, warn and strip mesh refs so MuJoCo
    #         still loads the kinematics.

    # Build a case-insensitive lookup: lower(basename) -> first Path found.
    # Seed with package-based meshes_dir tree, then also with any absolute
    # paths already present in the URDF (for generated/merged URDFs).
    _mesh_cache: dict[str, Path] = {}

    def _add_to_cache(p: Path) -> None:
        if p.is_file():
            key = p.name.lower()
            if key not in _mesh_cache:
                _mesh_cache[key] = p

    for p in sorted(meshes_dir.rglob("*")):
        _add_to_cache(p)

    # Also seed from absolute paths already in the URDF
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn and not fn.startswith("package://") and Path(fn).is_absolute():
            _add_to_cache(Path(fn))

    # Determine the directory to write the temp URDF into: the parent of the
    # first supported (STL/OBJ) mesh we can locate.
    mesh_subdir = meshes_dir
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        basename = fn.rsplit("/", 1)[-1]
        found = _mesh_cache.get(basename.lower())
        if found is not None and found.suffix.lower() in (".stl", ".obj"):
            mesh_subdir = found.parent
            break

    # Collect all mesh elements that need rewriting (package:// or absolute path)
    def _needs_rewrite(fn: str) -> bool:
        return fn.startswith("package://") or (
            Path(fn).is_absolute() if fn else False
        )

    all_meshes = [m for m in root.iter("mesh") if _needs_rewrite(m.get("filename", ""))]

    # Classify as supported (STL/OBJ) or unsupported (DAE / not found)
    def _is_supported(mesh_el) -> bool:
        basename = mesh_el.get("filename", "").rsplit("/", 1)[-1]
        found = _mesh_cache.get(basename.lower())
        return found is not None and found.suffix.lower() in (".stl", ".obj")

    has_unsupported = any(not _is_supported(m) for m in all_meshes)
    if has_unsupported and all_meshes:
        print(
            f"[PiperMuJoCoViz] WARNING: {path.name} contains Collada (.dae) meshes "
            "which MuJoCo cannot decode. Affected visual/collision geometry will be stripped.",
            flush=True,
        )

    # Replace supported mesh refs with bare filenames (MuJoCo strips dirs).
    # Remove <visual>/<collision> blocks whose mesh is unsupported.
    for mesh in all_meshes:
        basename = mesh.get("filename", "").rsplit("/", 1)[-1]
        found = _mesh_cache.get(basename.lower())
        if found is not None and found.suffix.lower() in (".stl", ".obj"):
            mesh.set("filename", found.name)

    if has_unsupported:
        for link in root.iter("link"):
            for tag in ("visual", "collision"):
                for child in link.findall(tag):
                    geom = child.find("geometry")
                    if geom is None:
                        continue
                    m = geom.find("mesh")
                    if m is not None and not _is_supported(m):
                        link.remove(child)

    # 2. Remove inertial blocks with non-positive-definite tensors.
    #    A symmetric 3×3 matrix is positive definite iff all principal minors
    #    are positive.  A quick necessary condition: all diagonal entries > 0.
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia")
        if inertia is None:
            continue
        ixx = float(inertia.get("ixx", "1"))
        iyy = float(inertia.get("iyy", "1"))
        izz = float(inertia.get("izz", "1"))
        if ixx <= 0 or iyy <= 0 or izz <= 0:
            link.remove(inertial)

    # 3. Inject MuJoCo compiler hint so geometry-only links get inertia.
    compiler_el = ET.fromstring('<mujoco><compiler inertiafromgeom="auto"/></mujoco>')
    root.insert(0, compiler_el)

    # Write temp URDF into the directory that contains the mesh files so that
    # MuJoCo can find them by bare filename (MuJoCo URDF loader strips dirs).
    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="wb", dir=str(mesh_subdir)
    )
    tree.write(tmp.name, xml_declaration=True, encoding="utf-8")
    tmp.flush()
    tmp.close()
    return tmp.name


def _add_env_defaults(spec) -> None:
    """
    Add a ground plane, skybox, directional lights, and headlight to a MjSpec.
    Called after loading the robot model so the environment is always present.
    """
    import mujoco

    # Skybox (gradient blue → black)
    sky = spec.add_texture()
    sky.name = "__env_skybox__"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.40, 0.58, 0.78]
    sky.rgb2 = [0.02, 0.02, 0.04]
    sky.width = 512
    sky.height = 3072

    # Checker texture + material for floor
    grid_tex = spec.add_texture()
    grid_tex.name = "__env_grid__"
    grid_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid_tex.rgb1 = [0.15, 0.20, 0.25]
    grid_tex.rgb2 = [0.25, 0.30, 0.35]
    grid_tex.width = 300
    grid_tex.height = 300
    grid_tex.mark = mujoco.mjtMark.mjMARK_EDGE
    grid_tex.markrgb = [0.35, 0.42, 0.50]

    grid_mat = spec.add_material()
    grid_mat.name = "__env_grid_mat__"
    grid_mat.textures[0] = "__env_grid__"
    grid_mat.texuniform = True
    grid_mat.texrepeat = [5, 5]
    grid_mat.reflectance = 0.15

    # Ground plane
    floor = spec.worldbody.add_geom()
    floor.name = "__env_floor__"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]   # infinite plane
    floor.material = "__env_grid_mat__"
    floor.rgba = [1, 1, 1, 1]

    # Key light (casts shadows)
    key = spec.worldbody.add_light()
    key.name = "__env_key__"
    key.pos = [0, -1.5, 3.0]
    key.dir = [0, 0.4, -1]
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.diffuse = [0.80, 0.80, 0.78]
    key.specular = [0.20, 0.20, 0.18]
    key.castshadow = True

    # Fill light (no shadow, softer)
    fill = spec.worldbody.add_light()
    fill.name = "__env_fill__"
    fill.pos = [2, 2, 2]
    fill.dir = [-1, -1, -1]
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.diffuse = [0.35, 0.35, 0.38]
    fill.specular = [0, 0, 0]
    fill.castshadow = False

    # Headlight (follows camera)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.70, 0.70, 0.70]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]



def _detect_companion_joints(model_path: str | Path) -> list[tuple[str, str, float]]:
    """
    Parse a URDF and return a list of (companion_name, primary_name, multiplier)
    for prismatic joints that share the same parent link (symmetric gripper fingers).

    For .xml (MJCF) files or any file without companion joints returns [].
    MuJoCo 3.x does not parse URDF <mimic> tags, so we detect coupling here
    and enforce it in PiperMuJoCoViz.set_qpos() instead.
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    path = Path(model_path)
    if path.suffix.lower() != ".urdf":
        return []

    try:
        root = ET.parse(str(path)).getroot()
    except Exception:
        return []

    parent_to_prismatic: dict[str, list[ET.Element]] = defaultdict(list)
    for joint in root.iter("joint"):
        # Handle both flat URDF (direct children) and already-merged URDFs
        # where joints may reference prefixed link names.
        if joint.get("type") == "prismatic":
            p_el = joint.find("parent")
            if p_el is not None:
                parent_to_prismatic[p_el.get("link", "")].append(joint)

    companions = []
    for p_joints in parent_to_prismatic.values():
        if len(p_joints) < 2:
            continue
        primary = p_joints[0]
        p_axis_el = primary.find("axis")
        p_axis_z = float((p_axis_el.get("xyz") if p_axis_el is not None else "0 0 1").split()[-1])
        for companion in p_joints[1:]:
            c_axis_el = companion.find("axis")
            c_axis_z = float((c_axis_el.get("xyz") if c_axis_el is not None else "0 0 1").split()[-1])
            mult = (c_axis_z / p_axis_z) if p_axis_z != 0 else -1.0
            companions.append((companion.get("name"), primary.get("name"), mult))

    return companions


class PiperMuJoCoViz:
    """
    Real-time MuJoCo visualizer for Piper (and compatible) robot models.

    Parameters
    ----------
    model_path:
        Path to an MJCF .xml file or a URDF .urdf file.
        URDF files with package:// mesh URIs are rewritten automatically.
    control_mode:
        "qpos"  – directly set qpos (joint positions), bypassing actuators.
                  Use this when you want instantaneous pose without dynamics.
        "ctrl"  – set ctrl (actuator setpoints); the built-in PD controllers
                  drive the arm to the target. More physically realistic.
    sim_rate:
        How many mujoco simulation steps to run per viewer sync. Increase for
        smoother motion when using ctrl mode.
    """

    def __init__(
        self,
        model_path: str | Path,
        control_mode: str = "qpos",
        sim_rate: int = 5,
        constraint_box: dict | None = None,
    ):
        import mujoco

        resolved = _resolve_model(model_path)
        try:
            spec = mujoco.MjSpec.from_file(resolved)
            # MjSpec loaded from URDF silently ignores textures/materials
            # added after load.  Roundtrip through to_xml() / from_string()
            # converts the spec to pure MJCF so _add_env_defaults works.
            # We must preserve meshdir so mesh file references still resolve.
            if Path(resolved).suffix.lower() == ".urdf":
                meshes_dir = str(Path(resolved).parent)
                spec = mujoco.MjSpec.from_string(spec.to_xml())
                spec.meshdir = meshes_dir
                spec.modelfiledir = meshes_dir
        finally:
            if resolved != str(Path(model_path).resolve()):
                Path(resolved).unlink(missing_ok=True)

        _add_env_defaults(spec)

        # Optional: semi-transparent box marking the 3-D safety constraint boundary.
        # constraint_box is a dict with keys x_min, x_max, y_min, y_max, z_min, z_max.
        _constraint_geom_name = "__constraint_box__"
        if constraint_box is not None:
            x_min = float(constraint_box["x_min"])
            x_max = float(constraint_box["x_max"])
            y_min = float(constraint_box["y_min"])
            y_max = float(constraint_box["y_max"])
            z_min = float(constraint_box["z_min"])
            z_max = float(constraint_box["z_max"])

            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            cz = (z_min + z_max) / 2.0
            hx = (x_max - x_min) / 2.0
            hy = (y_max - y_min) / 2.0
            hz = (z_max - z_min) / 2.0

            cbox = spec.worldbody.add_geom()
            cbox.name = _constraint_geom_name
            cbox.type = mujoco.mjtGeom.mjGEOM_BOX
            cbox.size = [hx, hy, hz]
            cbox.pos  = [cx, cy, cz]
            cbox.rgba = [1.0, 0.15, 0.05, 0.12]   # translucent red-orange
            cbox.contype  = 0   # no collision
            cbox.conaffinity = 0

        self._model = spec.compile()

        # Robot geoms have no material and keep their original URDF colors
        # (light grey/blue).  Override all material-less geoms to near-black,
        # but preserve the constraint box colour.
        import numpy as np
        constraint_geom_id = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, _constraint_geom_name)
            if constraint_box is not None else -1
        )
        for i in range(self._model.ngeom):
            if self._model.geom_matid[i] == -1 and i != constraint_geom_id:
                self._model.geom_rgba[i] = [0.08, 0.08, 0.10, 1.0]

        self._data = mujoco.MjData(self._model)

        if control_mode not in ("qpos", "ctrl"):
            raise ValueError("control_mode must be 'qpos' or 'ctrl'")
        self._control_mode = control_mode
        self._sim_rate = sim_rate

        self._dof = self._model.nq  # physical DOFs (includes mimic joints)
        self._nu = self._model.nu   # number of actuators

        # ── Detect companion (mimic) joints from the original URDF ─────
        # MuJoCo 3.x does not parse URDF <mimic> tags, so we detect
        # symmetric gripper finger pairs from the URDF ourselves and
        # enforce the coupling in set_qpos().
        #
        # _mimic_map: list of (companion_qpos_idx, primary_qpos_idx, multiplier)
        raw_companions = _detect_companion_joints(model_path)  # [(comp_name, prim_name, mult)]

        def _qaddr(joint_name: str) -> int:
            return int(self._model.jnt_qposadr[self._model.joint(joint_name).id])

        self._mimic_map: list[tuple[int, int, float]] = []
        for comp_name, prim_name, mult in raw_companions:
            try:
                self._mimic_map.append((_qaddr(comp_name), _qaddr(prim_name), mult))
            except Exception:
                pass  # joint name not in model (shouldn't happen)

        mimic_qidxs = {entry[0] for entry in self._mimic_map}
        self._independent_indices = [i for i in range(self._dof) if i not in mimic_qidxs]
        self._logical_dof = len(self._independent_indices)

        joint_names = [self._model.joint(i).name for i in range(self._model.njnt)]
        actuator_names = [self._model.actuator(i).name for i in range(self._nu)]
        print(f"[PiperMuJoCoViz] Loaded: {Path(model_path).name}")
        print(f"  DOF (nq): {self._dof}   Logical DOF: {self._logical_dof}   Actuators (nu): {self._nu}")
        print(f"  Joints:    {joint_names}")
        if self._mimic_map:
            mimic_names = [c for c, _, _ in raw_companions]
            print(f"  Coupled gripper joints (auto-mirrored): {mimic_names}")
        print(f"  Actuators: {actuator_names}")
        print(f"  Control mode: {control_mode}")

        self._lock = threading.Lock()
        self._pending_qpos: list[float] | None = None
        self._running = False
        self._viewer_thread: threading.Thread | None = None
        self._key_callback: callable | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dof(self) -> int:
        """
        Logical (independent) degrees of freedom.

        This is the length of the array expected by set_qpos().
        Mimic/coupled joints (e.g. symmetric gripper fingers) are excluded —
        they are set automatically when you provide the primary joint value.

        For a single piper_description arm: 7 (joint1-6 + gripper).
        For piper_dual_description:        14 (7 per arm).
        """
        return self._logical_dof

    @property
    def dof_physical(self) -> int:
        """Total number of qpos entries in the MuJoCo model (includes mimic joints)."""
        return self._dof

    def start(self, key_callback: callable | None = None) -> None:
        """
        Launch the viewer in a background thread (non-blocking).

        Parameters
        ----------
        key_callback:
            Optional callable(keycode: int) invoked on every key press inside
            the MuJoCo viewer window.  Runs on the viewer thread — use
            threading primitives (e.g. threading.Event) to communicate with
            the main thread.
        """
        if self._running:
            return
        self._key_callback = key_callback
        self._running = True
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    def set_qpos(self, qpos: list[float] | None) -> None:
        """
        Send a new joint-position command.

        Parameters
        ----------
        qpos:
            Joint positions in radians (revolute) or meters (prismatic).
            Length must equal self.dof (logical DOF — mimic joints excluded).
            Pass None to stop updating.
        """
        if qpos is None:
            with self._lock:
                self._pending_qpos = None
            return

        if len(qpos) != self._logical_dof:
            raise ValueError(
                f"Expected {self._logical_dof} DOF (logical), got {len(qpos)}."
            )

        # Expand logical → physical qpos, filling mimic joints automatically
        full = [0.0] * self._dof
        for logical_i, phys_i in enumerate(self._independent_indices):
            full[phys_i] = qpos[logical_i]
        for (mimic_idx, primary_idx, mult) in self._mimic_map:
            full[mimic_idx] = mult * full[primary_idx]

        with self._lock:
            self._pending_qpos = full

    def update_constraint_box(self, box: dict) -> None:
        """
        Update the constraint box geom geometry in-place (no model recompile).

        The viewer camera and all other settings are left unchanged.

        Parameters
        ----------
        box : dict with keys x_min, x_max, y_min, y_max, z_min, z_max
        """
        import mujoco
        gid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "__constraint_box__"
        )
        if gid == -1:
            return  # no constraint box was created at init time

        x_min, x_max = float(box["x_min"]), float(box["x_max"])
        y_min, y_max = float(box["y_min"]), float(box["y_max"])
        z_min, z_max = float(box["z_min"]), float(box["z_max"])

        with self._lock:
            self._model.geom_size[gid] = [
                (x_max - x_min) / 2.0,
                (y_max - y_min) / 2.0,
                (z_max - z_min) / 2.0,
            ]
            self._model.geom_pos[gid] = [
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                (z_min + z_max) / 2.0,
            ]

    def close(self) -> None:
        """Stop the viewer."""
        self._running = False
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=3.0)

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _viewer_loop(self) -> None:
        import mujoco
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            self._model, self._data,
            key_callback=self._key_callback,
        ) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.2

            while self._running and viewer.is_running():
                with self._lock:
                    pending = self._pending_qpos

                if pending is not None:
                    if self._control_mode == "qpos":
                        self._data.qpos[:] = pending
                        mujoco.mj_forward(self._model, self._data)
                    else:  # ctrl mode
                        self._data.ctrl[: self._nu] = pending[: self._nu]
                        for _ in range(self._sim_rate):
                            mujoco.mj_step(self._model, self._data)

                viewer.sync()
                time.sleep(0.02)

        self._running = False


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _stdin_loop(viz: PiperMuJoCoViz) -> None:
    """Read JSON arrays from stdin and forward to viz."""
    print(f"[stdin] Send {viz.dof}-element JSON arrays (e.g. [0,0,0,0,0,0]).", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            qpos = json.loads(line)
            viz.set_qpos(qpos)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[stdin] Error: {e}", flush=True)


def _socket_loop(viz: PiperMuJoCoViz, port: int) -> None:
    """Listen on TCP port and forward newline-delimited JSON arrays to viz."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)
    print(f"[socket] Listening on port {port}. Send {viz.dof}-element JSON arrays.", flush=True)
    while viz.is_running():
        try:
            srv.settimeout(1.0)
            conn, addr = srv.accept()
        except TimeoutError:
            continue
        print(f"[socket] Connection from {addr}")
        buf = ""
        with conn:
            while viz.is_running():
                try:
                    chunk = conn.recv(4096).decode()
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        qpos = json.loads(line)
                        viz.set_qpos(qpos)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"[socket] Error: {e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo real-time visualizer for Piper robots.")
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Path to the model file. Accepts MJCF .xml or URDF .urdf. "
            "Example: assets/piper_urdf/piper_description/mujoco_model/piper_description.xml"
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=["qpos", "ctrl"],
        default="qpos",
        help="'qpos': set joint positions directly. 'ctrl': use PD actuators (default: qpos).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="If set, listen on this TCP port instead of stdin.",
    )
    args = parser.parse_args()

    viz = PiperMuJoCoViz(args.model, control_mode=args.control_mode)
    viz.start()

    try:
        if args.port is not None:
            _socket_loop(viz, args.port)
        else:
            _stdin_loop(viz)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()


if __name__ == "__main__":
    main()
