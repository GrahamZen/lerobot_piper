#!/usr/bin/env python3
"""
Generate piper_dual_description.urdf by merging two piper_description URDFs.

Layout (top-down view, both arms face +x):

        +x (forward)
        ^
        |
  left  |  right
  arm   |  arm
 (L_y) | (-L_y)
        |
        +----> +y

Left arm:  y = +ARM_SPACING / 2
Right arm: y = -ARM_SPACING / 2

Joint order in the merged model (14 DOF):
  left_joint1 .. left_joint6  left_joint7  left_joint8
  right_joint1 .. right_joint6 right_joint7 right_joint8

Usage:
    python assets/piper_dual_description/generate_dual_urdf.py
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ARM_SPACING = 0.5          # metres between the two base_link origins (y axis)
SOURCE_URDF = Path(__file__).parent.parent / "piper_urdf/piper_description/urdf/piper_description.urdf"
MESHES_ABS  = (Path(__file__).parent.parent / "piper_urdf/piper_description/meshes").resolve()
OUT_URDF    = Path(__file__).parent / "urdf/piper_dual_description.urdf"
# ────────────────────────────────────────────────────────────────────────────


def resolve_mesh_path(filename: str) -> str:
    """Convert package://piper_description/meshes/foo.STL → absolute path."""
    if filename.startswith("package://"):
        bare = filename.split("/", 3)[-1]                # meshes/foo.STL
        if bare.startswith("meshes/"):
            bare = bare[len("meshes/"):]                 # foo.STL
        return str(MESHES_ABS / bare)
    return filename


def _gripper_companions(source_root: ET.Element) -> dict[str, str]:
    """
    Find pairs of prismatic joints that share the same parent link
    (i.e. symmetric gripper fingers).  Returns {companion_name: primary_name}.
    The first prismatic joint on each parent is the "primary"; any additional
    ones are "companions" that should mimic the primary.
    """
    from collections import defaultdict
    parent_to_joints: dict[str, list[ET.Element]] = defaultdict(list)
    for j in source_root.findall("joint"):
        if j.get("type") == "prismatic":
            p = j.find("parent").get("link")
            parent_to_joints[p].append(j)

    companions = {}
    for joints in parent_to_joints.values():
        if len(joints) < 2:
            continue
        primary = joints[0].get("name")
        for companion in joints[1:]:
            companions[companion.get("name")] = primary
    return companions


def make_arm(source_root: ET.Element, prefix: str, y_offset: float) -> tuple:
    """
    Return (links, joints) for one arm with all names prefixed and
    mesh paths resolved to absolute filesystem paths.

    The arm's base_link is connected to 'world' via a fixed joint at
    (0, y_offset, 0).

    Companion gripper joints (symmetric fingers) get a <mimic> tag so
    they track the primary gripper joint — reducing effective DOF from
    8 to 7 per arm.
    """
    links  = []
    joints = []

    companions = _gripper_companions(source_root)  # {companion: primary}

    # ── Links ──
    for link in source_root.findall("link"):
        l = deepcopy(link)
        l.set("name", f"{prefix}{l.get('name')}")
        for mesh in l.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn:
                mesh.set("filename", resolve_mesh_path(fn))
        links.append(l)

    # ── Fixed joint: world → <prefix>base_link ──
    attach = ET.Element("joint", name=f"{prefix}attach", type="fixed")
    ET.SubElement(attach, "parent", link="world")
    ET.SubElement(attach, "child",  link=f"{prefix}base_link")
    ET.SubElement(attach, "origin", xyz=f"0 {y_offset:.4f} 0", rpy="0 0 0")
    joints.append(attach)

    # ── Existing joints (prefixed, companion joints get <mimic>) ──
    for joint in source_root.findall("joint"):
        j = deepcopy(joint)
        original_name = j.get("name")
        j.set("name", f"{prefix}{original_name}")

        parent_el = j.find("parent")
        child_el  = j.find("child")
        parent_el.set("link", f"{prefix}{parent_el.get('link')}")
        child_el.set( "link", f"{prefix}{child_el.get('link')}")

        if original_name in companions:
            primary = companions[original_name]
            # Determine multiplier from axis sign comparison
            companion_axis = j.find("axis")
            primary_el     = next(jj for jj in source_root.findall("joint") if jj.get("name") == primary)
            primary_axis   = primary_el.find("axis")
            c_z = float((companion_axis.get("xyz") if companion_axis is not None else "0 0 1").split()[-1])
            p_z = float((primary_axis.get("xyz")   if primary_axis   is not None else "0 0 1").split()[-1])
            multiplier = c_z / p_z if p_z != 0 else -1.0
            ET.SubElement(j, "mimic", joint=f"{prefix}{primary}", multiplier=str(multiplier), offset="0")

        joints.append(j)

    return links, joints


def build_dual_urdf() -> ET.Element:
    tree = ET.parse(str(SOURCE_URDF))
    src  = tree.getroot()

    robot = ET.Element("robot", name="piper_dual")

    # World link (massless root — no inertial element needed)
    ET.SubElement(robot, "link", name="world")

    left_links,  left_joints  = make_arm(src, "left_",  +ARM_SPACING / 2)
    right_links, right_joints = make_arm(src, "right_", -ARM_SPACING / 2)

    for el in left_links + left_joints + right_links + right_joints:
        robot.append(el)

    return robot


def indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty-print indentation (stdlib ET has no built-in indent before 3.9)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():   # noqa: F821
            child.tail = pad
    if not elem.tail or not elem.tail.strip():
        elem.tail = pad


def main():
    robot = build_dual_urdf()
    indent(robot)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")   # Python ≥ 3.9; falls back gracefully
    OUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(OUT_URDF), xml_declaration=True, encoding="unicode")
    print(f"Written: {OUT_URDF}")

    # Quick sanity check
    joints = [j for j in robot.findall("joint") if j.get("type") != "fixed"]
    print(f"  Actuated joints: {len(joints)}")
    for j in joints:
        print(f"    {j.get('name')} ({j.get('type')})")


if __name__ == "__main__":
    main()
