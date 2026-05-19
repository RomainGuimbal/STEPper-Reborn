import bpy
import bmesh
import numpy as np
from .trimesh import TriMesh
from .utils import bpy_update_object_data, create_new_obj_with_mesh


def build_mesh(step_reader, obj, shp, lind, angd, vcol_name="Colors"):
    hacks = set([])
    if bpy.context.scene.stepper.hack_skip_zero_solids:
        hacks.add("skip_solids")

    mesh: TriMesh = step_reader.build_trimesh(
        shp, lin_def=lind, ang_def=angd, hacks=hacks
    )

    mesh.fuse_verts()
    mesh.filter_zero_area()
    mesh.filter_same_face()

    print(f"[bm] {len(mesh.verts)}", end="")

    # --- Build geometry at C level (replaces N individual bm.verts.new /
    #     bm.faces.new Python calls in add_to_bm) ---
    objdata = obj.data
    objdata.from_pydata(mesh.verts, [], [t.indices for t in mesh.tris])

    # Load into BMesh for edge marking and color / material assignment.
    # from_mesh() is a single C-level bulk operation, much faster than
    # constructing the BMesh vertex by vertex.
    bm = bmesh.new()
    bm.from_mesh(objdata)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Pre-extract batch per face to avoid repeated Python attribute access in loop
    batches = [t.batch for t in mesh.tris]

    # Seam edges: mark boundaries between different shape batches
    for e in bm.edges:
        f = e.link_faces
        if len(f) == 2 and batches[f[0].index] != batches[f[1].index]:
            e.seam = True

    # Sharp edges: mark where per-vertex normals are discontinuous.
    # Pre-build face_vert_norms[face_idx][global_vert_idx] = norm to replace
    # the inner range(3) scan with O(1) dict lookups.
    face_vert_norms = [{t.indices[j]: t.norms[j] for j in range(3)} for t in mesh.tris]

    def _project_plane_normalize(plane, vec):
        prj = vec - (plane * np.dot(plane, vec))
        prjn = np.linalg.norm(prj)
        return prj / prjn if prjn != 0.0 else np.array([0.0, 0.0, 1.0])

    def _prjtest(plane, n0, n1, margin):
        p0 = _project_plane_normalize(plane, n0)
        prj = _project_plane_normalize(plane, n1)
        return np.dot(p0, prj) < 1.0 - margin

    margin = 0.02
    for e in bm.edges:
        fi = [f.index for f in e.link_faces]
        if len(fi) != 2:
            continue
        fvn0 = face_vert_norms[fi[0]]
        fvn1 = face_vert_norms[fi[1]]
        ev0 = e.verts[0].index
        ev1 = e.verts[1].index
        n00 = fvn0.get(ev0)
        n10 = fvn1.get(ev0)
        n01 = fvn0.get(ev1)
        n11 = fvn1.get(ev1)
        if n00 is None or n10 is None or n01 is None or n11 is None:
            e.smooth = True
            continue
        plane = np.array((e.verts[0].co - e.verts[1].co).normalized())
        if _prjtest(plane, n00, n10, margin) and _prjtest(plane, n01, n11, margin):
            e.smooth = False
        else:
            e.smooth = True

    mesh.fill_empty_color()
    bpy_update_object_data(
        objdata,
        bm,
        vcol_name,
        mesh.get_loop_colors(),
        mesh.get_loop_uvs(),
        mesh.get_loop_normals(),
        mesh.get_loop_material_names(),
        build_materials=bpy.context.scene.stepper.build_materials,
    )

    return mesh.matrix


def mesh_from_shape(
    step_reader,
    shp,
    tree,
    filename,
    filepath,
    hierarchy_empties,
    node_index,
    created_names,
    lin_deflection,
    ang_deflection,
    created_uuid,
    total,
    i,
):
    parent_uuid, self_uuid, tag, name, _, local_t, global_t = tree.nodes[
        node_index
    ].get_values()

    if name == "root":
        name = filename + ".empties"

    shape_name = "tt_" + repr(tag)

    obj = None

    # Shape found in leaf
    if shp:
        print("\nBuilding ({}/{}): {} ".format(i + 1, total, name), end="", flush=True)
        print("[T" + repr(shp.ShapeType()) + "]", end="", flush=True)

        # If object already build, just copy it, using linked mesh data
        if shape_name in created_names:
            print("[Link]", end="", flush=True)
            source_obj = created_names[shape_name]
            obj = source_obj.copy()

        else:  # Create new mesh and object from scratch
            print("[Build]", end="", flush=True)
            obj = create_new_obj_with_mesh(name)
            bpy.ops.object.mode_set(mode="OBJECT")
            build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)
            created_names[shape_name] = obj

    # No shape in leaf, empty creation enabled, do this
    elif hierarchy_empties:
        # Create empty
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_size = 2
        obj.empty_display_type = "PLAIN_AXES"
        # set_obj_matrix_world(obj, global_t)

    # Object has been created
    if obj:
        # assign custom properties to blender objects
        obj["STEP_tag"] = tag
        obj["STEP_parent"] = parent_uuid
        obj["STEP_uuid"] = self_uuid
        obj["STEP_file"] = filepath
        obj["STEP_name"] = name
        obj["STEP_tree_location"] = node_index
        created_uuid[self_uuid] = obj    
    return obj
