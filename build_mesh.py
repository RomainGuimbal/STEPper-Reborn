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

    def _project_vector_onto_plane_normalized(plane, vector):
        projected_vector = vector - (plane * np.dot(plane, vector))
        projected_norm = np.linalg.norm(projected_vector)
        return (
            projected_vector / projected_norm
            if projected_norm != 0.0
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

    def _face_normals_disagree_on_edge_plane(plane, normal_a, normal_b, margin):
        projected_a = _project_vector_onto_plane_normalized(plane, normal_a)
        projected_b = _project_vector_onto_plane_normalized(plane, normal_b)
        return np.dot(projected_a, projected_b) < 1.0 - margin

    sharp_norm_margin = 0.02
    for e in bm.edges:
        face_indices = [f.index for f in e.link_faces]
        if len(face_indices) != 2:
            continue

        norms_face_0 = face_vert_norms[face_indices[0]]
        norms_face_1 = face_vert_norms[face_indices[1]]
        vert_index_a = e.verts[0].index
        vert_index_b = e.verts[1].index

        norm_a_face_0 = norms_face_0.get(vert_index_a)
        norm_a_face_1 = norms_face_1.get(vert_index_a)
        norm_b_face_0 = norms_face_0.get(vert_index_b)
        norm_b_face_1 = norms_face_1.get(vert_index_b)
        if (
            norm_a_face_0 is None
            or norm_a_face_1 is None
            or norm_b_face_0 is None
            or norm_b_face_1 is None
        ):
            e.smooth = True
            continue

        edge_direction = np.array((e.verts[0].co - e.verts[1].co).normalized())
        if _face_normals_disagree_on_edge_plane(
            edge_direction, norm_a_face_0, norm_a_face_1, sharp_norm_margin
        ) and _face_normals_disagree_on_edge_plane(
            edge_direction, norm_b_face_0, norm_b_face_1, sharp_norm_margin
        ):
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
