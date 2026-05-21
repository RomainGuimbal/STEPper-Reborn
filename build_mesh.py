import bpy
import bmesh
import numpy as np
from .trimesh import TriMesh
from .utils import bpy_update_object_data
from .step_reader import build_trimesh


def is_vert_sharp_vectorized(edge_dirs, norms_a, norms_b, margin_sq):
    dot_ab = np.einsum("ij,ij->i", norms_a, norms_b)
    ea = np.einsum("ij,ij->i", edge_dirs, norms_a)
    eb = np.einsum("ij,ij->i", edge_dirs, norms_b)

    proj_dot = dot_ab - ea * eb
    len_sq_a = 1.0 - ea * ea
    len_sq_b = 1.0 - eb * eb

    # not sharp if either normal is too small
    are_too_small = np.logical_or(np.less(len_sq_a, 1e-12), np.less(len_sq_b, 1e-12))

    over_margin = np.less((proj_dot * proj_dot), margin_sq * len_sq_a * len_sq_b)
    return np.logical_and(over_margin, np.logical_not(are_too_small))


def mark_edges(objdata, trimesh: TriMesh):
    # Load into BMesh for color / material assignment.
    # from_mesh() is a single C-level bulk operation, much faster than
    # constructing the BMesh vertex by vertex.
    bm = bmesh.new()
    bm.from_mesh(objdata)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Pre-extract batch per face to avoid repeated Python attribute access in loop
    batches = [t.batch for t in trimesh.tris]

    # Seam edges: mark boundaries between different shape batches
    linked_faces_id = [[lf.index for lf in e.link_faces] for e in bm.edges]

    # Sharp edges: mark where per-vertex normals are discontinuous
    ## Pre-build face_vert_norms[face_idx][global_vert_idx] = norm to replace
    ## the inner range(3) scan with O(1) dict lookups.
    face_vert_norms = [
        {t.indices[j]: t.norms[j] for j in range(3)} for t in trimesh.tris
    ]
    margin = 0.02
    margin_sq = (1 - margin) ** 2

    norms_face1 = [
        face_vert_norms[lf[0]] if len(lf) == 2 else {} for lf in linked_faces_id
    ]
    norms_face2 = [
        face_vert_norms[lf[1]] if len(lf) == 2 else {} for lf in linked_faces_id
    ]
    edge_vert1_id = [e.verts[0].index for e in bm.edges]
    edge_vert2_id = [e.verts[1].index for e in bm.edges]

    get_fallback = lambda a: a if a is not None else np.zeros(3, dtype=np.float32)

    norm_face1_vert1 = np.float32(
        [get_fallback(norms_face1[i].get(v)) for i, v in enumerate(edge_vert1_id)]
    )
    norm_face2_vert1 = np.float32(
        [get_fallback(norms_face2[i].get(v)) for i, v in enumerate(edge_vert1_id)]
    )
    norm_face1_vert2 = np.float32(
        [get_fallback(norms_face1[i].get(v)) for i, v in enumerate(edge_vert2_id)]
    )
    norm_face2_vert2 = np.float32(
        [get_fallback(norms_face2[i].get(v)) for i, v in enumerate(edge_vert2_id)]
    )

    vert1 = np.float32([np.float32(e.verts[0].co) for e in bm.edges])
    vert2 = np.float32([np.float32(e.verts[1].co) for e in bm.edges])
    edge_dir = vert1 - vert2

    is_vert1_sharp = is_vert_sharp_vectorized(
        edge_dir, norm_face1_vert1, norm_face2_vert1, margin_sq
    )
    is_vert2_sharp = is_vert_sharp_vectorized(
        edge_dir, norm_face1_vert2, norm_face2_vert2, margin_sq
    )
    is_sharp = np.logical_and(is_vert1_sharp, is_vert2_sharp)

    bm.free()

    is_seam = np.bool(
        [len(lf) == 2 and batches[lf[0]] != batches[lf[1]] for lf in linked_faces_id]
    )

    if "sharp_edge" not in objdata.attributes:
        objdata.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
    if "uv_seam" not in objdata.attributes:
        objdata.attributes.new(name="uv_seam", type="BOOLEAN", domain="EDGE")
    objdata.update()

    # Get attributes
    sharp_att = objdata.attributes["sharp_edge"]
    seam_att = objdata.attributes["uv_seam"]

    # Set attributes
    sharp_att.data.foreach_set("value", is_sharp)
    seam_att.data.foreach_set("value", is_seam)


def build_shape_mesh(
    name,
    shape,
    sub_shapes_of_shape: list,
    shape_color,
    sub_shapes_colors: list,
    lind=0.8,
    angd=0.5,
    vert_color_name="Colors",
) -> bpy.types.Mesh:
    
    trimesh: TriMesh = build_trimesh(
        shape,
        sub_shapes_of_shape,
        shape_color,
        sub_shapes_colors,
        lin_def=lind,
        ang_def=angd,
        skip_zero_solids=bpy.context.scene.stepper.hack_skip_zero_solids,
    )

    trimesh.fuse_verts()
    trimesh.filter_zero_area()
    trimesh.filter_same_face()
    trimesh.fill_empty_color()

    print(f"[bm] {len(trimesh.verts)}", end="")

    # --- Build geometry at C level (replaces N individual bm.verts.new /
    #     bm.faces.new Python calls in add_to_bm) ---
    objdata = bpy.data.meshes.new(name)
    objdata.from_pydata(trimesh.verts, [], [t.indices for t in trimesh.tris])

    mark_edges(objdata, trimesh)

    bpy_update_object_data(
        objdata,
        vert_color_name,
        trimesh.get_loop_colors(),
        trimesh.get_loop_uvs(),
        trimesh.get_loop_normals(),
        trimesh.get_loop_material_names(),
        build_materials=bpy.context.scene.stepper.build_materials,
    )

    return objdata


def bl_obj_from_mesh_shape(
    mesh,
    shp,
    node,
    name,
    filepath,
    node_index,
    created_uuid,
    total,
    i,
):
    parent_uuid, self_uuid, tag, _, _, _, _ = node.get_values()

    print("\nBuilding ({}/{}): {} ".format(i + 1, total, name), end="", flush=True)
    print("[T" + repr(shp.ShapeType()) + "]", end="", flush=True)

    print("[Build]", end="", flush=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # assign custom properties to blender objects
    obj["STEP_tag"] = tag
    obj["STEP_parent"] = parent_uuid
    obj["STEP_uuid"] = self_uuid
    obj["STEP_file"] = filepath
    obj["STEP_name"] = name
    obj["STEP_tree_location"] = node_index
    created_uuid[self_uuid] = obj

    return obj


def bl_obj_from_instance_shape(
    shp,
    node,
    name,
    filepath,
    node_index,
    created_uuid,
    created_names,
    total,
    i,
):
    parent_uuid, self_uuid, tag, name, _, local_t, global_t = node.get_values()
    tt_tag = "tt_" + repr(tag)

    print("\nBuilding ({}/{}): {} ".format(i + 1, total, name), end="", flush=True)
    print("[T" + repr(shp.ShapeType()) + "]", end="", flush=True)

    # If object already build, just copy it, using linked mesh data
    print("[Link]", end="", flush=True)
    source_obj = created_names[tt_tag]
    obj = source_obj.copy()

    # Object has been created
    # assign custom properties to blender objects
    obj["STEP_tag"] = tag
    obj["STEP_parent"] = parent_uuid
    obj["STEP_uuid"] = self_uuid
    obj["STEP_file"] = filepath
    obj["STEP_name"] = name
    obj["STEP_tree_location"] = node_index
    created_uuid[self_uuid] = obj

    return obj


def bl_hierarchy_empties(node, name, filepath, node_index, created_uuid):
    parent_uuid, self_uuid, tag, name, _, local_t, global_t = node.get_values()

    # Create empty
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 2
    obj.empty_display_type = "PLAIN_AXES"
    # set_obj_matrix_world(obj, global_t)

    # assign custom properties to blender objects
    obj["STEP_tag"] = tag
    obj["STEP_parent"] = parent_uuid
    obj["STEP_uuid"] = self_uuid
    obj["STEP_file"] = filepath
    obj["STEP_name"] = name
    obj["STEP_tree_location"] = node_index
    created_uuid[self_uuid] = obj

    return obj
