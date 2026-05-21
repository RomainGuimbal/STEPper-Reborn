import bpy
import bmesh
import time
import ntpath
import numpy as np
from OCP import TopAbs
from .trimesh import TriMesh
from .utils import bpy_update_object_data
from .step_reader import build_trimesh
from .utils import (
    obj_unlink_all,
    transform_to_up,
    choose_hierarchy_types,
)
from multiprocessing import Pool
from functools import partial
from .build_blender_hierarchy import build_blender_hierarchy

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
    sub_shapes_of_shape: list[TopAbs.TopAbs_SHAPE],
    shape_color,
    sub_shapes_colors: list,
    lind=0.8,
    angd=0.5,
    vert_color_name="Colors",
) -> bpy.types.Mesh:
    
    trimesh, empty_shape_count = build_trimesh(
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



def load_step(
    context,
    filepath,
    step_reader,
    custom_scale=None,
    lin_deflection=0.8,
    ang_deflection=0.5,
    # merge_distance=0.001,
    up_as="Y",
    htypes="TREE",
):
    hierarchy_flat, hierarchy_tree, hierarchy_empties = choose_hierarchy_types(htypes)

    filename = "".join(ntpath.basename(filepath).split(".")[:-1])

    tree = step_reader.tree
    scale = step_reader.scale
    if custom_scale is not None:
        scale = custom_scale

    # divide by Blender unit length
    scale /= context.scene.unit_settings.scale_length
    print("Current Blender scale set at:", context.scene.unit_settings.scale_length)

    wm = bpy.context.window_manager
    wm.progress_begin(0, total)

    created_objs = []
    created_names = {}
    created_uuid = {}

    # traverse shapes, render in "face" mode
    start_time = time.time()
    all_shapes = tree.get_shapes()
    total = len(all_shapes)

    # Generate meshes

    # Gather parameters
    # Split shapes depending unique or not
    all_tt_tags = [
        "tt_" + repr(tree.nodes[node_index].tag) for _, node_index in all_shapes
    ]
    shp_dict = dict(zip(all_tt_tags, all_shapes))
    unique_tt_tags_set = set()  # alleged as 5000x faster for lookup
    unique_tt_tags = []  # kind of forced to have both to preserve order
    unique_shapes_tp = [] # shape tuples
    instanced_shapes = []
    empties = []
    for t in all_tt_tags:
        shp = shp_dict.get(t)
        if shp[0]:
            if t in unique_tt_tags_set:
                instanced_shapes.append(shp)
            else:
                unique_shapes_tp.append(shp)
                unique_tt_tags_set.add(t)
                unique_tt_tags.append(t)
        else:  # is empty shape
            empties.append(shp[1])  # append just the index

    unique_shapes = [s for s,_ in unique_shapes_tp]

    # Rename roots
    rename = lambda name: name if name != "root" else filename + ".empties"
    names_of_unique = [tree.nodes[node_index].name for _, node_index in unique_shapes_tp]
    names_of_instances = [
        tree.nodes[node_index].name for _, node_index in instanced_shapes
    ]
    names_of_empties = [rename(tree.nodes[node_index].name) for node_index in empties]

    # Other params
    sub_shapes_of_shapes = [step_reader.sub_shapes[shape] for shape in unique_shapes]
    shape_colors = [step_reader.face_colors[shape] for shape in unique_shapes]
    sub_shapes_colors = [
        [step_reader.face_colors[sub_shp] for sub_shp in sub_shapes_of_shapes[i]]
        for i in range(len(unique_shapes_tp))
    ]

    # TODO flatten subshapes and shapes in a single list (/!\ and preserve instancing)
    args = zip(
        names_of_unique,
        unique_shapes,
        sub_shapes_of_shapes,
        shape_colors,
        sub_shapes_colors,
    )

    # Deflections are fixed for all tasks
    worker = partial(build_shape_mesh, lind=lin_deflection, angd=ang_deflection)
    # # Exectute meshing
    # with Pool(4) as pool:
    #     objsdata = pool.map(worker, args)

    # Multiprocess fails for the moment, use that instead
    objsdata = [worker(*a) for a in args]

    # Create objects with mesh
    for i, (shp, node_index) in enumerate(unique_shapes_tp):
        obj = bl_obj_from_mesh_shape(
            objsdata[i],
            shp,
            tree.nodes[node_index],
            names_of_unique[i],
            filepath,
            node_index,
            created_uuid,
            total,
            i,
        )
        created_objs.append(obj)
        created_names[unique_tt_tags[i]] = obj
        wm.progress_update(i)

    # Create instanced objects
    for i, (shp, node_index) in enumerate(instanced_shapes):
        created_objs.append(
            bl_obj_from_instance_shape(
                shp,
                tree.nodes[node_index],
                names_of_instances[i],
                filepath,
                node_index,
                created_uuid,
                created_names,
                total,
                i,
            )
        )

    # Create empties objects
    if hierarchy_empties:
        for i, node_index in enumerate(empties):
            created_objs.append(
                bl_hierarchy_empties(
                    tree.nodes[node_index],
                    names_of_empties[i],
                    filepath,
                    node_index,
                    created_uuid,
                )
            )

    # assert len(created_objs) == len(shapes_labels)
    print("\n" + repr(step_reader.import_problems))

    # remove all temporary links (for RAM ?)
    for tobj in created_objs:
        obj_unlink_all(tobj)

    # build hierarchy
    build_blender_hierarchy(
        filename,
        tree,
        created_objs,
        hierarchy_flat,
        hierarchy_tree,
        hierarchy_empties,
        created_uuid,
    )

    transform_to_up(up_as[0], created_objs, scale)

    wm.progress_end()
    print(f"STEP loading time elapsed: {time.time()-start_time:.2f}")

    return True