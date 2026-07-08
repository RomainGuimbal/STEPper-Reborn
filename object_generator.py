import ntpath
import time
import numpy as np
import bmesh
import bpy
from enum import IntEnum
from mathutils import Matrix
from .trimesh import TriMesh
from .utils import (
    add_material,
    set_obj_matrix_world,
    obj_unlink_all,
    transform_to_up,
    create_new_obj_with_mesh,
    faces_of_edges,
    vert_coordinates_of_edges,
    vert_of_edges,
    scale_translation,
)
from .importer import ShapeTreeNode

GLOBAL_FILE_CACHE = {}
ROOT_PARENT = -1


class HierarchyType(IntEnum):
    EMPTIES_TREE = 0
    COLLECTION_FLAT = 1
    COLLECTION_TREE = 2
    COLLECTION_FLAT_AND_TREE = 3
    COLLECTION_INSTANCES = 4


def bpy_update_object_data(
    objdata, vcol_name, colors, uvs, norms, mat_names, build_materials=True
):
    bm = bmesh.new()  # create an empty BMesh
    bm.from_mesh(objdata)

    # Already existing object materials
    if build_materials:
        # set colors and mats
        obj_mats = {}
        for obi, ob_mat in enumerate(objdata.materials):
            obj_mats[ob_mat.name] = obi
        mat_counter = 0

    if len(colors) > 0:
        color_layer = bm.loops.layers.color.get(vcol_name)
        if color_layer is None:
            color_layer = bm.loops.layers.color.new(vcol_name)
        # uv_layer = bm.loops.layers.uv.verify()
        i = 0
        for face in bm.faces:
            mat_col = (0.5, 0.5, 0.5)
            mat_col_name = None
            for loop in face.loops:
                # TODO: good, proper aspect ratio UV
                # loop[uv_layer].uv = uvs[i]
                if colors[i][0] >= 0.0:
                    loop[color_layer] = (*colors[i], 1.0)
                    mat_col = colors[i]
                    mat_col_name = mat_names[i]
                else:
                    # No color: set it to default gray
                    loop[color_layer] = (0.5, 0.5, 0.5, 1.0)
                i += 1

            if build_materials:
                # Translate color into name, if not defined
                if mat_col_name is None:
                    mat_col_name = "STEP_" + "".join(
                        "{0:0{1}x}".format(int(mat_col[i] * 255), 2) for i in range(3)
                    )

                # If material doesn't exist, create it
                if mat_col_name not in bpy.data.materials:
                    add_material(mat_col_name, mat_col, link_vertex_color=False)

                # If material exists but it's not yet in object material slot, add it
                if mat_col_name not in obj_mats:
                    obj_mats[mat_col_name] = mat_counter
                    objdata.materials.append(bpy.data.materials[mat_col_name])
                    mat_counter += 1

                face.material_index = obj_mats[mat_col_name]
    else:
        # TODO: if no colors defined, create and apply default material
        pass

    # print("Polys: {}, Verts: {}".format(len(bm.faces), len(bm.verts)))

    # Save face situation so we can adjust accordingly later
    # pre_faces = bm.faces[:]

    # # Merge verts near each other
    # if merge_distance > 0.0:
    #     print("Removing doubles at distance:", merge_distance)
    #     bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=merge_distance)

    # Remove normals from array which don't exist in the mesh anymore
    # removed = set()
    # for fi, f in enumerate(pre_faces):
    #     if not f.is_valid:
    #         for i in range(fi * 3, fi * 3 + 3):
    #             removed.add(i)

    # Update mesh from Bmesh
    bm.to_mesh(objdata)
    bm.free()

    if len(norms) > 0:
        # Apply normals to mesh if they exist
        # objdata.use_auto_smooth = True
        # objdata.auto_smooth_angle = 3.14159

        # Filter removed items from norms
        # norms = [n for ni, n in enumerate(norms) if ni not in removed]
        objdata.normals_split_custom_set(np.array(norms))


def project_and_compare_normals(proj_plane_normal, norms_a, norms_b, margin_sq):
    """
    Normals are projected and compared.
    Vectorized.
    Computation is simplified so it doesn't resemble a direct projection with normalized vectors.
    """
    dot_ab = np.einsum("ij,ij->i", norms_a, norms_b)
    ea = np.einsum("ij,ij->i", proj_plane_normal, norms_a)
    eb = np.einsum("ij,ij->i", proj_plane_normal, norms_b)

    proj_dot = dot_ab - ea * eb
    len_sq_a = 1.0 - ea * ea
    len_sq_b = 1.0 - eb * eb

    # not sharp if either normal is too small
    are_too_small = np.logical_or(np.less(len_sq_a, 1e-12), np.less(len_sq_b, 1e-12))

    over_margin = np.less((proj_dot * proj_dot), margin_sq * len_sq_a * len_sq_b)
    return np.logical_and(over_margin, np.logical_not(are_too_small))


def find_seams(trimesh, face_pair_of_edges):
    """
    Mark boundaries between different shape batches
    """
    batches = trimesh.tris.batch
    is_seam = np.bool([batches[f1] != batches[f2] for f1, f2 in face_pair_of_edges])
    return is_seam


def find_sharp(ob_data, trimesh, face_pair_of_edges, margin):
    # Sharp edges: mark where per-vertex normals are discontinuous
    edge_count = len(ob_data.edges)

    # for each tris, stores all normals of the 3 vertices
    face_vert_norms = [
        {
            trimesh.tris.indices[i][0]: trimesh.tris.norms[i][0],
            trimesh.tris.indices[i][1]: trimesh.tris.norms[i][1],
            trimesh.tris.indices[i][2]: trimesh.tris.norms[i][2],
        }
        for i in range(len(trimesh.tris))
    ]

    # get the 3 verts normals for the 2 faces of each edges
    norms_face1 = [face_vert_norms[f1] for f1, _ in face_pair_of_edges]
    norms_face2 = [face_vert_norms[f2] for _, f2 in face_pair_of_edges]

    # Vert ids of each edge
    edge_verts = vert_of_edges(ob_data)

    # Get normals for each face of each vertex of each edge
    norm_face1_vert1 = np.empty((edge_count, 3), dtype=np.float32)
    norm_face2_vert1 = np.empty((edge_count, 3), dtype=np.float32)
    norm_face1_vert2 = np.empty((edge_count, 3), dtype=np.float32)
    norm_face2_vert2 = np.empty((edge_count, 3), dtype=np.float32)

    get_fallback = lambda a: (a if a is not None else np.zeros(3, dtype=np.float32))
    for i in range(edge_count):
        v1, v2 = edge_verts[i]
        norm_face1_vert1[i] = get_fallback(norms_face1[i].get(v1))
        norm_face2_vert1[i] = get_fallback(norms_face2[i].get(v1))
        norm_face1_vert2[i] = get_fallback(norms_face1[i].get(v2))
        norm_face2_vert2[i] = get_fallback(norms_face2[i].get(v2))

    # Edge normal plane's normal vector
    vert_co_0, vert_co_1 = vert_coordinates_of_edges(ob_data, edge_verts)
    edge_dir = vert_co_0 - vert_co_1

    # margin squared is used for faster computation
    margin_sq = (1 - margin) ** 2
    is_vert1_sharp = project_and_compare_normals(
        edge_dir, norm_face1_vert1, norm_face2_vert1, margin_sq
    )
    is_vert2_sharp = project_and_compare_normals(
        edge_dir, norm_face1_vert2, norm_face2_vert2, margin_sq
    )
    is_sharp = np.logical_and(is_vert1_sharp, is_vert2_sharp)

    return is_sharp


def mark_edges(objdata, trimesh: TriMesh, seams, sharp, margin=0.02):
    if not (seams or sharp):
        return

    # Face pair of edges
    foe = faces_of_edges(objdata)
    face_pair_of_edges = np.zeros((len(objdata.edges), 2), dtype=np.int32)
    for i in range(len(face_pair_of_edges)):
        foei = foe[i]
        if len(foei) == 2:
            face_pair_of_edges[i, 0] = foei[0]
            face_pair_of_edges[i, 1] = foei[1]

    # Compute and createattributes
    if seams:
        is_seam = find_seams(trimesh, face_pair_of_edges)
        if "uv_seam" not in objdata.attributes:
            objdata.attributes.new(name="uv_seam", type="BOOLEAN", domain="EDGE")
    if sharp:
        is_sharp = find_sharp(objdata, trimesh, face_pair_of_edges, margin)
        if "sharp_edge" not in objdata.attributes:
            objdata.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
    objdata.update()

    # Get and Set attributes
    if seams:
        seam_att = objdata.attributes["uv_seam"]
        seam_att.data.foreach_set("value", is_seam)
    if sharp:
        sharp_att = objdata.attributes["sharp_edge"]
        sharp_att.data.foreach_set("value", is_sharp)


def build_mesh(
    step_reader,
    obj,
    shp,
    lind_bl_unit,  # bl_unit
    angd,
    vcol_name="Colors",
    edges_as_seams=True,
    discontinuity_as_sharp=True,
):
    hacks = set([])
    if bpy.context.scene.stepper.hack_skip_zero_solids:
        hacks.add("skip_solids")

    import time

    start_time = time.time()

    trimesh: TriMesh = step_reader.build_trimesh(
        shp, lin_def_bl_unit=lind_bl_unit, ang_def=angd, hacks=hacks
    )

    end_time = time.time()
    print(f"Trimesh build time: {end_time - start_time:.2f} seconds")

    trimesh.fuse_verts()
    trimesh.filter_zero_area()
    trimesh.filter_same_face()
    trimesh.fill_empty_color()

    # Temporary fix. Original addon was apparently smarter at replacing only the rebuilt faces
    obj.data.clear_geometry()

    trimesh.add_to_mesh(obj.data)
    obj.data.update()

    if trimesh.tris:
        mark_edges(
            obj.data, trimesh, edges_as_seams, discontinuity_as_sharp, margin=0.02
        )

    bpy_update_object_data(
        obj.data,
        vcol_name,
        trimesh.get_loop_colors(),
        trimesh.get_loop_uvs(),
        trimesh.get_loop_normals(),
        trimesh.get_loop_material_names(),
        build_materials=bpy.context.scene.stepper.build_materials,
    )

    return trimesh.matrix


def build_hierarchy_collection(tree, created_objs, filename):
    tree_collection = bpy.data.collections.new(filename + ".hierarchy")
    bpy.context.scene.collection.children.link(tree_collection)
    hierarchy_collections = {}
    hierarchy_collections[-1] = tree_collection

    def node_parse(
        node: ShapeTreeNode, level: int, parent_collection: bpy.types.Collection
    ):
        # if "name" in node and node["children"] is not None:
        if len(node.children) > 0:
            collection_node = bpy.data.collections.new(node.name)
            assert node.index not in hierarchy_collections
            hierarchy_collections[node.index] = collection_node

            parent_collection.children.link(collection_node)
            for c in node.children:
                node_parse(tree.nodes[c], level + 1, collection_node)

    root = tree.nodes[0]
    if len(root.children) > 0:
        for c in root.children:
            node_parse(tree.nodes[c], 0, tree_collection)

        # link objects to tree
        if len(hierarchy_collections.items()) > 0:
            for obj in created_objs:
                parent_col_uuid = obj.get("STEP_parent", ROOT_PARENT)
                parent_col = hierarchy_collections.get(
                    parent_col_uuid, hierarchy_collections[-1]
                )
                parent_col.objects.link(obj)
                global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
                set_obj_matrix_world(obj, global_t)

    return hierarchy_collections


def build_hierarchy_empties(tree, created_objs, created_uuid):
    for obj in created_objs:
        global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
        set_obj_matrix_world(obj, global_t)
        bpy.context.scene.collection.objects.link(obj)

        # Parent objs
        parent_id = obj["STEP_parent"]
        if parent_id in created_uuid:
            parent = created_uuid[parent_id]
            obj.parent = parent
            obj.matrix_parent_inverse = parent.matrix_world.inverted()


def build_hierarchy_flat(tree, created_objs, filename):
    flat_collection = bpy.data.collections.new(filename + ".flat")
    bpy.context.scene.collection.children.link(flat_collection)

    created_collections = {}
    for obj in created_objs:
        group_name = obj["STEP_name"]

        # max collection name len = 61
        if len(group_name) > 50:
            group_name = group_name[:25] + "_" + group_name[-25:]

        # TODO: check dupe collections for dupe imports
        if group_name not in created_collections:
            group_collection = bpy.data.collections.new(group_name)
            created_collections[group_name] = group_collection
            flat_collection.children.link(group_collection)
        else:
            group_collection = created_collections[group_name]

        global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
        set_obj_matrix_world(obj, global_t)
        group_collection.objects.link(obj)


def build_collection_instances(instanced_objects, hierarchy_collections, scale):
    # Add container at root collection
    root_col = hierarchy_collections[-1]
    components_container = bpy.data.collections.new("Components")

    # Put component collection first
    other_children_snapshot = list(root_col.children).copy()
    for c in other_children_snapshot:
        root_col.children.unlink(c)

    root_col.children.link(components_container)

    for c in other_children_snapshot:
        root_col.children.link(c)

    # Exclude
    layer_col = bpy.context.view_layer.layer_collection.children[
        root_col.name
    ].children[components_container.name]
    layer_col.exclude = True

    # for each instanced object
    for source_obj, instances_info in instanced_objects.items():
        # put original in a collection to be instanced and reset its transform
        component_col = bpy.data.collections.new(source_obj.name)
        components_container.children.link(component_col)
        parent_collection = source_obj.users_collection[0]
        trsf = source_obj.matrix_world.copy()
        source_obj.matrix_world = Matrix()
        source_obj.users_collection[0].objects.unlink(source_obj)
        component_col.objects.link(source_obj)

        # instance original
        empty = bpy.data.objects.new(source_obj.name, None)
        empty.instance_type = "COLLECTION"
        empty.instance_collection = component_col
        empty.matrix_world = trsf
        empty.location *= scale
        empty.empty_display_size = scale
        parent_collection.objects.link(empty)

        # instance copies
        for shape_name, local_t, global_t, parent_uuid in instances_info:
            # create instance
            empty = bpy.data.objects.new(source_obj.name + "_instance", None)
            empty.instance_type = "COLLECTION"
            empty.instance_collection = component_col
            empty.empty_display_size = scale
            scale_translation(global_t, scale)
            set_obj_matrix_world(empty, global_t)
            parent_col = hierarchy_collections.get(
                parent_uuid, hierarchy_collections[-1]
            )
            parent_col.objects.link(empty)


def load_step(
    context,
    filepath,
    custom_scale=None,
    lin_deflection=0.8,
    ang_deflection=0.5,
    # merge_distance=0.001,
    up_as="Y",
    htypes=HierarchyType.COLLECTION_TREE,
):
    # Find file
    from . import importer

    filename = "".join(ntpath.basename(filepath).split(".")[:-1])

    # Try retrieve step_reader from cache
    if filepath not in GLOBAL_FILE_CACHE:
        try:
            step_reader = importer.ReadSTEP(filepath)
            GLOBAL_FILE_CACHE[filepath] = step_reader
        except AssertionError as e:
            print(e)
            return False
    else:
        step_reader = GLOBAL_FILE_CACHE[filepath]
        print("Loaded file from cache")

    # Get Tree
    tree = step_reader.tree

    # Define final object scale
    scale_final = step_reader.scale_stp
    if custom_scale is not None:
        scale_final = custom_scale
    # Divide by Blender unit length
    scale_final /= context.scene.unit_settings.scale_length
    print("Current Blender scale set at:", context.scene.unit_settings.scale_length)

    # Init data
    wm = bpy.context.window_manager
    created_objs = []
    created_names = {}
    created_uuid = {}
    instanced_objects = {}  # {Original ref : [instances info]}

    # Traverse shapes, render in "face" mode
    start_time = time.time()
    all_shapes = tree.get_shapes()
    total = len(all_shapes)

    # Build shapes objects
    wm.progress_begin(0, total)
    for i, (shp, node_index) in enumerate(all_shapes):
        parent_uuid, self_uuid, tag, obj_name, _, local_t, global_t = tree.nodes[
            node_index
        ].get_values()

        if obj_name == "root":
            obj_name = filename + ".empties"

        shape_name = "tt_" + repr(tag)
        wm.progress_update(i)
        obj = None

        # Shape found in leaf
        if shp:
            print(
                "\nBuilding ({}/{}): {} ".format(i + 1, total, obj_name),
                end="",
                flush=True,
            )
            print("[T" + repr(shp.ShapeType()) + "]", end="", flush=True)

            # If object already build, just copy it, using linked mesh data
            if shape_name in created_names:
                print("[Link]", end="", flush=True)

                source_obj = created_names[shape_name]
                if htypes == HierarchyType.COLLECTION_INSTANCES:
                    if source_obj in instanced_objects:
                        parent_for_instance = (
                            parent_uuid if parent_uuid != 0 else ROOT_PARENT
                        )
                        instanced_objects[source_obj].append(
                            (shape_name, local_t, global_t, parent_for_instance)
                        )
                    else:
                        parent_for_instance = (
                            parent_uuid if parent_uuid != 0 else ROOT_PARENT
                        )
                        instanced_objects[source_obj] = [
                            (shape_name, local_t, global_t, parent_for_instance)
                        ]
                else:
                    obj = source_obj.copy()
                    created_objs.append(obj)

            else:
                print("[Build]", end="", flush=True)

                # Create new mesh and object from scratch
                obj = create_new_obj_with_mesh(obj_name)
                bpy.ops.object.mode_set(mode="OBJECT")
                build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)

                created_objs.append(obj)
                created_names[shape_name] = obj

                # bpy.ops.object.mode_set(mode="OBJECT")
                # build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)

        # No shape in leaf, empty creation enabled, do this
        elif htypes == HierarchyType.EMPTIES_TREE:
            # Create empty
            obj = bpy.data.objects.new(obj_name, None)
            obj.empty_display_size = 2
            obj.empty_display_type = "PLAIN_AXES"
            created_objs.append(obj)
            # set_obj_matrix_world(obj, global_t)

        # Object has been created
        if obj:
            # assign property to obj
            obj["STEP_tag"] = tag
            obj_parent = parent_uuid if parent_uuid != 0 else ROOT_PARENT
            obj["STEP_parent"] = obj_parent
            obj["STEP_uuid"] = self_uuid
            obj["STEP_file"] = filepath
            obj["STEP_name"] = obj_name
            obj["STEP_tree_location"] = node_index
            created_uuid[self_uuid] = obj

    # assert len(created_objs) == len(shapes_labels)
    print("\n" + repr(step_reader.import_problems))

    # remove all temporary links
    for tobj in created_objs:
        obj_unlink_all(tobj)

    match htypes:
        case HierarchyType.COLLECTION_FLAT_AND_TREE:
            build_hierarchy_flat(tree, created_objs, filename)
            build_hierarchy_collection(tree, created_objs, filename)
        case HierarchyType.COLLECTION_TREE:
            build_hierarchy_collection(tree, created_objs, filename)
        case HierarchyType.COLLECTION_FLAT:
            build_hierarchy_flat(tree, created_objs, filename)
        case HierarchyType.EMPTIES_TREE:
            build_hierarchy_empties(tree, created_objs, created_uuid)
        case HierarchyType.COLLECTION_INSTANCES:
            hierarchy_collections = build_hierarchy_collection(
                tree, created_objs, filename
            )
            build_collection_instances(instanced_objects, hierarchy_collections, scale_final)

    transform_to_up(up_as[0], created_objs, scale_final)

    wm.progress_end()
    print(f"STEP loading time elapsed: {time.time()-start_time:.2f}")

    return True
