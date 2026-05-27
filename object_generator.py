import ntpath
import time
import numpy as np
import bmesh
import bpy

from .trimesh import TriMesh
from .utils import (
    add_material,
    set_obj_matrix_world,
    obj_unlink_all,
    transform_to_up,
    create_new_obj_with_mesh,
    faces_of_edges,
)

GLOBAL_FILE_CACHE = {}


def bpy_update_object_data(
    objdata, bm, vcol_name, colors, uvs, norms, mat_names, build_materials=True
):
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

    if len(norms) > 0:
        # Apply normals to mesh if they exist
        # objdata.use_auto_smooth = True
        # objdata.auto_smooth_angle = 3.14159

        # Filter removed items from norms
        # norms = [n for ni, n in enumerate(norms) if ni not in removed]
        objdata.normals_split_custom_set(np.array(norms))


def choose_hierarchy_types(htypes):
    """
    Return hierarchy types selection from input string
    """
    hierarchy_flat = False
    hierarchy_tree = False
    hierarchy_empties = False

    if htypes == "FLAT_AND_TREE":
        hierarchy_flat = True
        hierarchy_tree = True
    elif htypes == "TREE":
        hierarchy_tree = True
    elif htypes == "FLAT":
        hierarchy_flat = True
    elif htypes == "EMPTIES":
        hierarchy_empties = True
    else:
        assert False, "Invalid input parameter"

    return hierarchy_flat, hierarchy_tree, hierarchy_empties

# def is_vert_sharp_vectorized(edge_dirs, norms_a, norms_b, margin_sq):
#     dot_ab = np.einsum("ij,ij->i", norms_a, norms_b)
#     ea = np.einsum("ij,ij->i", edge_dirs, norms_a)
#     eb = np.einsum("ij,ij->i", edge_dirs, norms_b)

#     proj_dot = dot_ab - ea * eb
#     len_sq_a = 1.0 - ea * ea
#     len_sq_b = 1.0 - eb * eb

#     # not sharp if either normal is too small
#     are_too_small = np.logical_or(np.less(len_sq_a, 1e-12), np.less(len_sq_b, 1e-12))

#     over_margin = np.less((proj_dot * proj_dot), margin_sq * len_sq_a * len_sq_b)
#     return np.logical_and(over_margin, np.logical_not(are_too_small))


# def find_seams(trimesh, face_pair_of_edges):
#     """
#     Mark boundaries between different shape batches
#     """
#     batches = [
#         t.batch for t in trimesh.tris
#     ]  # TODO: make it a mono-block array of TriMesh
#     is_seam = np.bool([batches[lf[0]] != batches[lf[1]] for lf in face_pair_of_edges])
#     return is_seam

def find_seams(trimesh, bm):
    # mark all non-manifold edges as seams
    for e in bm.edges:
        f = e.link_faces
        # TODO: is batch indexing broken?
        if (
            len(f) == 2
            and trimesh.tris[f[0].index].batch != trimesh.tris[f[1].index].batch
        ):
            e.seam = True
        # if not e.is_manifold:
        #     e.seam = True

# def find_sharp(ob_data, trimesh, face_pair_of_edges, margin):
#     # Sharp edges: mark where per-vertex normals are discontinuous
#     ## Pre-build face_vert_norms[face_idx][global_vert_idx] = norm to replace
#     ## the inner range(3) scan with O(1) dict lookups.

#     # Normals at verts of each triangle
#     face_vert_norms = [
#         {t.indices[j]: t.norms[j] for j in range(3)} for t in trimesh.tris
#     ]

#     # get vert normals for 2 faces of the edge
#     norms_face1 = [face_vert_norms[lf[0]] for lf in face_pair_of_edges]
#     norms_face2 = [face_vert_norms[lf[1]] for lf in face_pair_of_edges]

#     # Vert ids of each edge
#     edge_verts = np.empty(len(ob_data.edges) * 2, dtype=np.int32)
#     ob_data.edges.foreach_get("vertices", edge_verts)
#     edge_verts = edge_verts.reshape(-1, 2)  # transpose

#     # Vert positions
#     coords = np.empty(len(ob_data.vertices) * 3, dtype=np.float32)
#     ob_data.vertices.foreach_get("co", coords)
#     coords = coords.reshape(-1, 3)

#     # Edge verts positions
#     vert_co_0 = np.empty((len(ob_data.edges), 3), dtype=np.float32)
#     vert_co_1 = np.empty((len(ob_data.edges), 3), dtype=np.float32)
#     for i in range(ob_data.edges):
#         vert_co_0[i] = coords[edge_verts[i, 0]]
#         vert_co_1[i] = coords[edge_verts[i, 1]]

#     get_fallback = lambda a: (a if a is not None else np.zeros(3, dtype=np.float32))

#     # Get normals for each face of each vertex of each edge
#     norm_face1_vert1 = np.empty((len(ob_data.edges), 2, 3), dtype=np.float32)
#     for e in ob_data.edges:
#         []

#     norm_face1_vert1 = np.float32(
#         [get_fallback(norms_face1[i].get(v)) for i, v in enumerate(vert1_ex)]
#     )
#     norm_face2_vert1 = np.float32(
#         [get_fallback(norms_face2[i].get(v)) for i, v in enumerate(vert1_ex)]
#     )
#     norm_face1_vert2 = np.float32(
#         [get_fallback(norms_face1[i].get(v)) for i, v in enumerate(vert2_ex)]
#     )
#     norm_face2_vert2 = np.float32(
#         [get_fallback(norms_face2[i].get(v)) for i, v in enumerate(vert2_ex)]
#     )

#     edge_dir = vert_co_0 - vert_co_1

#     margin_sq = (1 - margin) ** 2
#     is_vert1_sharp = is_vert_sharp_vectorized(
#         edge_dir, norm_face1_vert1, norm_face2_vert1, margin_sq
#     )
#     is_vert2_sharp = is_vert_sharp_vectorized(
#         edge_dir, norm_face1_vert2, norm_face2_vert2, margin_sq
#     )
#     is_sharp = np.logical_and(is_vert1_sharp, is_vert2_sharp)

#     return is_sharp
   

def _project_plane_normalize(plane, vec):
    # if False:
    #     l0 = np.linalg.norm(plane)
    #     l1 = np.linalg.norm(vec)
    #     assert l0 > 0.99 and l0 < 1.01
    #     assert l1 > 0.99 and l1 < 1.01
    prj = vec - (plane * np.dot(plane, vec))
    prjn = np.linalg.norm(prj)
    if prjn == 0.0:
        prj = np.array([0.0, 0.0, 1.0])
    else:
        prj /= prjn
    return prj

def _face_normals_disagree_on_edge_plane(plane, norms, margin):
    # Project norms to plane: norm - (plane * dot(plane, norm))
    # .. and calc dots
    p0 = _project_plane_normalize(plane, norms[0])
    dmax = 1.0
    for i in norms[1:]:
        prj = _project_plane_normalize(plane, i)
        dd = np.dot(p0, prj)
        if dd < dmax:
            dmax = dd
    if dmax < 1.0 - margin:
        return True
    return False


def find_sharp(trimesh, bm, margin):
    # gather all normals based on vert index
    # compare dot products for those gathered
    # if above threshold, vert index discontinuity = true
    # go through edges and based on vert indices match mark as sharp

    # norms_in_vert = defaultdict(list)
    # for t in self.tris:
    #     for ni, n in enumerate(t.norms):
    #         norms_in_vert[t.indices[ni]].append(n)

    # Needs to be based on (2) face verts of the edge, not all faces of vert
    for e in bm.edges:
        fi = [f.index for f in e.link_faces]
        if len(fi) != 2:
            continue

        t0, t1 = trimesh.tris[fi[0]], trimesh.tris[fi[1]]

        # find connected face filtered norms for edge verts
        # TODO: maybe optimize this, (test if all are clockwise)
        e_norms = [[], []]
        for i in range(3):
            if t0.indices[i] == e.verts[0].index:
                e_norms[0].append(t0.norms[i])
            if t1.indices[i] == e.verts[0].index:
                e_norms[0].append(t1.norms[i])
            if t0.indices[i] == e.verts[1].index:
                e_norms[1].append(t0.norms[i])
            if t1.indices[i] == e.verts[1].index:
                e_norms[1].append(t1.norms[i])

        # Check the dots on plane defined by the edge as normal
        plane = np.array((e.verts[0].co - e.verts[1].co).normalized())
        if _face_normals_disagree_on_edge_plane(
            plane, e_norms[0], margin
        ) and _face_normals_disagree_on_edge_plane(plane, e_norms[1], margin):
            e.smooth = False
        else:
            e.smooth = True





def mark_edges(objdata, trimesh: TriMesh, seams, sharp, margin=0.02):
    if not(seams or sharp):
        return
    
    # Face pair of edges
    foe = faces_of_edges(objdata)
    face_pair_of_edges = np.zeros((len(objdata.edges), 2), dtype=np.int32)
    for i in range(face_pair_of_edges):
        foei = foe[i]
        if len(foei) == 2:
            face_pair_of_edges[i, 0] = foei[0]
            face_pair_of_edges[i, 1] = foei[1]

    # Compute and createattributes
    if seams :
        is_seam = find_seams(trimesh, face_pair_of_edges)
        if "uv_seam" not in objdata.attributes:
            objdata.attributes.new(name="uv_seam", type="BOOLEAN", domain="EDGE")
    if sharp:
        is_sharp = find_sharp(trimesh, face_pair_of_edges, margin)
        if "sharp_edge" not in objdata.attributes:
            objdata.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
    objdata.update()

    # Get and Set attributes
    if seams :
        seam_att = objdata.attributes["uv_seam"]
        seam_att.data.foreach_set("value", is_seam)
    if sharp:
        sharp_att = objdata.attributes["sharp_edge"]
        sharp_att.data.foreach_set("value", is_sharp)

        


def build_mesh(
    step_reader,
    obj,
    shp,
    lind,
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
        shp, lin_def=lind, ang_def=angd, hacks=hacks
    )

    end_time = time.time()
    print(f"Trimesh build time: {end_time - start_time:.2f} seconds")

    trimesh.fuse_verts()
    trimesh.filter_zero_area()
    trimesh.filter_same_face()

    print(f"[bm] {len(trimesh.verts)}", end="")
    bm = bmesh.new()
    trimesh.add_to_bm(bm)
    trimesh.fill_empty_color()

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


def build_nurbs(step_reader, shp, name):
    nurbs_data = step_reader.build_nurbs(shp)
    debug_faces = False
    if debug_faces:
        obj = create_new_obj_with_mesh(name)
        bm = bmesh.new()
        for nb in nurbs_data:
            nb_u = nb.uv_points
            uw, vw = len(nb_u), len(nb_u[0])
            for u in range(uw - 1):
                nb_v0 = nb_u[u]
                nb_v1 = nb_u[u + 1]
                for v in range(vw - 1):
                    a = bm.verts.new(nb_v0[v].location())
                    b = bm.verts.new(nb_v0[v + 1].location())
                    c = bm.verts.new(nb_v1[v + 1].location())
                    d = bm.verts.new(nb_v1[v].location())
                    bm.faces.new((d, c, b, a))
        prev_mode = bpy.context.object.mode
        bm.to_mesh(obj.data)
        # obj.display_type = 'WIRE'
        return obj
    else:
        blender_nurbs = []
        for nb in nurbs_data:
            surface_data = bpy.data.curves.new("wook", "SURFACE")
            surface_data.dimensions = "3D"

            upoints = nb.uv_points

            usize, vsize = len(upoints), len(upoints[0])

            splines = []
            for v in range(usize):
                spline = surface_data.splines.new(type="NURBS")
                spline.points.add(vsize - 1)
                splines.append(spline)

            for ui, vpoints in enumerate(upoints):
                for vi, p in enumerate(vpoints):
                    # points have weight attribute
                    splines[ui].points[vi].co = p.as_vector()

            blender_nurbs.append(surface_data)

        # print(dir(nurbs[0].splines[0])) =>
        # 'bezier_points', 'bl_rna', 'calc_length', 'character_index', 'hide', 'material_index',
        # 'order_u', 'order_v', 'point_count_u', 'point_count_v', 'points', 'radius_interpolation',
        # 'resolution_u', 'resolution_v', 'rna_type', 'tilt_interpolation', 'type', 'use_bezier_u',
        # 'use_bezier_v', 'use_cyclic_u', 'use_cyclic_v', 'use_endpoint_u',
        # 'use_endpoint_v', 'use_smooth'
        created_objs = []
        for ni, n in enumerate(blender_nurbs):
            occ_nurb = nurbs_data[ni]
            surface_object = bpy.data.objects.new(name, n)
            bpy.context.collection.objects.link(surface_object)
            for s in surface_object.data.splines:
                for p in s.points:
                    p.select = True

            bpy.context.view_layer.objects.active = surface_object
            prev_mode = bpy.context.object.mode
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.make_segment()
            bpy.ops.object.mode_set(mode=prev_mode)
            created_objs.append(surface_object)

        for obi, ob in enumerate(created_objs):
            occ_nurb = nurbs_data[obi]
            for s in ob.data.splines:
                s.use_endpoint_u = True
                s.use_endpoint_v = True
                # s.use_endpoint_u = occ_nurb.u_closed
                # s.use_endpoint_v = occ_nurb.v_closed
                # s.use_cyclic_u = occ_nurb.u_periodic
                # s.use_cyclic_v = occ_nurb.v_periodic
                s.order_u = occ_nurb.u_degree + 1
                s.order_v = occ_nurb.v_degree + 1
                # print(s.order_u, s.order_v, occ_nurb.u_degree, occ_nurb.v_degree)

        # Join objects
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for o in created_objs:
            o.select_set(True)
        bpy.ops.object.join()
        return bpy.context.view_layer.objects.active


def load_step(
    context,
    filepath,
    custom_scale=None,
    lin_deflection=0.8,
    ang_deflection=0.5,
    # merge_distance=0.001,
    up_as="Y",
    htypes="TREE",
):
    from . import importer

    hierarchy_flat, hierarchy_tree, hierarchy_empties = choose_hierarchy_types(htypes)

    filename = "".join(ntpath.basename(filepath).split(".")[:-1])

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

    tree = step_reader.tree
    scale = step_reader.scale
    if custom_scale is not None:
        scale = custom_scale

    # divide by Blender unit length
    scale /= context.scene.unit_settings.scale_length
    print("Current Blender scale set at:", context.scene.unit_settings.scale_length)

    wm = bpy.context.window_manager

    created_objs = []
    created_names = {}
    created_uuid = {}

    # traverse shapes, render in "face" mode
    start_time = time.time()
    all_shapes = tree.get_shapes()
    total = len(all_shapes)

    # Build mesh objects
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
                obj = source_obj.copy()
                created_objs.append(obj)
            else:
                print("[Build]", end="", flush=True)

                # Create new mesh and object from scratch
                obj = create_new_obj_with_mesh(obj_name)
                bpy.ops.object.mode_set(mode="OBJECT")
                build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)

                # TODO: nurbs changes here
                # obj = build_nurbs(step_reader, shp, name)

                created_objs.append(obj)
                created_names[shape_name] = obj

                # bpy.ops.object.mode_set(mode="OBJECT")
                # build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)

        # No shape in leaf, empty creation enabled, do this
        elif hierarchy_empties:
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
            obj["STEP_parent"] = parent_uuid
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

    # build flat collection
    if hierarchy_flat:
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

    # build tree of collections
    if hierarchy_tree:
        tree_collection = bpy.data.collections.new(filename + ".hierarchy")
        bpy.context.scene.collection.children.link(tree_collection)
        hierarchy_collections = {}
        hierarchy_collections[-1] = tree_collection

        def node_parse(node, level, parent_collection):
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
                    hierarchy_collections[obj["STEP_parent"]].objects.link(obj)
                    global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
                    set_obj_matrix_world(obj, global_t)

    # build hierarchy with empties
    if hierarchy_empties:
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

    transform_to_up(up_as[0], created_objs, scale)

    wm.progress_end()
    print(f"STEP loading time elapsed: {time.time()-start_time:.2f}")

    return True
