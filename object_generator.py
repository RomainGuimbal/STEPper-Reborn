import ntpath
import time
import numpy as np
import bmesh
import bpy

from .trimesh import TriMesh

GLOBAL_FILE_CACHE = {}

def scalemat(mat, sl):
    scaling = np.zeros_like(mat)
    scaling[np.diag_indices(4)] = sl
    # print(scaling)
    return np.matmul(scaling, mat)


def obj_unlink_all(obj):
    """Unlink object from all collections"""
    old_col = obj.users_collection

    # bugfix: not in master collection bug
    # collection_name.objects.unlink(obj)
    if len(old_col) > 0:
        for c in old_col:
            c.objects.unlink(obj)


def add_material(name, color, link_vertex_color=False, overwrite=False):
    assert len(color) == 3
    assert isinstance(color, tuple)
    if len(name) > 60:
        name = name[:60]
    if name not in bpy.data.materials.keys() or overwrite:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True

        # TODO: If language is set to slovensky, this will fail
        # seems to not be issue for other languages tested so far
        # bsdf = mat.node_tree.nodes["Principled BSDF"]
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break

        # Set base color
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)

        # # Connect alpha
        # a = mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"]
        # mat.node_tree.links.new(sn.outputs["Alpha"], a)

        vcol = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
        vcol.location = [-400.0, 300.0]
        vcol.layer_name = "Colors"

        if link_vertex_color:
            mat.node_tree.links.new(vcol.outputs[0], bsdf.inputs[0])
    else:
        mat = bpy.data.materials[name]

    # mat.blend_method = "BLEND"
    # mat.shadow_method = "CLIP"
    # mat.node_tree.nodes["Image Texture"].image = image
    return mat


def bpy_update_object_data(
    objdata, bm, vcol_name, colors, uvs, norms, mat_names, build_materials=True
):
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
    # Apply also in edit mode, not just object mode
    prev_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode="OBJECT")

    bm.to_mesh(objdata)

    if len(norms) > 0:
        # Apply normals to mesh if they exist
        # objdata.use_auto_smooth = True
        # objdata.auto_smooth_angle = 3.14159

        # Filter removed items from norms
        # norms = [n for ni, n in enumerate(norms) if ni not in removed]
        objdata.normals_split_custom_set(np.array(norms))

    # Return to previous object/edit mode
    bpy.ops.object.mode_set(mode=prev_mode)


def calculate_detail_level(dlev):
    """Angular deflection, Linear deflection"""
    if dlev < 100:
        l_def = 100.0 / float(dlev)
    else:
        l_def = (100.0 / float(dlev)) ** 2.0
    return 0.8, l_def


def set_obj_matrix_world(obj, mtx):
    """
    Copy Numpy matrix into Blender matrix
    """
    for row in range(mtx.shape[0]):
        for col in range(mtx.shape[1]):
            obj.matrix_world[row][col] = mtx[row][col]


def create_new_obj_with_mesh(name, set_active=True):
    """
    Create new empty object and mesh, link them, and optionally set to active
    """
    empty_mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, empty_mesh)
    bpy.context.collection.objects.link(obj)
    if set_active:
        bpy.context.view_layer.objects.active = obj
    return obj


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


def transform_to_up(up, chosen_objects, scale, to_cursor=True):
    """
    Set all chosen_objects transforms <up>["X", "Y", "Z"] as up
    Optionally move to cursor <to_cursor>
    Set scale to scale
    """

    # transforms and processing of objects
    # bpy.ops.object.select_all(action="DESELECT")

    cursor_pos = bpy.context.scene.cursor.location

    # up
    # up_as = self.up_as
    up_axis = {"X": 0, "Y": 1, "Z": 2}[up]

    # forward
    # fw_as = self.prg.fw_as
    # fw_axis = {"X": 0, "Y": 1, "Z": 2}[fw_as[0]]

    for obj in chosen_objects:
        # up, forward
        mat = np.array(obj.matrix_world)

        # blender default: Y(1) = forward, Z(2) = up
        if up_axis != 2:
            # if negate axis, do mirror
            # if up_as[1] == "N":
            #     dg = [1, 1, 1, 1]
            #     dg[up_axis] = -1
            #     mat = _scalemat(mat, dg)

            mat[[up_axis, 2]] = mat[[2, up_axis]]
            mat[up_axis] *= -1

        # scale
        mat = scalemat(mat, [*([scale] * 3), 1])

        # move to cursor position
        mat[0][3] += cursor_pos.x
        mat[1][3] += cursor_pos.y
        mat[2][3] += cursor_pos.z

        # apply
        set_obj_matrix_world(obj, mat)

    # Apply scale
    # for obj in created_objs:
    #     # Apply object scale
    #     obj.select_set(True)
    #     bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    #     obj.select_set(False)

    # for obj in created_objs:
    #     obj.select_set(True)


from OCP.AIS import AIS_Shape


def shape_size(shp):
    bb = AIS_Shape(shp).BoundingBox()
    if bb.IsVoid():
        return 1.0
    diag = (bb.CornerMax().Distance(bb.CornerMin())) / 100000
    return diag


def build_mesh(step_reader, obj, shp, lind, angd, vcol_name="Colors"):
    hacks = set([])
    if bpy.context.scene.stepper.hack_skip_zero_solids:
        hacks.add("skip_solids")

    # adaptative = bpy.context.scene.stepper.use_adaptive_resolution
    # if adaptative :
    #     size = shape_size(shp)
    #     angd *= size

    import time

    start_time = time.time()
    mesh: TriMesh = step_reader.build_trimesh(
        shp, lin_def=lind, ang_def=angd, hacks=hacks
    )
    end_time = time.time()
    print(f"Trimesh build time: {end_time - start_time:.2f} seconds")

    mesh.fuse_verts()
    mesh.filter_zero_area()
    mesh.filter_same_face()

    print(f"[bm] {len(mesh.verts)}", end="")
    bm = bmesh.new()
    mesh.add_to_bm(bm, edges_as_seams=True, discontinuity_as_sharp=True)
    mesh.fill_empty_color()
    bpy_update_object_data(
        obj.data,
        bm,
        vcol_name,
        mesh.get_loop_colors(),
        mesh.get_loop_uvs(),
        mesh.get_loop_normals(),
        mesh.get_loop_material_names(),
        build_materials=bpy.context.scene.stepper.build_materials,
    )

    return mesh.matrix


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
        bpy.ops.object.mode_set(mode="OBJECT")
        bm.to_mesh(obj.data)
        bpy.ops.object.mode_set(mode=prev_mode)
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

    wm.progress_begin(0, total)
    for i, (shp, node_index) in enumerate(all_shapes):
        parent_uuid, self_uuid, tag, name, _, local_t, global_t = tree.nodes[
            node_index
        ].get_values()

        if name == "root":
            name = filename + ".empties"

        shape_name = "tt_" + repr(tag)
        wm.progress_update(i)
        obj = None

        # Shape found in leaf
        if shp:
            print(
                "\nBuilding ({}/{}): {} ".format(i + 1, total, name), end="", flush=True
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
                obj = create_new_obj_with_mesh(name)
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
            obj = bpy.data.objects.new(name, None)
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
            obj["STEP_name"] = name
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
