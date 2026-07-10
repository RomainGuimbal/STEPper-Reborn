import bpy
import numpy as np
import math


def scalemat(mat, sl):
    scaling = np.zeros_like(mat)
    scaling[np.diag_indices(4)] = sl
    return np.matmul(scaling, mat)


def scale_translation(mat, sl):
    mat[0][3] *= sl
    mat[1][3] *= sl
    mat[2][3] *= sl
    return mat


def obj_unlink_all(obj):
    """Unlink object from all collections"""
    old_col = obj.users_collection

    # bugfix: not in master collection bug
    # collection_name.objects.unlink(obj)
    if len(old_col) > 0:
        for c in old_col:
            c.objects.unlink(obj)


def set_obj_matrix_world(obj, mtx):
    """
    Copy Numpy matrix into Blender matrix
    """
    for row in range(mtx.shape[0]):
        for col in range(mtx.shape[1]):
            obj.matrix_world[row][col] = mtx[row][col]


def calculate_deflections_for_artist_friendly(dlev):
    """
    Angular deflection, Linear deflection. From detail level
    """
    # curve passing through (1,100), (100,0.002), (1000,0.00001)
    # arbitrary values which make sense
    # Factors comes from :
    # 5/3-3/2*log(500) ~= -2.3818
    # 1/2*log(500)-4/3 ~= 0.016152

    l_def = 100 * 10 ** (
        -2.3818 * math.log10(dlev) + 0.016152 * (math.log10(dlev)) ** 2
    )

    # 0.8 ~= 45°
    return 0.8, l_def


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

        if to_cursor:
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
        vcol.location = [-400.0, 200.0]
        vcol.layer_name = "Colors"

        if link_vertex_color:
            mat.node_tree.links.new(vcol.outputs[0], bsdf.inputs[0])
    else:
        mat = bpy.data.materials[name]

    # mat.blend_method = "BLEND"
    # mat.shadow_method = "CLIP"
    # mat.node_tree.nodes["Image Texture"].image = image
    return mat


def vert_of_edges(mesh: bpy.types.Mesh):
    edge_verts = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edge_verts)
    edge_verts = edge_verts.reshape(-1, 2)  # transpose

    return edge_verts


def vert_coordinates_of_edges(mesh: bpy.types.Mesh, edge_verts):
    # Vert positions
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)

    # Edge verts positions
    vert_co_0 = np.empty((len(mesh.edges), 3), dtype=np.float32)
    vert_co_1 = np.empty((len(mesh.edges), 3), dtype=np.float32)
    for i in range(len(mesh.edges)):
        vert_co_0[i] = coords[edge_verts[i, 0]]
        vert_co_1[i] = coords[edge_verts[i, 1]]

    return vert_co_0, vert_co_1


def faces_of_edges(mesh: bpy.types.Mesh):
    # Loop data
    loop_edge = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("edge_index", loop_edge)

    # Poly data
    loop_total = np.empty(len(mesh.polygons), dtype=np.int32)
    loop_start = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    mesh.polygons.foreach_get("loop_start", loop_start)

    # loop index -> poly index
    loop_to_poly = np.repeat(np.arange(len(mesh.polygons)), loop_total)

    # edge index -> list of poly indices
    faces_of_edges = [[] for _ in range(len(mesh.edges))]
    for loop_i, edge_i in enumerate(loop_edge):
        faces_of_edges[edge_i].append(loop_to_poly[loop_i])

    return faces_of_edges


# from OCP.AIS import AIS_Shape


# def shape_size(shp):
#     bb = AIS_Shape(shp).BoundingBox()
#     if bb.IsVoid():
#         return 1.0
#     diag = (bb.CornerMax().Distance(bb.CornerMin())) / 100000
#     return diag
