# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Mesh building utilities: geometry processing, material creation, timing."""

import numpy as np
import bmesh
import bpy
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------------
# Color / material helpers
# ---------------------------------------------------------------------------

# Color quantization precision for material merging (~1.5% tolerance)
_COLOR_MERGE_PRECISION = 64


def _quantize_color(col):
    """Round color components to merge near-identical materials."""
    return tuple(
        round(c * _COLOR_MERGE_PRECISION) / _COLOR_MERGE_PRECISION for c in col
    )


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


# ---------------------------------------------------------------------------
# Object / transform helpers
# ---------------------------------------------------------------------------


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


def transform_to_up(up, chosen_objects, scale, to_cursor=True, apply_scale=True):
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

    # Apply scale — bake scale into mesh vertices so obj.scale = (1,1,1)
    # Uses direct vertex scaling instead of bpy.ops.object.transform_apply
    # to avoid "Cannot apply to a multi user" errors on instanced meshes.
    if apply_scale and scale != 1.0:
        processed_meshes = set()
        for obj in chosen_objects:
            if obj.data is None:
                continue
            mesh = obj.data
            if mesh not in processed_meshes:
                vert_count = len(mesh.vertices)
                if vert_count > 0:
                    verts = np.empty(vert_count * 3, dtype=np.float32)
                    mesh.vertices.foreach_get("co", verts)
                    verts *= scale
                    mesh.vertices.foreach_set("co", verts)
                    mesh.update()
                processed_meshes.add(mesh)

        for obj in chosen_objects:
            if obj.data is None:
                continue
            loc, rot, _ = obj.matrix_world.decompose()
            obj.matrix_world = Matrix.LocRotScale(loc, rot, Vector((1.0, 1.0, 1.0)))


# ---------------------------------------------------------------------------
# bmesh / vertex-color mesh update (TriMesh path)
# ---------------------------------------------------------------------------


def bpy_update_object_data(
    objdata, bm, vcol_name, colors, uvs, norms, mat_names, build_materials=True
):
    # Flush BMesh (seams, sharp edges) to mesh.  Color and material indices are
    # applied afterwards via the faster mesh-level foreach_set API, which avoids
    # N_tris*3 Python-level BMesh attribute writes.
    prev_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode="OBJECT")
    bm.to_mesh(objdata)
    if len(norms) > 0:
        objdata.normals_split_custom_set(np.array(norms))
    bpy.ops.object.mode_set(mode=prev_mode)

    if not colors:
        return

    n_loops = len(colors)  # = N_tris * 3
    n_faces = n_loops // 3

    # All 3 loops of a triangle share the same per-face color/mat_name.
    # Build per-face RGBA once, then tile to loops with a single numpy repeat.
    face_rgba = np.empty((n_faces, 4), dtype=np.float32)
    face_rgba[:, 3] = 1.0

    mat_index_arr = np.zeros(n_faces, dtype=np.int32) if build_materials else None

    if build_materials:
        obj_mats = {ob_mat.name: obi for obi, ob_mat in enumerate(objdata.materials)}
        mat_counter = 0

    for fi in range(n_faces):
        col = colors[fi * 3]
        mn = mat_names[fi * 3]
        if col is not None and col[0] >= 0.0:
            r, g, b = float(col[0]), float(col[1]), float(col[2])
        else:
            r, g, b = 0.5, 0.5, 0.5
            mn = None
        face_rgba[fi, 0] = r
        face_rgba[fi, 1] = g
        face_rgba[fi, 2] = b

        if build_materials:
            if mn is None:
                mn = "STEP_" + "{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
            if mn not in bpy.data.materials:
                add_material(mn, (r, g, b), link_vertex_color=False)
            if mn not in obj_mats:
                obj_mats[mn] = mat_counter
                objdata.materials.append(bpy.data.materials[mn])
                mat_counter += 1
            mat_index_arr[fi] = obj_mats[mn]

    # Tile face RGBA to per-loop data and write at C level
    flat_rgba = np.repeat(face_rgba, 3, axis=0).ravel()
    color_layer = objdata.vertex_colors.get(vcol_name)
    if color_layer is None:
        color_layer = objdata.vertex_colors.new(name=vcol_name)
    color_layer.data.foreach_set("color", flat_rgba)

    if build_materials:
        objdata.polygons.foreach_set("material_index", mat_index_arr)


# ---------------------------------------------------------------------------
# NURBS builder
# ---------------------------------------------------------------------------


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
