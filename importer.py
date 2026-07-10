# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2021 Tommi Hyppänen
#
# Modified 2025 Romain Guimbal


import os
import numpy as np

from collections import OrderedDict
from dataclasses import dataclass, field

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.gp import gp
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TColStd import TColStd_SequenceOfAsciiString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape, TopoDS
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import (
    XCAFDoc_DocumentTool,
    XCAFDoc_ColorGen,
    XCAFDoc_ColorSurf,
    XCAFDoc_ColorCurv,
)
from OCP.XSControl import XSControl_WorkSession
from OCP.UnitsMethods import (
    UnitsMethods_LengthUnit_Undefined,
    UnitsMethods_LengthUnit_Inch,
    UnitsMethods_LengthUnit_Millimeter,
    UnitsMethods_LengthUnit_Foot,
    UnitsMethods_LengthUnit_Mile,
    UnitsMethods_LengthUnit_Meter,
    UnitsMethods_LengthUnit_Kilometer,
    UnitsMethods_LengthUnit_Mil,
    UnitsMethods_LengthUnit_Micron,
    UnitsMethods_LengthUnit_Centimeter,
    UnitsMethods_LengthUnit_Microinch,
)
from OCP.Interface import Interface_Static

from .trimesh import TrianglesData, TriMesh


def b_colorname(col):
    return Quantity_Color.StringName_s(Quantity_Color.Name(col))


def b_XYZ(v):
    x = v.XYZ()
    return (x.X(), x.Y(), x.Z())


def b_RGB(c):
    return (c.Red(), c.Green(), c.Blue())


def trsf_matrix(shp):
    trsf = shp.Location().Transformation()
    matrix = np.zeros((3, 4), dtype=np.float32)
    for row in range(1, 4):
        for col in range(1, 5):
            matrix[row - 1, col - 1] = trsf.Value(row, col)
    return matrix


def _test_shape(sh):
    tmr = trsf_matrix(sh)
    if np.any(tmr != np.eye(4, dtype=np.float32)[:3, :4]):
        print(tmr)


def force_ascii(i_file):
    from pathlib import Path

    print("Attempting to format STEP file as ASCII 7-bit")
    p = Path(i_file)
    print(p.stat().st_size // 1024, "kB")
    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="ASCII") as fo:
        temp_name = fo.name
        print(temp_name)
        with p.open("rb") as f:
            while il := f.readline():
                fo.write(il.decode("ASCII"))
    print("done ASCII conversion.")
    return temp_name


# TODO: proper parametrization
def equalize_2d_points(pts):
    """Equalize aspect ratio of 2D point dimensions"""
    x_a, x_b = 1.0, 0.0
    y_a, y_b = 1.0, 0.0

    for i, uv in enumerate(pts):
        if uv[0] < x_a:
            x_a = uv[0]
        if uv[0] > x_b:
            x_b = uv[0]
        if uv[1] < y_a:
            y_a = uv[1]
        if uv[1] > y_b:
            y_b = uv[1]

    rx = abs(x_b - x_a)
    ry = abs(y_b - y_a)
    if rx != 0.0 and ry != 0.0:
        ratio = rx / ry
    else:
        ratio = 1.0

    ratio1 = 1 / ratio
    for i, uv in enumerate(pts):
        pts[i] = (pts[i][0] * ratio1, pts[i][1])

    return pts


def get_label_name(label):
    """Return the name of a TDF_Label as a string, fallback to EntryDumpToString or Tag if needed."""

    # Try to use name label if available
    name_attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
        return name_attr.Get().ToExtString()

    # Fallback: use EntryDumpToString or Tag
    if hasattr(label, "EntryDumpToString"):
        return label.EntryDumpToString()
    elif hasattr(label, "Tag"):
        return str(label.Tag())
    else:
        return str(label)


@dataclass
class ShapeTreeNode:
    """
    A node for the OpenCASCADE CAD data ShapeTree
    """

    parent: int
    index: int
    tag: int
    name: str
    children: list[int] = field(default_factory=list)
    local_transform: np.ndarray = field(default_factory=np.eye(4, dtype=np.float32))
    global_transform: np.ndarray = field(default_factory=np.eye(4, dtype=np.float32))
    shape: TopoDS_Shape = None

    def __init__(self, parent, index, tag, name):
        self.parent = parent
        self.index = index
        self.tag = tag
        self.name = name
        self.children = []
        self.local_transform = np.eye(4, dtype=np.float32)
        self.global_transform = np.eye(4, dtype=np.float32)
        self.shape = None

    def get_values(self):
        """
        parent, index, tag, name
        """
        return (
            self.parent,
            self.index,
            self.tag,
            self.name,
            self.shape,
            self.local_transform,
            self.global_transform,
        )

    def set_shape(self, shape):
        if shape:
            if not isinstance(shape, TopoDS_Shape):
                raise ValueError("Input shape is not OpenCASCADE TopoDS_Shape")
            self.shape = shape
        else:
            self.shape = None


class ShapeTree:
    """
    Intermediary data structure to partially abstract OpenCASCADE away from the rest of the program
    """

    def __init__(self):
        self.nodes = []

        # Root node has special values
        self.nodes.append(ShapeTreeNode(-1, 0, -1, "root"))

    def get_root_id(self):
        return 0

    def get_max_id(self):
        return len(self.nodes) - 1

    def add(self, parent, lab) -> ShapeTreeNode:
        """
        Args:
            lab: Shape label
        """
        loc = len(self.nodes)
        node = ShapeTreeNode(parent, loc, lab.Tag(), get_label_name(lab))
        self.nodes[parent].children.append(loc)
        self.nodes.append(node)
        return self.nodes[-1]

    def get_shapes(self):
        # return {i.shape: i.index for i in self.nodes if i.shape}
        return [(i.shape, i.index) for i in self.nodes]

    def print_transforms(self):
        for i in self.nodes:
            print(i.local_transform)


class ReadSTEP:
    def __init__(self, filepath):
        self.filepath = filepath
        self.read_file()

    def query_color(self, lab, overwrite=False):
        """
        Args:
            lab: shape label
        """
        # default color = pink
        color = Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB)
        iscolorset = False
        colortype = None

        shape = self.shape_tool.GetShape_s(lab)
        # print(shape.ShapeType())

        # Overwrites each type
        c_gen_exists = self.color_tool.GetColor(shape, XCAFDoc_ColorGen, color)
        c_surf_exists = self.color_tool.GetColor(shape, XCAFDoc_ColorSurf, color)
        c_curv_exists = False  # self.color_tool.GetColor(shape, XCAFDoc_ColorCurv, color) Supposed to be a fallback but overwrite't

        if c_gen_exists or c_surf_exists or c_curv_exists:
            iscolorset = True
            # Color priority (1/type) is the same as CAD assistant material tree display
            colortype = c_gen_exists * 1 + c_surf_exists * 2 + c_curv_exists * 3

        return color, colortype, iscolorset

    def print_all_colors(self):
        tcol = Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB)
        clabs = TDF_LabelSequence()
        self.color_tool.GetColors(clabs)
        for i in range(clabs.Length()):
            res = self.color_tool.GetColor(clabs.Value(i + 1), tcol)
            if res:
                print(b_colorname(tcol))

    def label_matrix(self, lab):
        trsf = self.shape_tool.GetLocation_s(lab).Transformation()
        matrix = np.eye(4, dtype=np.float32)
        for row in range(1, 4):
            for col in range(1, 5):
                matrix[row - 1, col - 1] = trsf.Value(row, col)
        # print(matrix)
        return matrix

    def explore_partial(self, shp, te_type):
        c_set = set([])
        ex = TopExp_Explorer(shp, te_type)
        # Todo: use label->tag
        while ex.More():
            c = ex.Current()
            if c not in c_set:
                c_set.add(c)
            ex.Next()
        return len(c_set)

    def explore_shape(self, shp):
        return (
            self.explore_partial(shp, TopAbs_COMPOUND),
            self.explore_partial(shp, TopAbs_SOLID),
            self.explore_partial(shp, TopAbs_SHELL),
            self.explore_partial(shp, TopAbs_FACE),
            self.explore_partial(shp, TopAbs_WIRE),
            self.explore_partial(shp, TopAbs_EDGE),
            self.explore_partial(shp, TopAbs_VERTEX),
        )

    def shape_info(self, shp):
        st = self.shape_tool
        lab = self.shape_label[shp]
        vals = (
            st.IsAssembly_s(lab),
            st.IsFree_s(lab),
            st.IsShape_s(lab),
            st.IsCompound_s(lab),
            st.IsComponent_s(lab),
            st.IsSimpleShape_s(lab),
            shp.Locked(),
        )

        lookup = ["A", "F", "S", "C", "T", "s", "L"]
        res = "".join([lookup[i] for i, v in enumerate(vals) if v])

        # res += f", C:{shp.NbChildren()}"

        res += ", C:{} So:{} Sh:{} F:{} Wi:{} E:{} V:{}".format(
            *self.explore_shape(shp)
        )

        return " " + res + " "

    def transfer(self):
        file = self.filepath

        # Create the XCAF document
        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
        app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

        # Init new doc and reader
        reader = STEPCAFControl_Reader()
        reader.SetColorMode(True)
        reader.SetNameMode(True)
        # reader.SetMatMode(True)
        # reader.SetLayerMode(True)

        # Scale
        # Read simple STEP file for correct units
        session = XSControl_WorkSession()
        step_simple_reader = STEPControl_Reader(session)

        status = step_simple_reader.ReadFile(file)
        if status != IFSelect_RetDone:
            raise AssertionError("Error: can't read file. File possibly damaged.")

        # https://dev.opencascade.org/content/loading-step-file-crashes-edgeloop
        # Default is 1, try also 0
        # Interface_Static.SetVal("read.surfacecurve.mode", 3)

        # read units
        ulen_names = TColStd_SequenceOfAsciiString()
        uang_names = TColStd_SequenceOfAsciiString()
        usld_names = TColStd_SequenceOfAsciiString()
        step_simple_reader.FileUnits(ulen_names, uang_names, usld_names)

        # # Info about unit conversions
        # # https://dev.opencascade.org/content/step-unit-conversion-and-meshing

        # # for i in range(ulen_names.Length()):
        # #     ulen = ulen_names.Value(i + 1)
        # #     uang = uang_names.Value(i + 1)
        # #     usld = usld_names.Value(i + 1)
        # #     print(ulen.ToCString(), uang.ToCString(), usld.ToCString())

        # if ulen_names.Length() > 0:
        #     unitname = ulen_names.Value(1).ToCString().lower()

        #     scales_map = {
        #         "millimeter": 1,
        #         "millimetre": 1,
        #         "centimeter": 10,
        #         "centimetre": 10,
        #         "kilometer": 1000000.0,
        #         "kilometre": 1000000.0,
        #         "meter": 1000.0,
        #         "metre": 1000.0,
        #         "inch": 25.4,
        #         "foot": 304.8,
        #         "mile": 1609340,
        #         "mil": 0.0254,
        #     }

        #     scale = scales_map.get(unitname, 1.0)
        #     print(scale)
        XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 1, UnitsMethods_LengthUnit_Meter)

        # Read
        status = reader.ReadFile(file)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to read STEP file: {file}")

        if not reader.Transfer(doc):
            raise RuntimeError(
                f"Failed to transfer STEP data into XCAF document: {file}"
            )
        else:
            print("DataExchange: Transfer done")

        self.doc = doc

    def read_file(self):
        """Returns list of tuples (topods_shape, label, color)
        Use OCAF.
        """

        # Init Reader
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError("%s not found." % self.filepath)

        self.transfer()
        # self.transfer_simple(self.filename)

        self.shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        self.color_tool = XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main())

        # material_tool = XCAFDoc_DocumentTool_MaterialTool(doc.Main())
        # layer_tool = XCAFDoc_DocumentTool_LayerTool(doc.Main())

        # use OrderedDict and make sure the order is maintained through the entire pipeline

        self.shape_label = {}
        self.sub_shapes = OrderedDict()

        self.face_colors = {}
        self.face_color_priority = {}
        self.tag_info = {}
        self.skipped_shapes = set([])
        self.import_problems = {
            "Triangulation": 0,
            "Undefined normals": 0,
            "Empty shape": 0,
        }

        def _set_color_and_get_priority(shape_label, shape):
            """Type == Priority here"""
            c, color_type, has_color = self.query_color(shape_label)
            self.face_colors[shape] = c if has_color else None
            if has_color:
                return color_type
            else:
                return 0

        def _get_sub_shapes(lab, level, tree, leaf_id):

            # print(" " * (2 * level) + get_label_name(lab))
            master_leaf = tree.nodes[leaf_id]
            # l_comps = TDF_LabelSequence()
            # self.shape_tool.GetComponents_s(lab, l_comps)
            if self.shape_tool.IsAssembly_s(lab):
                # Get transform for pure transform (empty)
                # Empty has eye transform, inherit global from parent

                # empty = tree.add(leaf.index, lab, empty=True)
                # output_shapes[shape] = empty

                # Read contained shapes
                l_c = TDF_LabelSequence()
                self.shape_tool.GetComponents_s(lab, l_c)
                for i in range(l_c.Length()):
                    label = l_c.Value(i + 1)
                    if self.shape_tool.IsReference_s(label):
                        label_reference = TDF_Label()
                        self.shape_tool.GetReferredShape_s(label, label_reference)

                        label_transform = self.label_matrix(label)
                        node = tree.add(master_leaf.index, label_reference)
                        new_leaf = tree.nodes[node.index]
                        new_leaf.local_transform = label_transform
                        new_leaf.global_transform = (
                            master_leaf.global_transform @ label_transform
                        )

                        _get_sub_shapes(label_reference, level + 1, tree, node.index)
                    else:
                        # TODO: process rest of the data
                        pass

            elif self.shape_tool.IsSimpleShape_s(lab):
                # TODO: self.shape_label stops being unique when shapes aren't transformed
                shape = self.shape_tool.GetShape_s(lab)
                master_leaf.set_shape(shape)
                if shape in self.shape_label:
                    # Shape already in
                    return

                self.shape_label[shape] = lab

                # TODO Color priority usage looks non-implemented
                self.face_color_priority[shape] = _set_color_and_get_priority(
                    lab, shape
                )

                l_subss = TDF_LabelSequence()
                self.shape_tool.GetSubShapes_s(lab, l_subss)
                self.sub_shapes[shape] = []
                for i in range(l_subss.Length()):
                    lab_subs = l_subss.Value(i + 1)
                    shape_sub = self.shape_tool.GetShape_s(lab_subs)
                    self.shape_label[shape_sub] = lab_subs
                    self.sub_shapes[shape].append(shape_sub)
                    self.face_color_priority[shape_sub] = _set_color_and_get_priority(
                        lab_subs, shape_sub
                    )
            else:
                print("DataExchange error: Item is neither assembly or a simple shape")

        def _get_shapes():
            """
            Root shapes are not part of shapes
            """
            # self.shape_tool.UpdateAssemblies()

            # Get free shapes labels
            labels = TDF_LabelSequence()
            self.shape_tool.GetFreeShapes(labels)

            # Get sub shapes of each free shape and add to tree
            tree = ShapeTree()
            for i in range(labels.Length()):
                print(f"DataExchange: Reading shape ({i + 1}/{labels.Length()})")

                root_item = labels.Value(i + 1)
                node = tree.add(tree.get_root_id(), root_item)
                _get_sub_shapes(root_item, 0, tree, node.index)

            return tree

        tree = _get_shapes()
        self.tree = tree

    def triangulate_face(self, face, tform, brep_tool):
        location = TopLoc_Location()
        facing = brep_tool.Triangulation_s(face, location)
        if facing is None:
            # Mesh error, no triangulation found for part
            self.import_problems["Triangulation"] += 1
            return None

        # nsurf = brep_tool.Surface(face)
        surface = BRepAdaptor_Surface(face)
        prop = BRepLProp_SLProps(surface, 2, gp.Resolution_s())
        # prop = BRepLProp_SLProps(surface, 2, 1e-4)
        # face_uv = facing.UVNode()

        # Calculate UV bounds
        Umin, Umax, Vmin, Vmax = 0.0, 0.0, 0.0, 0.0
        # for t in range(1, face_uv.Length()):
        for t in range(1, facing.NbNodes() + 1):
            # v = face_uv.Value(t)
            v = facing.UVNode(t)
            x, y = v.X(), v.Y()
            if t == 1:
                Umin, Umax, Vmin, Vmax = x, x, y, y
            if x < Umin:
                Umin = x
            if x > Umax:
                Umin = x
            if y < Vmin:
                Vmin = y
            if y > Vmax:
                Vmin = y

        Ucenter = (Umin + Umax) * 0.5
        Vcenter = (Vmin + Vmax) * 0.5

        # tab = facing.Nodes()
        tri = facing.Triangles()

        verts = []
        norms = []
        tris = []
        uvs = []

        undef_normals = False

        itform = tform.Inverted()

        # Build normals
        d_nbnodes = facing.NbNodes()
        for t in range(1, d_nbnodes + 1):
            # pt = tab.Value(t)
            pt = facing.Node(t)
            loc = b_XYZ(pt)

            # nvert = bm.verts.new(loc)
            # nvert.index = t - 1

            # assert len(loc) == 3
            # assert loc[0] is float
            # assert loc is tuple
            verts.append(loc)

            # Get triangulation normal

            # pt = gp_Pnt(loc[0], loc[1], loc[2])
            # pt_surf = GeomAPI_ProjectPointOnSurf(pt, nsurf)
            # fU, fV = pt_surf.Parameters(1)
            # prop = GeomLProp_SLProps(nsurf, fU, fV, 2, gp.Resolution_s())

            uv = facing.UVNode(t)
            u, v = uv.X(), uv.Y()
            uvs.append((u, v))

            # The edges of UV give invalid normals, hence this
            prop.SetParameters(
                (u - Ucenter) * 0.999 + Ucenter, (v - Vcenter) * 0.999 + Vcenter
            )

            if prop.IsNormalDefined():
                normal = prop.Normal().Transformed(itform)
                # normal = prop.Normal()
                nn = np.array(b_XYZ(normal))
                if face.Orientation() == TopAbs_REVERSED:
                    nn = -nn
            else:
                nn = np.array((0.0, 0.0, 1.0))
                undef_normals = True

            # norms.append(tuple(float(nnn) for nnn in nn))
            norms.append(np.float32(nn))

        # Build triangulation
        d_nbtriangles = facing.NbTriangles()
        for t in range(1, d_nbtriangles + 1):
            T1, T2, T3 = tri(t).Get()

            if face.Orientation() != TopAbs_FORWARD:
                T1, T2 = T2, T1

            # v_list = (verts[T1 - 1], verts[T2 - 1], verts[T3 - 1])
            # nf = bm.faces.new(v_list)
            # nf.smooth = True
            # nf.normal_update()
            tris.append((T1 - 1, T2 - 1, T3 - 1))

            # for v in (T1, T2, T3):
            #     if norms[v - 1] is None:
            #         added_norms.append(np.array(nf.normal))
            #     else:
            #         added_norms.append(norms[v - 1])

            # new_norms.append(norms[v - 1])

        if undef_normals:
            self.import_problems["Undefined normals"] += 1

        tris_data = TrianglesData(
            indices=tris,
            # TODO : probably optimizable
            norms=[(norms[t[0]], norms[t[1]], norms[t[2]]) for t in tris],
            uvs=[(uvs[t[0]], uvs[t[1]], uvs[t[2]]) for t in tris],
            colors=[None] * len(tris),  # None means no color
            material=[None] * len(tris),
            material_name=[None] * len(tris),
            batch=[None] * len(tris),
        )

        return TriMesh(verts=verts, tris=tris_data)

    def build_trimesh(
        self, shape, lin_def_bl_unit=0.8, ang_def=0.5, hacks=set([])
    ) -> TriMesh:
        out_mesh = TriMesh()
        out_mesh.matrix = np.eye(4, dtype=np.float32)

        # TODO: this is hack
        if "skip_solids" in hacks and self.explore_partial(shape, TopAbs_SOLID) == 0:
            self.skipped_shapes.add(get_label_name(self.shape_label[shape]))
            return out_mesh

        iter_shapes = [shape] + self.sub_shapes[shape]
        iter_shapes.sort(key=lambda x: x.Checked())

        face_data = OrderedDict()
        batch = 0

        brep_tool = BRep_Tool()

        # Iterate over the main shape and its sub shapes
        for _, shp in enumerate(iter_shapes):
            col = self.face_colors[shp]
            if col is not None:
                col_rgb = b_RGB(col)
                col_name = b_colorname(col)
            else:
                col_name = ""

            # Clean all previous triangulations
            BRepTools.Clean_s(shp)

            # Subshape transforms can be different from the mainshape transform
            ex = TopExp_Explorer(shp, TopAbs_FACE)
            if not ex.More():
                self.import_problems["Empty shape"] += 1
                continue

            brepmesh = BRepMesh_IncrementalMesh(
                shp,
                lin_def_bl_unit
                * 2,  # probably because tolerence range and not max distance
                False,
                ang_def,
                True,  # parallel
            )
            brepmesh.Perform()
            trf = shp.Location().Transformation()

            # Iterate through faces
            while ex.More():
                exc = ex.Current()
                face = TopoDS.Face_s(exc)

                mesh = self.triangulate_face(face, trf, brep_tool)
                if mesh:
                    # If shape or sub-shape has defined color, set it so
                    mesh.set_batch(batch)
                    if col is not None:
                        mesh.colorize(col_rgb)
                        mesh.set_material_name(col_name)

                    # First filter in overwriting a face/color
                    face_data[face] = (0, mesh, "EMPTY")

                ex.Next()
                batch += 1

        for _, b in face_data.items():
            _, mesh, col_name = b
            if len(mesh.verts) > 0:
                out_mesh.add_mesh(mesh)

        print("[l]", end="", flush=True)

        return out_mesh
