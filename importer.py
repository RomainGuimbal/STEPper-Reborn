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

import sys
from os.path import dirname

file_dirname = dirname(__file__)
if file_dirname not in sys.path:
    sys.path.append(file_dirname)

import importlib
import os
import random
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field

import numpy as np

# import trimesh works in dev, but not in deploy
from . import trimesh
from . import nurbs

importlib.reload(trimesh)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_Transform
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomConvert import geomconvert_SurfaceToBSplineSurface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp, gp_Dir, gp_Pln, gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec, gp_XYZ

# from OCC.Core.Standard import Standard_Real
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.IMeshTools import IMeshTools_Parameters
from OCC.Core.Interface import Interface_Static_SetIVal
from OCC.Core.Poly import poly
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.STEPControl import STEPControl_Reader

# from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TColStd import TColStd_SequenceOfAsciiString
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_INTERNAL,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
    topabs_ShapeTypeToString,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location

# from OCC.Core.TopExp import topexp_MapShapes
# from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.XCAFApp import XCAFApp_Application_GetApplication
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorCurv
from OCC.Core.XSControl import XSControl_WorkSession

import OCC

print("--> STEPper OpenCASCADE version:", OCC.VERSION)


def b_colorname(col):
    return Quantity_Color.StringName(Quantity_Color.Name(col))


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


def nurbs_parse(current_face):
    """Get NURBS points for a TopAbs_FACE"""

    _test_shape(current_face)
    nurbs_converter = BRepBuilderAPI_NurbsConvert(current_face)
    nurbs_converter.Perform(current_face)
    result_shape = nurbs_converter.Shape()
    _test_shape(result_shape)
    brep_face = BRep_Tool.Surface(topods.Face(result_shape))
    occ_face = geomconvert_SurfaceToBSplineSurface(brep_face)
    # _test_shape(occ_face)

    # extract the Control Points of each face
    n_poles_u = occ_face.NbUPoles()
    n_poles_v = occ_face.NbVPoles()

    # cycle over the poles to get their coordinates
    points = []
    for pole_u_direction in range(n_poles_u):
        points.append([])
        for pole_v_direction in range(n_poles_v):
            pos = (pole_u_direction + 1, pole_v_direction + 1)
            coords = occ_face.Pole(*pos)
            np_coords = np.array((coords.X(), coords.Y(), coords.Z()))
            weight = occ_face.Weight(*pos)
            pt = nurbs.NurbsPoint((*np_coords, weight))
            points[-1].append(pt)

    # Get surface data (closed, periodic, degree)
    assert len(points) > 1
    assert len(points[0]) > 1
    nbd = nurbs.NurbsData(points)

    nbd.u_closed = occ_face.IsUClosed()
    nbd.v_closed = occ_face.IsVClosed()
    nbd.u_periodic = occ_face.IsUPeriodic()
    nbd.v_periodic = occ_face.IsVPeriodic()
    nbd.u_degree = occ_face.UDegree()
    nbd.v_degree = occ_face.VDegree()

    return nbd


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

    def add(self, parent, label) -> ShapeTreeNode:
        loc = len(self.nodes)
        node = ShapeTreeNode(parent, loc, label.Tag(), label.GetLabelName())
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
    def __init__(self, filename):
        self.read_file(filename)

    def query_color(self, label, overwrite=False):
        # default color = pink
        c = Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB)
        colorset = False
        colortype = None

        c_gen = self.color_tool.GetColor(label, XCAFDoc_ColorGen, c)
        c_surf = self.color_tool.GetColor(label, XCAFDoc_ColorSurf, c)
        c_curv = self.color_tool.GetColor(label, XCAFDoc_ColorCurv, c)
        if c_gen or c_surf or c_curv:
            colorset = True
            colortype = c_gen * 1 + c_surf * 2 + c_curv * 3

        return c, colortype, colorset

    def print_all_colors(self):
        tcol = Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB)
        clabs = TDF_LabelSequence()
        self.color_tool.GetColors(clabs)
        for i in range(clabs.Length()):
            res = self.color_tool.GetColor(clabs.Value(i + 1), tcol)
            if res:
                print(b_colorname(tcol))

    def label_matrix(self, lab):
        trsf = self.shape_tool.GetLocation(lab).Transformation()
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
            st.IsAssembly(lab),
            st.IsFree(lab),
            st.IsShape(lab),
            st.IsCompound(lab),
            st.IsComponent(lab),
            st.IsSimpleShape(lab),
            shp.Locked(),
        )

        lookup = ["A", "F", "S", "C", "T", "s", "L"]
        res = "".join([lookup[i] for i, v in enumerate(vals) if v])

        # res += f", C:{shp.NbChildren()}"

        res += ", C:{} So:{} Sh:{} F:{} Wi:{} E:{} V:{}".format(*self.explore_shape(shp))

        return " " + res + " "

    def transfer_with_units(self, filename):
        print("Init transfer with units")

        # Init new doc and reader
        doc = TDocStd_Document("STEP")
        step_reader = STEPCAFControl_Reader()
        step_reader.SetColorMode(True)
        step_reader.SetNameMode(True)
        step_reader.SetMatMode(True)
        step_reader.SetLayerMode(True)

        # Read simple STEP file for correct units
        session = XSControl_WorkSession()
        step_simple_reader = STEPControl_Reader(session)

        print("DataExchange: Reading STEP")

        status = step_simple_reader.ReadFile(filename)
        if status != IFSelect_RetDone:
            raise AssertionError("Error: can't read file. File possibly damaged.")

        print("STEP read into memory")

        # https://dev.opencascade.org/content/loading-step-file-crashes-edgeloop
        # Default is 1, try also 0
        # Interface_Static_SetIVal("read.surfacecurve.mode", 3)

        # read units
        ulen_names = TColStd_SequenceOfAsciiString()
        uang_names = TColStd_SequenceOfAsciiString()
        usld_names = TColStd_SequenceOfAsciiString()
        step_simple_reader.FileUnits(ulen_names, uang_names, usld_names)

        # Info about unit conversions
        # https://dev.opencascade.org/content/step-unit-conversion-and-meshing

        # for i in range(ulen_names.Length()):
        #     ulen = ulen_names.Value(i + 1)
        #     uang = uang_names.Value(i + 1)
        #     usld = usld_names.Value(i + 1)
        #     print(ulen.ToCString(), uang.ToCString(), usld.ToCString())

        # default is MM
        scale = 0.001

        if ulen_names.Length() > 0:
            scaleval = ulen_names.Value(1).ToCString().lower()

            # INCH, MM, FT, MI, M, KM, MIL, CM
            # UM, UIN ??

            scales = {
                "millimeter": 0.001,
                "millimetre": 0.001,
                "centimeter": 0.01,
                "centimetre": 0.01,
                "kilometer": 1000.0,
                "kilometre": 1000.0,
                "meter": 1.0,
                "metre": 1.0,
                "inch": 0.0254,
                "foot": 0.3048,
                "mile": 1609.34,
                "mil": 0.0254 * 0.001,
            }

            if scaleval in scales:
                scale = scales[scaleval]
            else:
                print("ERROR: Undefined scale:", scaleval)

            print("Scale from file (meters per unit):", scaleval, scale)

        else:
            print("Using default scale (millimeters)")

        self.scale = scale

        status = step_reader.ReadFile(self.filename)
        assert status == IFSelect_RetDone

        print("DataExchange: Transferring")
        # print("Roots:", step_reader.NbRootsForTransfer())
        transfer_result = step_reader.Transfer(doc)
        if not transfer_result:
            print("Dataexchange transfer FAILED.")
        else:
            print("DataExchange: Transfer done")

        self.doc = doc

    def transfer_simple(self, fname):
        # see stepanalyzer.py for license details
        print("Init simple transfer")

        # Create the application, empty document and shape_tool
        doc = TDocStd_Document("STEP")
        app = XCAFApp_Application_GetApplication()
        app.NewDocument("MDTV-XCAF", doc)

        # Read file and return populated doc
        step_reader = STEPCAFControl_Reader()
        step_reader.SetColorMode(True)
        step_reader.SetLayerMode(True)
        step_reader.SetNameMode(True)
        step_reader.SetMatMode(True)
        status = step_reader.ReadFile(fname)
        if status == IFSelect_RetDone:
            step_reader.Transfer(doc)
        self.scale = 0.001

        self.doc = doc

    def init_reader(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError("%s not found." % filename)

        # self.filename = force_ascii(filename)
        self.filename = filename

        self.transfer_with_units(self.filename)
        # self.transfer_simple(self.filename)

        self.shape_tool = XCAFDoc_DocumentTool.ShapeTool(self.doc.Main())
        self.color_tool = XCAFDoc_DocumentTool.ColorTool(self.doc.Main())

        # material_tool = XCAFDoc_DocumentTool_MaterialTool(doc.Main())
        # layer_tool = XCAFDoc_DocumentTool_LayerTool(doc.Main())

        # use OrderedDict and make sure the order is maintained through the entire pipeline

        self.shape_label = {}
        self.sub_shapes = OrderedDict()

        self.face_colors = {}
        self.face_color_priority = {}
        self.tag_info = {}
        self.skipped_shapes = set([])
        self.import_problems = {"Triangulation": 0, "Undefined normals": 0, "Empty shape": 0}

    def read_file(self, filename):
        """Returns list of tuples (topods_shape, label, color)
        Use OCAF.
        """

        self.init_reader(filename)

        # output_shapes = {}
        # outliers = defaultdict(set)

        def _cprio(lab, shape):
            "Get label color"
            tc, ctype, ok = self.query_color(lab)
            self.face_colors[shape] = tc if ok else None
            if ok:
                return ctype
            else:
                return 0

        def _get_sub_shapes(lab, level, tree, leaf_id):

            # print(" " * (2 * level) + lab.GetLabelName())
            master_leaf = tree.nodes[leaf_id]
            # l_comps = TDF_LabelSequence()
            # self.shape_tool.GetComponents(lab, l_comps)
            if self.shape_tool.IsAssembly(lab):
                # Get transform for pure transform (empty)
                # Empty has eye transform, inherit global from parent

                # empty = tree.add(leaf.index, lab, empty=True)
                # output_shapes[shape] = empty

                # Read contained shapes
                l_c = TDF_LabelSequence()
                self.shape_tool.GetComponents(lab, l_c)
                for i in range(l_c.Length()):
                    label = l_c.Value(i + 1)
                    if self.shape_tool.IsReference(label):
                        label_reference = TDF_Label()
                        self.shape_tool.GetReferredShape(label, label_reference)

                        label_transform = self.label_matrix(label)
                        node = tree.add(master_leaf.index, label_reference)
                        new_leaf = tree.nodes[node.index]
                        new_leaf.local_transform = label_transform
                        new_leaf.global_transform = master_leaf.global_transform @ label_transform

                        _get_sub_shapes(label_reference, level + 1, tree, node.index)
                    else:
                        # TODO: process rest of the data
                        pass

            elif self.shape_tool.IsSimpleShape(lab):
                # TODO: self.shape_label stops being unique when shapes aren't transformed
                shape = self.shape_tool.GetShape(lab)
                master_leaf.set_shape(shape)
                if shape in self.shape_label:
                    # Shape already in
                    return

                self.shape_label[shape] = lab

                self.face_color_priority[shape] = _cprio(lab, shape)

                l_subss = TDF_LabelSequence()
                self.shape_tool.GetSubShapes(lab, l_subss)
                self.sub_shapes[shape] = []
                for i in range(l_subss.Length()):
                    lab_subs = l_subss.Value(i + 1)
                    shape_sub = self.shape_tool.GetShape(lab_subs)
                    self.shape_label[shape_sub] = lab_subs
                    self.sub_shapes[shape].append(shape_sub)
                    self.face_color_priority[shape_sub] = _cprio(lab_subs, shape_sub)
                # Color priority is the same as CAD assistant material tree display
            else:
                print("DataExchange error: Item is neither assembly or a simple shape")

        def _get_shapes():
            # self.shape_tool.UpdateAssemblies()

            labels = TDF_LabelSequence()
            self.shape_tool.GetFreeShapes(labels)

            tree = ShapeTree()
            for i in range(labels.Length()):
                print("DataExchange: Reading shape ({}/{})".format(i + 1, labels.Length()))

                root_item = labels.Value(i + 1)
                node = tree.add(tree.get_root_id(), root_item)
                _get_sub_shapes(root_item, 0, tree, node.index)

            return tree

        tree = _get_shapes()
        self.tree = tree

    def triangulate_face(self, face, tform):
        bt = BRep_Tool()
        location = TopLoc_Location()
        facing = bt.Triangulation(face, location)
        if facing is None:
            # Mesh error, no triangulation found for part
            self.import_problems["Triangulation"] += 1
            return None

        normcalc = poly()
        normcalc.ComputeNormals(facing)

        # nsurf = bt.Surface(face)
        surface = BRepAdaptor_Surface(face)
        prop = BRepLProp_SLProps(surface, 2, gp.Resolution())
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
            # prop = GeomLProp_SLProps(nsurf, fU, fV, 2, gp.Resolution())

            uv = facing.UVNode(t)
            u, v = uv.X(), uv.Y()
            uvs.append((u, v))

            # The edges of UV give invalid normals, hence this
            prop.SetParameters((u - Ucenter) * 0.999 + Ucenter, (v - Vcenter) * 0.999 + Vcenter)

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
            T1, T2, T3 = tri.Value(t).Get()

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

        tri_data = []
        for ti, t in enumerate(tris):
            tri_data.append(trimesh.TriData(t, [norms[i] for i in t], [uvs[i] for i in t], None, None, None, None))

        return trimesh.TriMesh(verts=verts, tris=tri_data)

    def build_trimesh(self, shape, lin_def=0.8, ang_def=0.5, hacks=set([])):
        out_mesh = trimesh.TriMesh()
        out_mesh.matrix = np.eye(4, dtype=np.float32)

        # TODO: this is hack
        if "skip_solids" in hacks and self.explore_partial(shape, TopAbs_SOLID) == 0:
            self.skipped_shapes.add(self.shape_label[shape].GetLabelName())
            return out_mesh

        iter_shapes = [shape] + self.sub_shapes[shape]
        iter_shapes.sort(key=lambda x: x.Checked())

        brt = breptools()

        face_data = OrderedDict()
        batch = 0

        # Iterate over the main shape and its sub shapes
        for shp_i, shp in enumerate(iter_shapes):
            col = self.face_colors[shp]
            if col is not None:
                col_rgb = b_RGB(col)
                col_name = b_colorname(col)
            else:
                col_name = ""

            # Clean all previous triangulations
            brt.Clean(shp)

            # Subshape transforms can be different from the mainshape transform
            ex = TopExp_Explorer(shp, TopAbs_FACE)
            if not ex.More():
                self.import_problems["Empty shape"] += 1
                continue

            brepmesh = BRepMesh_IncrementalMesh(shp, lin_def, False, ang_def, False)
            brepmesh.Perform()
            trf = shp.Location().Transformation()
            # Iterate through faces with TopExp_Explorer
            while ex.More():
                exc = ex.Current()
                face = topods.Face(exc)

                mesh = self.triangulate_face(face, trf)
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

        for fc, b in face_data.items():
            prio, mesh, col_name = b
            if len(mesh.verts) > 0:
                out_mesh.add_mesh(mesh)

        print("[l]", end="", flush=True)

        return out_mesh

    def build_nurbs(self, shape):
        iter_shapes = [shape]
        nbs = []
        for shp_i, shp in enumerate(iter_shapes):
            ex = TopExp_Explorer(shp, TopAbs_FACE)
            if not ex.More():
                self.import_problems["Empty shape"] += 1
                return []

            while ex.More():
                pt = nurbs_parse(topods.Face(ex.Current()))
                nbs.append(pt)
                ex.Next()

        return nbs
