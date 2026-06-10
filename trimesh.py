import numpy as np
import bpy
from dataclasses import dataclass


def make_tri_hash(f):
    return frozenset([tuple(f[0]), tuple(f[1]), tuple(f[2])])


@dataclass
class TrianglesData:
    """
    Must stay aligned
    """

    indices = []
    norms = []
    colors = []  # None means no color
    material = []
    material_name = []
    uvs = []
    batch = []

    def __init__(
        self,
        indices=[],
        norms=[],
        colors=[],
        material=[],
        material_name=[],
        uvs=[],
        batch=[],
    ):
        self.indices = indices
        self.norms = norms
        self.colors = colors
        self.material = material
        self.material_name = material_name
        self.uvs = uvs
        self.batch = batch

    def __getitem__(self, index):
        # Could also be a dictionary
        return (
            self.indices[index],
            self.norms[index],
            self.colors[index],
            self.material[index],
            self.material_name[index],
            self.uvs[index],
            self.batch[index],
        )

    def __len__(self):
        return len(self.indices)

    def __setitem__(self, index, value):
        (
            self.indices[index],
            self.norms[index],
            self.colors[index],
            self.material[index],
            self.material_name[index],
            self.uvs[index],
            self.batch[index],
        ) = value

    def append(self, value):
        self.indices.append(value[0]),
        self.norms.append(value[1]),
        self.colors.append(value[2]),
        self.material.append(value[3]),
        self.material_name.append(value[4]),
        self.uvs.append(value[5]),
        self.batch.append(value[6]),

    def extend(self, other, offset):
        self.indices.extend(
            [(tis[0] + offset, tis[1] + offset, tis[2] + offset) for tis in other.indices]
        )
        self.norms.extend(other.norms)
        self.uvs.extend(other.uvs)
        self.colors.extend(other.colors)
        self.material.extend(other.material)
        self.material_name.extend(other.material_name)
        self.batch.extend(other.batch)


class TriMesh:
    """Triangle mesh. Array of triangles of which each item has three pointers
    to locations inside array of verts. Each triangle data is defined through TriData class
    """

    def __init__(self, verts=None, tris=None, matrix=None):

        if verts is None and tris is None:
            self.tris = TrianglesData()
            self.verts = []
            self.matrix = np.empty((3, 4), dtype=np.float32)

        else:
            assert verts is not None

            if matrix is None:
                matrix = np.empty((3, 4), dtype=np.float32)
            assert hasattr(matrix, "shape")
            assert matrix.shape == (3, 4)

            self.matrix = matrix
            self.verts = verts

            if tris is not None:
                self.tris = tris
                # self.tris.batch = self.batch_index
            else:
                self.tris = None

    def check_same_face(self):
        """Check all tris for overlap. Return None if none found"""
        faces = set([])
        # TODO: all matches, not just the first one
        for i, tis in enumerate(self.tris.indices):
            # Filter zero area and existing faces
            locs = [self.verts[i] for i in tis]
            same_loc = locs[0] == locs[1] or locs[1] == locs[2] or locs[2] == locs[0]
            same_face = False
            f_hash = tuple(sorted(locs))
            if f_hash not in faces:
                faces.add(f_hash)
            else:
                same_face = True
            if same_loc or same_face:
                res = {}
                if same_loc:
                    res["same_loc"] = (i, tis, locs)
                if same_face:
                    res["same_face"] = (i, tis, f_hash)
                return res
        return None

    def filter_zero_area(self):
        """Remove all tris with zero area"""
        new_tris = TrianglesData()
        for i, t in enumerate(self.tris):
            tris = self.tris.indices[i]
            locs = (self.verts[tris[0]], self.verts[tris[1]], self.verts[tris[2]])
            same_loc = locs[0] == locs[1] or locs[1] == locs[2] or locs[2] == locs[0]
            if not same_loc:
                new_tris.append(t)
        self.tris = new_tris

    def filter_same_face(self):
        """Remove all duplicate tris"""
        # TODO: might cause color issue
        new_tris = TrianglesData()
        faces = set([])
        for i, t in enumerate(self.tris):
            tris = self.tris.indices[i]
            locs = (self.verts[tris[0]], self.verts[tris[1]], self.verts[tris[2]])
            f_hash = tuple(sorted(locs))

            # add only if hash not already there 
            if f_hash not in faces:
                faces.add(f_hash)
                new_tris.append(t)
        self.tris = new_tris

    def fuse_verts(self):
        """Make verts in identical locations the same, update tris"""

        verts = {}
        tri_map = {}
        new_verts = []
        new_index = 0
        for vi, v in enumerate(self.verts):
            # Find duplicate verts based on location "hash"
            v_hash = tuple(v)
            if v_hash not in verts:
                new_verts.append(v)
                verts[v_hash] = new_index
                tri_map[vi] = new_index
                new_index += 1
            else:
                # Vert already exists in new_verts
                tri_map[vi] = verts[v_hash]

        # Change tri indices to match new fused verts
        new_tris = []
        for tis in self.tris.indices:
            new_tris.append((tri_map[tis[0]], tri_map[tis[1]], tri_map[tis[2]]))

        # Save new data
        self.verts = new_verts
        self.tris.indices = new_tris

    def add_mesh(self, other):
        # assert other is TriMesh
        offset = len(self.verts)
        self.verts.extend(other.verts)
        self.tris.extend(other.tris, offset)

    def colorize(self, col):
        "Fill with color, color can be None"
        # assert len(self.tris) > 0
        self.tris.color = [col] * len(self.tris)

    def set_batch(self, batch):
        "Set all triangle data to batch index"
        # assert len(self.tris) > 0
        self.tris.color = [batch] * len(self.tris)

    def set_material_name(self, name):
        "Set material name for all tris"
        # assert len(self.tris) > 0
        self.tris.material_name = [name] * len(self.tris)

    def fill_empty_color(self):
        "Fill color==None with undef_color (currently pink)"
        # TODO : make undef black and add attributes to tell if face is colored so the shader can know
        undef_color = (1.0, 0.0, 1.0)  # pink
        if len(self.tris) == 0:
            # Empty mesh
            return
        self.tris.color = [c if c else undef_color for c in self.tris.color]

    def add_to_mesh(self, mesh: bpy.types.Mesh):
        mesh.from_pydata(self.verts, [], self.tris.indices)

    def get_loop_colors(self):
        "Return colors in triangle loop creation order"
        return [self.tris.colors[i // 3] for i in range(len(self.tris)) * 3]

    def get_loop_material_names(self):
        "Return material names in triangle loop creation order"
        return [self.tris.material_name[i // 3] for i in range(len(self.tris)) * 3]

    def get_loop_normals(self):
        "Return normals in triangle loop creation order"
        return [self.tris.norms[i // 3] for i in range(len(self.tris)) * 3]

    def get_loop_uvs(self):
        "Return UVs in triangle loop creation order"
        return [self.tris.uvs[i // 3] for i in range(len(self.tris)) * 3]
