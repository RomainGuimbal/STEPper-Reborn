import numpy as np
from mathutils import Vector


class NurbsPoint:
    def __init__(self, point):
        assert len(point) == 3 or len(point) == 4

        self.point = point
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        if len(point) == 3:
            self.w = 1.0
        else:
            self.w = point[3]

    def location(self):
        return (self.x, self.y, self.z)

    def as_vector(self):
        return Vector((self.x, self.y, self.z, self.w))


class NurbsData:
    def __init__(self, uv_points):
        assert len(uv_points) > 1
        assert isinstance(uv_points[0][0], NurbsPoint)
        self.uv_points = uv_points

        self.u_closed = False
        self.v_closed = False

        self.u_periodic = False
        self.v_periodic = False

        self.u_degree = 2
        self.v_degree = 2
