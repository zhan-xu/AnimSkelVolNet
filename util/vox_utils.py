from __future__ import absolute_import
import numpy as np


def Cartesian2Voxcoord(v, translate, scale, resolution=82):
    vc = (v - translate) / scale * resolution
    vc = np.round(vc).astype(int)
    return vc[0], vc[1], vc[2]


def Voxcoord2Cartesian(vc, translate, scale, resolution=82):
    v = vc / resolution * scale + translate
    return v[0], v[1], v[2]
