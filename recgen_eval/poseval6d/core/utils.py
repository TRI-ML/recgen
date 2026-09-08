# Utility functions
from collections import OrderedDict, defaultdict

import open3d as o3d
import numpy as np


def symmetry_tfs_from_info(info, rot_angle_discrete=5):
  symmetry_tfs = [np.eye(4)]
  if 'symmetries_discrete' in info:
    tfs = np.array(info['symmetries_discrete']).reshape(-1,4,4)
    tfs[...,:3,3] *= 0.001
    symmetry_tfs = [np.eye(4)]
    symmetry_tfs += list(tfs)
  if 'symmetries_continuous' in info:
    from transformations import euler_matrix
    axis = np.array(info['symmetries_continuous'][0]['axis']).reshape(3)
    offset = info['symmetries_continuous'][0]['offset']
    rxs = [0]
    rys = [0]
    rzs = [0]
    if axis[0]>0:
      rxs = np.arange(0,360,rot_angle_discrete)/180.0*np.pi
    elif axis[1]>0:
      rys = np.arange(0,360,rot_angle_discrete)/180.0*np.pi
    elif axis[2]>0:
      rzs = np.arange(0,360,rot_angle_discrete)/180.0*np.pi
    for rx in rxs:
      for ry in rys:
        for rz in rzs:
          tf = euler_matrix(rx, ry, rz)
          tf[:3,3] = offset
          symmetry_tfs.append(tf)
  if len(symmetry_tfs)==0:
    symmetry_tfs = [np.eye(4)]
  symmetry_tfs = np.array(symmetry_tfs)
  return symmetry_tfs

def make_yaml_dumpable(D):
  if isinstance(D, np.ndarray):
    return D.tolist()
  for d in D:
    if isinstance(D[d], dict) or isinstance(D[d], OrderedDict) or isinstance(D[d], defaultdict):
      D[d] = dict(D[d])
      D[d] = make_yaml_dumpable(D[d])
      continue
    if isinstance(D[d], np.ndarray):
      D[d] = D[d].tolist()
      continue
    if np.issubdtype(type(D[d]), int):
      D[d] = int(D[d])
      continue
    if np.issubdtype(type(D[d]), float):
      D[d] = float(D[d])
      continue
    if np.issubdtype(type(D[d]), str):
      D[d] = str(D[d])
      continue
    if isinstance(D[d], list):
      for i in range(len(D[d])):
        D[d][i] = make_yaml_dumpable(D[d][i])
      continue
  return dict(D)

def points_to_open3d(points, colors: np.ndarray=None, normals: np.ndarray=None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        # Normalize colors to [0, 1] range
        max_value_color_type = np.iinfo(colors.dtype).max
        colors = colors / max_value_color_type
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud

def homogeneous(pts):
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo
