#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
This script is for generating training/testing data, which is saved as a h5 file.
"""

import sys
sys.path.append('./')

import numpy as np
import os
import h5py
import json
import scipy.ndimage as ndimage
from tqdm import tqdm

import util.binvox_rw as binvox_rw
from util.vox_utils import Cartesian2Voxcoord
from util.rigging_parser.skel_parser import Skel
from util.rigging_parser.obj_parser import Mesh_obj


def unique_rows(a):
    # remove repeat rows from numpy array
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def draw_jointmap(img, pt, sigma):
    # Draw a 3D gaussian
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma), int(pt[2] - 3*sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1), int(pt[2] + 3*sigma +1)]
    if (ul[0] >= img.shape[0] or ul[1] >= img.shape[1] or ul[2] >= img.shape[2] or
            br[0] < 0 or br[1] < 0 or br[2] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    z = y[..., np.newaxis]
    x0 = y0 = z0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[0]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[1]) - ul[1]
    g_z = max(0, -ul[2]), min(br[2], img.shape[2]) - ul[2]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[0])
    img_y = max(0, ul[1]), min(br[1], img.shape[1])
    img_z = max(0, ul[2]), min(br[2], img.shape[2])

    img[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]] = \
        np.maximum(img[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]],
                   g[g_x[0]:g_x[1], g_y[0]:g_y[1], g_z[0]:g_z[1]])
    return img


def draw_bonemap(heatmap, p_pos, c_pos, output_resulotion):
    # create 3D bone heatmap. Voxels along the bone have value 1, otherwise 0
    c_pos = np.asarray(c_pos)
    ray = c_pos - p_pos
    i_step = np.arange(1, 100)
    unit_step = (ray / 100)[np.newaxis,:]
    unit_step = np.repeat(unit_step, 99, axis=0)
    pos_step = p_pos + unit_step * i_step[:,np.newaxis]
    pos_step = np.round(pos_step).astype(np.uint8)
    pos_step = np.array([p for p in pos_step if np.all(p >= 0) and np.all(p < output_resulotion)])
    if len(pos_step) != 0:
        heatmap[pos_step[:, 0], pos_step[:, 1], pos_step[:, 2]] += 1
    np.clip(heatmap, 0.0, 1.0, out=heatmap)
    return heatmap


def getConditions(fs_filename):
    '''
    Read in our feature size file and return the 5th percentile of all feature size
    Our feature size file contains one bone sample per row.
    The first three numbers are coordinates of the sample. The last number is the feature size.
    Feature size is calculated in cpp, where we shoot rays on a plane perpendicular to the bone.
    For each bone sample, its "feature size" is the median distance to all nearest hits of the rays from it.
    Our feature size file use "new bone" to seperate samples from different bones.
    :param fs_filename: filename of our feature size file.
    :return: 5th percentile of all the feature size, used as contional variable during training and testing.
    '''
    with open(fs_filename, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fs_all = []
    fs_list_bone = []
    for li in lines:
        if li.strip() == 'new bone':
            fs_all.append(fs_list_bone)
            fs_list_bone = []
        else:
            words = li.split()
            fs_list_bone.append(float(words[3]))
    min_fs = []
    for i in fs_all:
        min_fs += i
    min_fs = np.array(min_fs)
    min_5_fs = np.percentile(min_fs, 5)
    return min_5_fs


def center_vox(volumn_input):
    #put the occupied voxels at the center instead of corner
    pos = np.where(volumn_input > 0)
    x_max, x_min = np.max(pos[0]), np.min(pos[0])
    y_max, y_min = np.max(pos[1]), np.min(pos[1])
    z_max, z_min = np.max(pos[2]), np.min(pos[2])
    side_length = volumn_input.shape[0]
    mid_len = int(side_length / 2)
    xr_low = int((x_max - x_min + 1) / 2)
    xr_high = x_max - x_min + 1 - xr_low
    yr_low = int((y_max - y_min + 1) / 2)
    yr_high = y_max - y_min + 1 - yr_low
    zr_low = int((z_max - z_min + 1) / 2)
    zr_high = z_max - z_min + 1 - zr_low
    content = volumn_input[x_min: x_max + 1, y_min: y_max + 1, z_min: z_max + 1]
    volumn_output = np.zeros((volumn_input.shape), dtype=np.bool)
    center_trans = [x_min - mid_len + xr_low, y_min - mid_len + yr_low, z_min - mid_len + zr_low]
    center_trans = list(map(int, center_trans))
    volumn_output[mid_len - xr_low:mid_len + xr_high, mid_len - yr_low:mid_len + yr_high,
    mid_len - zr_low:mid_len + zr_high] = content
    return volumn_output, center_trans


def bin2sdf(input):
    '''
    convert binary voxels into sign distance function field. Negetive for interior. Positive for exterior. Normalized.
    :param input: binary voxels
    :return: SDF representation of voxel.
    '''
    fill_map = np.zeros(input.shape, dtype=np.bool)
    output = np.zeros(input.shape, dtype=np.float16)
    # fill inside
    changing_map = input.copy()
    sdf_in = -1
    while np.sum(fill_map) != np.sum(input):
        changing_map_new = ndimage.binary_erosion(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        output[np.where(changing_map_new!=changing_map)] = sdf_in
        changing_map = changing_map_new.copy()
        sdf_in -= 1
    # fill outside. No need to fill all of them, since during training, outside part will be masked.
    changing_map = input.copy()
    sdf_out = 1
    while np.sum(fill_map) != np.size(input):
        changing_map_new = ndimage.binary_dilation(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        output[np.where(changing_map_new!=changing_map)] = sdf_out
        changing_map = changing_map_new.copy()
        sdf_out += 1
        if sdf_out == -sdf_in:
            break
    # Normalization
    output[np.where(output < 0)] /= (-sdf_in-1)
    output[np.where(output > 0)] /= (sdf_out-1)
    return output


def cal_avg_edge_length(mesh):
    # calculate the average length of all edges of the mesh, which is used as a multiplier when generating vertice density maps.
    points = mesh.v
    edge_index = np.concatenate((mesh.f[:, 0:2], mesh.f[:, 1:], mesh.f[:, [0, 2]]), axis=0)
    edge_index -= 1
    edge_index = unique_rows(edge_index)
    edge_length = np.linalg.norm(points[edge_index[:, 0]] - points[edge_index[:, 1]], axis=1)
    return points, np.mean(edge_length)


def get_surface_vertice(mesh, trans, scale, center_trans, dim_ori=82, r=3, dim_pad=88):
    points, avg_edge = cal_avg_edge_length(mesh)
    res = np.zeros((dim_pad, dim_pad, dim_pad), dtype=np.uint8)
    vc = Cartesian2Voxcoord(points, np.array([trans]), scale, resolution=dim_ori)
    vc = vc - np.array([center_trans]) + r
    for v in vc:
        res[v[0], v[1], v[2]] += 1
    return res, avg_edge


def genDataset_inner(root_folder, model_id, subset, dim_ori=82, r=3, dim_pad=88):
    '''
    generate necessary data for one sample
    :param root_folder: directory with all raw data
    :param model_id: model ID
    :param subset: 'train', 'val' or 'test'
    :param dim_ori: original voxel grid resolution
    :param r: padding added to the original voxel grid
    :param dim_pad: padded voxel grid resolution
    :return: input representation for one sample, including SDF-voxelization, k1, k2 curvature maps, shape diameter, vertice density
    '''
    mesh_file = os.path.join(root_folder, 'obj/{:d}.obj'.format(model_id))
    vox_file = os.path.join(root_folder, 'vox_82/{:d}.binvox'.format(model_id))
    skel_file = os.path.join(root_folder, 'skel/{:d}.txt'.format(model_id))
    fs_file = os.path.join(root_folder, 'fs/{:d}_featuresize.txt'.format(model_id))

    # read original voxels and pad it.
    with open(vox_file, 'rb') as f:
        mesh_vox = binvox_rw.read_as_3d_array(f)
    mesh_vox_padded = np.zeros((mesh_vox.dims[0] + 2 * r, mesh_vox.dims[1] + 2 * r, mesh_vox.dims[2] + 2 * r), dtype=np.float16)
    mesh_vox_padded[r:mesh_vox.dims[0] + r, r:mesh_vox.dims[1] + r, r:mesh_vox.dims[2] + r] = mesh_vox.data
    # put the occupied voxels at the center instead of left-top corner
    mesh_vox_padded, center_trans = center_vox(mesh_vox_padded)
    # convert binary voxels to SDF representation
    mesh_vox_padded = bin2sdf(mesh_vox_padded)

    mesh = Mesh_obj(mesh_file)
    min_5_fs = getConditions(fs_file) # get 5th-percentile control parameter
    skel = Skel(skel_file) # read in ground-truth skeleton
    heatmap_joint = np.zeros((int(mesh_vox.dims[0] + 2 * r), int(mesh_vox.dims[1] + 2 * r),
                              int(mesh_vox.dims[2] + 2 * r)), dtype=np.float16)
    heatmap_bones = np.zeros((int(mesh_vox.dims[0] + 2 * r), int(mesh_vox.dims[1] + 2 * r),
                              int(mesh_vox.dims[2] + 2 * r)), dtype=np.float16)
    # create vertice density heatmap.
    heatmap_verts, avg_edge = get_surface_vertice(mesh, mesh_vox.translate, mesh_vox.scale, center_trans, dim_ori, r, dim_pad)
    # start to create target joint&bone heatmaps. BFS iteration.
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            pos = Cartesian2Voxcoord(np.array(p_node.pos), mesh_vox.translate, mesh_vox.scale, mesh_vox.dims[0])
            pos = (pos[0] - center_trans[0] + r, pos[1] - center_trans[1] + r, pos[2] - center_trans[2] + r)
            pos = np.clip(pos, a_min=0, a_max=dim_pad-1)
            draw_jointmap(heatmap_joint, pos, sigma=0.6)
            next_level += p_node.children
            for c_node in p_node.children:
                ch_pos = Cartesian2Voxcoord(np.array(c_node.pos), mesh_vox.translate, mesh_vox.scale, mesh_vox.dims[0])
                ch_pos = (ch_pos[0] - center_trans[0] + r, ch_pos[1] - center_trans[1] + r, ch_pos[2] - center_trans[2] + r)
                draw_bonemap(heatmap_bones, pos, ch_pos, output_resulotion=dim_pad)
        this_level = next_level

    # read original curvature
    curvature_raw = np.load(os.path.join(root_folder, 'curvature/{:d}_curvature.npy'.format(model_id)))
    curvature_surface = np.zeros((2, dim_pad, dim_pad, dim_pad), dtype=np.float16)
    # read original shape diameter
    sd_raw = np.load(os.path.join(root_folder, 'shape_diameter/{:d}_sd.npy'.format(model_id)))
    sd_surface = np.zeros((1, dim_pad, dim_pad, dim_pad), dtype=np.float16)
    # only preserve values at surface voxels.
    data_bin = (mesh_vox_padded < 0)
    changing_map_new = data_bin.copy()
    changing_map_new = ndimage.binary_erosion(changing_map_new)
    fill_map = (changing_map_new != data_bin)
    coord_v = np.argwhere(fill_map)
    coord_v_trans = coord_v + np.array([center_trans]) - r
    curvature_surface[:, coord_v[:, 0], coord_v[:, 1], coord_v[:, 2]] = \
        curvature_raw[:, coord_v_trans[:, 0], coord_v_trans[:, 1], coord_v_trans[:, 2]]
    sd_surface[:, coord_v[:, 0], coord_v[:, 1], coord_v[:, 2]] = \
        sd_raw[coord_v_trans[:, 0], coord_v_trans[:, 1], coord_v_trans[:, 2]]

    anno = {'name': str(model_id), 'min_5_fs': min_5_fs, 'translate': mesh_vox.translate, 'scale': mesh_vox.scale,
            'subset': subset, 'center_trans': center_trans, 'avg_edge': avg_edge}
    return mesh_vox_padded, heatmap_joint, heatmap_bones, heatmap_verts, curvature_surface, sd_surface, anno


def genDataset(root_folder, dim_ori=82, padding=3, dim_pad=88):
    # read in train/val/test split
    train_id_list = np.loadtxt(os.path.join(root_folder, 'train_final.txt'), dtype=int)
    val_id_list = np.loadtxt(os.path.join(root_folder, 'val_final.txt'), dtype=int)
    test_id_list = np.loadtxt(os.path.join(root_folder, 'test_final.txt'), dtype=int)

    num_train = len(train_id_list)
    num_val = len(val_id_list)
    num_test = len(test_id_list)

    # create sub-datasets
    hf = h5py.File(os.path.join(root_folder, 'model-resource-volumetric.h5'), 'w')
    hf.create_dataset('train_data', (num_train, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('train_vert', (num_train, dim_pad, dim_pad, dim_pad), np.uint8)
    hf.create_dataset('train_curvature', (num_train, 2, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('train_sd', (num_train, 1, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('train_label_joint', (num_train, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('train_label_bone', (num_train, dim_pad, dim_pad, dim_pad), np.float16)

    hf.create_dataset('val_data', (num_val, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('val_vert', (num_val, dim_pad, dim_pad, dim_pad), np.uint8)
    hf.create_dataset('val_curvature', (num_val, 2, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('val_sd', (num_val, 1, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('val_label_joint', (num_val, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('val_label_bone', (num_val, dim_pad, dim_pad, dim_pad), np.float16)

    hf.create_dataset('test_data', (num_test, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('test_vert', (num_test, dim_pad, dim_pad, dim_pad), np.uint8)
    hf.create_dataset('test_curvature', (num_test, 2, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('test_sd', (num_test, 1, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('test_label_joint', (num_test, dim_pad, dim_pad, dim_pad), np.float16)
    hf.create_dataset('test_label_bone', (num_test, dim_pad, dim_pad, dim_pad), np.float16)

    # start to fill all sub-datasets
    anno_all = []
    for train_id in tqdm(range(len(train_id_list))):
        model_id = train_id_list[train_id]
        mesh_vox, heatmap_joint, heatmap_bones, heatmap_verts, curvature, sd, anno = \
            genDataset_inner(root_folder, model_id, subset='train', dim_ori=dim_ori, r=padding, dim_pad=dim_pad)
        hf['train_data'][train_id, :, :, :] = mesh_vox
        hf['train_vert'][train_id, :, :, :] = heatmap_verts
        hf['train_label_joint'][train_id, :, :, :] = heatmap_joint
        hf['train_label_bone'][train_id, :, :, :] = heatmap_bones
        hf['train_curvature'][train_id, ...] = curvature[np.newaxis, ...]
        hf['train_sd'][train_id, ...] = sd[np.newaxis, ...]
        anno_all.append(anno)

    for val_id in tqdm(range(len(val_id_list))):
        model_id = val_id_list[val_id]
        mesh_vox, heatmap_joint, heatmap_bones, heatmap_verts, curvature, sd, anno = \
            genDataset_inner(root_folder, model_id, subset='val', dim_ori=dim_ori, r=padding, dim_pad=dim_pad)
        hf['val_data'][val_id, :, :, :] = mesh_vox
        hf['val_vert'][val_id, :, :, :] = heatmap_verts
        hf['val_label_joint'][val_id, :, :, :] = heatmap_joint
        hf['val_label_bone'][val_id, :, :, :] = heatmap_bones
        hf['val_curvature'][val_id, ...] = curvature[np.newaxis, ...]
        hf['val_sd'][val_id, ...] = sd[np.newaxis, ...]
        anno_all.append(anno)

    for test_id in tqdm(range(len(test_id_list))):
        model_id = test_id_list[test_id]
        mesh_vox, heatmap_joint, heatmap_bones, heatmap_verts, curvature, sd, anno = \
            genDataset_inner(root_folder, model_id, subset='test', dim_ori=dim_ori, r=padding, dim_pad=dim_pad)
        hf['test_data'][test_id, :, :, :] = mesh_vox
        hf['test_vert'][test_id, :, :, :] = heatmap_verts
        hf['test_label_joint'][test_id, :, :, :] = heatmap_joint
        hf['test_label_bone'][test_id, :, :, :] = heatmap_bones
        hf['test_curvature'][test_id, ...] = curvature[np.newaxis, ...]
        hf['test_sd'][test_id, ...] = sd[np.newaxis, ...]
        anno_all.append(anno)

    hf.close()

    # save accompanying information as json
    with open(os.path.join(root_folder, 'model-resource-volumetric.json'), 'w') as outfile:
        json.dump(anno_all, outfile)


if __name__ == '__main__':
    root_folder = 'model_resource_data/' # the directory to put raw data and generated dataset
    genDataset(root_folder, dim_ori=82, padding=3, dim_pad=88)
