#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
This script is for generating skeleton based on the predicted joint&bone probability maps.
It is formulated as a minimal spinning tree problem.
A soft non-maximum suppression approach is used for better results.
Refer to our paper for more details.
"""

import os
import glob
import cv2
import sys
import numpy as np
from scipy.ndimage import correlate as correlate
from scipy.ndimage import binary_erosion as binary_erosion

from util.tree_utils import TreeNode
from util.rigging_parser.obj_parser import Mesh_obj
from util.rigging_parser.skel_parser import Skel
from util.open3d_utils import show_obj_skel


def nms_soft(heatmap, th_conf=0.1, size=15, sigma=6):
    '''
    soft nms algorithm proposed in "Soft-nms - improving object detection with one line of code"
    :param heatmap: predicted joint heatmaps
    :param th_conf: lowest threshold to stop
    :param size: gaussian kernel size
    :param sigma: gaussian kernel sigma
    :return: heatmap after soft-nms
    '''
    heatmap[heatmap < th_conf] = 0

    # we find all local maximums by looking into the gradient of each voxel.
    # Local maximums should have higher value to all its 26 neighbors.
    # so we apply 26 filters (each of them is 3*3*3) to compute the discrete gradient w.r.t. all directions/neighbors.
    kernels = np.zeros((26, 3, 3, 3))
    kernels[:, 1, 1, 1] = 1.0
    gradient_map = np.zeros((26, heatmap.shape[0], heatmap.shape[1], heatmap.shape[2]), np.float32)
    # create 26 kernels to compute gradient
    for c in range(26):
        if c < 13:
            i = c // 9
            j = (c - i * 9) // 3
            k = (c - i * 9) % 3
            kernels[c, i, j, k] = -1.0
        else:
            i = (c + 1) // 9
            j = (c + 1 - i * 9) // 3
            k = (c + 1 - i * 9) % 3
            kernels[c, i, j, k] = -1.0
        gradient_map[c, ...] = correlate(heatmap, kernels[c, ...], mode='constant', cval=0.0)
    gradient_map[gradient_map >= 0] = 0
    gradient_map[gradient_map < 0] = -1
    # Sum up all 26 filters. Local maximums will have 26 zeros. Otherwise summation will be negetive.
    gradient_map = np.sum(gradient_map, axis=0)

    maximum_map = np.logical_and((gradient_map == 0), (heatmap > 0))  # local maximum map
    heatmap = np.multiply(heatmap, maximum_map.astype(np.float32))  # pick all local maximum by masking

    # create gaussian kernel for soft-nms
    radius = size//2
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    z = y[..., np.newaxis]
    x0 = y0 = z0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    # This is actually 1 - gaussian
    g = 1 - np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))
    result = np.zeros((heatmap.shape), dtype=np.float32)
    # for symmetric reason, we only do nms to the left half.
    # Note we add 1 more column, because we don't want to ruin the middle part of heat map
    heatmap_half = heatmap[0:int(heatmap.shape[0] / 2) + 1, ...]

    while np.any(heatmap_half >= th_conf):
        idx_list = np.argwhere(heatmap_half == np.amax(heatmap_half))
        if len(idx_list) > 1:
            # first merge adjacent voxels
            idx_list = idx_list.tolist()
            # For adjacent voxels, we only preserve the one closest to the middle plane.
            # So here we sort according to their distance to the middle plane.
            idx_list.sort(key=lambda x: abs(x[0] - 43.5))
            idx_list_merge = []  # list to store joints after adjacent merging.
            for idx in idx_list:
                preserve_flag = True
                # check if any adjacent voxel has already preserved.
                for prefix in idx_list_merge:
                    if abs(prefix[0] - idx[0]) <= 1 and abs(prefix[1] - idx[1]) <= 1 and abs(prefix[2] - idx[2]) <= 1:
                        preserve_flag = False
                        break
                if preserve_flag: # if no neighbor has been preserved, we add this voxel to the preserved list.
                    idx_list_merge.append(idx)
            idx_list = np.array(idx_list_merge)
        # decay map is used to store how much we should reduce the heatmap probability
        decay_map = np.ones((heatmap_half.shape), dtype=np.float32)
        for idx in idx_list:
            # set the local maximum position to the original probability in the result heatmap
            result[idx[0], idx[1], idx[2]] = heatmap_half[idx[0], idx[1], idx[2]]
            heatmap_half[idx[0], idx[1], idx[2]] = 0
            ul = [int(idx[0] - radius), int(idx[1] - radius), int(idx[2] - radius)]
            br = [int(idx[0] + radius + 1), int(idx[1] + radius + 1), int(idx[2] + radius + 1)]
            g_x = max(0, -ul[0]), min(br[0], heatmap_half.shape[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_half.shape[1]) - ul[1]
            g_z = max(0, -ul[2]), min(br[2], heatmap_half.shape[2]) - ul[2]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_half.shape[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_half.shape[1])
            img_z = max(0, ul[2]), min(br[2], heatmap_half.shape[2])
            decay_map[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]] = \
                np.multiply(decay_map[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]],
                            g[g_x[0]:g_x[1], g_y[0]:g_y[1], g_z[0]:g_z[1]])
        heatmap_half = np.multiply(heatmap_half, decay_map)
        heatmap_half[heatmap_half < th_conf] = 0

    return result


def load_ts(ts_filename):
    trans = np.zeros(3)
    center_trans = np.zeros(3, dtype=np.int8)
    with open(ts_filename, 'r') as fts:
        line1 = fts.readline().strip().split()
        center_trans[0], center_trans[1], center_trans[2] = int(line1[0]), int(line1[1]), int(line1[2])
        line2 = fts.readline().strip().split()
        trans[0], trans[1], trans[2] = float(line2[0]), float(line2[1]), float(line2[2])
        line3 = fts.readline().strip()
        scl = float(line3)
    return trans, scl, center_trans


def loadSkel_recur(p_node, parent_id, joint_pos, parent):
    for i in range(len(parent)):
        if parent[i] == parent_id:
            ch_node = TreeNode('joint_{}'.format(i), tuple(joint_pos[i]))
            p_node.children.append(ch_node)
            ch_node.parent = p_node
            loadSkel_recur(ch_node, i, joint_pos, parent)


def getInitId(joint_pos):
    '''
    Root joint is chosen as the lowest joint near the middle symmetric plane
    :param joint_pos: all joint positions
    :return: root joint ID
    '''
    sorted_id = np.argsort(joint_pos[:,1])
    for i in range(len(sorted_id)):
        id = sorted_id[i]
        if joint_pos[id, 0] < 0.2:
            continue
        if abs(joint_pos[id, 0]) < 2e-2:
            return id
    return np.argsort(abs(joint_pos[:,0]))[0]


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def minKey(key, mstSet, nV):
    # Initilaize min value
    min = sys.maxsize

    for v in range(nV):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v

    return min_index


def flip(pred_joint, pred_bone, trans, center_trans, scl, input_dim=88, r=3):
    '''
    Enforcing predicted heatmap symmetric by reflecting left half to the right
    The symmetric voxel positions are computed by converting to euclidean space coordinates, finding symmetric positions and converting back.
    This is because voxel space has low resolution and accuracy. Euclidean space is more accurate.
    :param pred_joint: predicted joint heatmap
    :param pred_bone: predicted bone heatmap
    :param trans: translation vector from voxel space to euclidean space
    :param center_trans: translation vector from centered volume to original volume
    :param scl: scale from voxel space to euclidean space
    :param input_dim: voxel grid dimension
    :param r: voxel grid padding
    :return: symmetric heatmaps
    '''
    ori_dim = input_dim - 2 * r  # original voxel grid dimension without padding

    grid_x, grid_y, grid_z = np.meshgrid(np.arange(input_dim), np.arange(input_dim), np.arange(input_dim))
    grid_coord = np.concatenate((grid_y.flatten()[:, None], grid_x.flatten()[:, None], grid_z.flatten()[:, None]), axis=1)
    grid_coord = grid_coord - r + center_trans
    # grid_coord (88^3 * 3) stores correpsonding euclidean coordinates for all voxels in voxel grid
    grid_coord = grid_coord / np.array([ori_dim, ori_dim, ori_dim]) * scl + trans

    # for joint heatmap
    if pred_joint is not None:
        # reflect the left-half heatmap values to the right-half
        # flatten predicted joint map to (88^3, ), with the same order as grid_coord
        val_pred_joint = pred_joint.flatten()
        val_left = np.copy(val_pred_joint[np.logical_and(grid_coord[:, 0] < -2e-2, val_pred_joint > 0)])
        grid_coord_left = np.copy(grid_coord[np.logical_and(grid_coord[:, 0] < -2e-2, val_pred_joint > 0), :])
        grid_coord_right_ = grid_coord_left.copy()
        grid_coord_right_[:, 0] = -grid_coord_right_[:, 0]

        # transform euclidean coodinates back into voxel space.
        vc = np.round((grid_coord_right_ - trans) / scl * ori_dim - center_trans + r)  #
        vc = vc.astype(int)
        vc = np.clip(vc, 0, input_dim-1)
        pred_joint[vc[:,0], vc[:,1], vc[:,2]] = (pred_joint[vc[:,0], vc[:,1], vc[:,2]] + val_left) # add left to right

        # do the same for the right half, reflecting them into the left half.
        val_pred_joint = pred_joint.flatten()
        val_right = np.copy(val_pred_joint[np.logical_and(grid_coord[:, 0] > 2e-2, val_pred_joint > 0)])
        grid_coord_right = np.copy(grid_coord[np.logical_and(grid_coord[:, 0] > 2e-2, val_pred_joint > 0), :])
        grid_coord_left_ = grid_coord_right.copy()
        grid_coord_left_[:, 0] = -grid_coord_left_[:, 0]

        vc = np.round((grid_coord_left_ - trans) / scl * ori_dim - center_trans + r)
        vc = vc.astype(int)
        vc = np.clip(vc, 0, input_dim - 1)
        pred_joint[vc[:, 0], vc[:, 1], vc[:, 2]] = val_right  # don't add values here. Just copy. Otherwise it will unsymmetric again!

    # for bone heatmap
    if pred_bone is not None:
        val_pred_bone = pred_bone.flatten()
        # reflect left-half voxels to right half
        val_left = np.copy(val_pred_bone[np.logical_and(grid_coord[:, 0] < -2e-2, val_pred_bone > 0)])
        grid_coord_left = np.copy(grid_coord[np.logical_and(grid_coord[:, 0] < -2e-2, val_pred_bone > 0), :])
        grid_coord_right_ = grid_coord_left.copy()
        grid_coord_right_[:, 0] = -grid_coord_right_[:, 0]

        vc = np.round((grid_coord_right_ - trans) / scl * ori_dim - center_trans + r)
        vc = vc.astype(int)
        vc = np.clip(vc, 0, input_dim - 1)
        pred_bone[vc[:, 0], vc[:, 1], vc[:, 2]] = (pred_bone[vc[:, 0], vc[:, 1], vc[:, 2]] + val_left) / 2

        val_pred_bone = pred_bone.flatten()
        # reflect right-half voxels to left half
        val_right = np.copy(val_pred_bone[np.logical_and(grid_coord[:, 0] > 2e-2, val_pred_bone > 0)])
        grid_coord_right = np.copy(grid_coord[np.logical_and(grid_coord[:, 0] > 2e-2, val_pred_bone > 0), :])
        grid_coord_left_ = grid_coord_right.copy()
        grid_coord_left_[:, 0] = -grid_coord_left_[:, 0]

        vc = np.round((grid_coord_left_ - trans) / scl * ori_dim - center_trans + r)
        vc = vc.astype(int)
        vc = np.clip(vc, 0, input_dim - 1)
        pred_bone[vc[:, 0], vc[:, 1], vc[:, 2]] = val_right

    return pred_joint, pred_bone


def primMST(graph, init_id):
    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    for cout in range(nV):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)

        # Put the minimum distance vertex in
        # the shortest path tree
        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u,v] > 0 and mstSet[v] == False and key[v] > graph[u,v]:
                key[v] = graph[u,v]
                parent[v] = u

    return parent, key


def primMST_symmetry(graph, init_id, joints):
    '''
    My revised prim algorithm to find a MST as symmetric as possible.
    The function is sort of messy but...
    The basic idea is if a bone on the left is picked, its counterpart on the right should also be picked
    :param graph: cost matrix (N*N)
    :param init_id: joint ID to be first picked
    :param joints: joint position (N*3)
    :return:
    '''
    joint_mapping = {}
    # this is trick. Since we already reflect the joints, "joints" have the order as left_joints->middle_joints->right_joints
    # so we can find correspondence by simply following the original order after splitting three parts.
    left_joint_ids = np.argwhere(joints[:, 0] < -2e-2).squeeze(1).tolist()
    middle_joint_ids = np.argwhere(np.abs(joints[:, 0]) <= 2e-2).squeeze(1).tolist()
    right_joint_ids = np.argwhere(joints[:, 0] > 2e-2).squeeze(1).tolist()
    for i in range(len(left_joint_ids)):
        joint_mapping[left_joint_ids[i]] = right_joint_ids[i]
    for i in range(len(right_joint_ids)):
        joint_mapping[right_joint_ids[i]] = left_joint_ids[i]

    if init_id not in middle_joint_ids:
        #find nearest joint in the middle to be root
        if len(middle_joint_ids) > 0:
            nearest_id = np.argmin(np.linalg.norm(joints[middle_joint_ids, :] - joints[init_id, :][np.newaxis, :], axis=1))
            init_id = middle_joint_ids[nearest_id]

    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    while not all(mstSet):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)
        # left cases
        if u in left_joint_ids and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in left_joint_ids and parent[u] in left_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
                if mstSet[parent[u2]] is False:
                    mstSet[parent[u2]] = True
                    key[parent[u2]] = graph[parent[u2], parent[parent[u2]]]

        elif u in middle_joint_ids and parent[u] in left_joint_ids:
            # form loop in the tree, but we can do nothing
            u2 = None
        # right cases
        elif u in right_joint_ids and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in right_joint_ids and parent[u] in right_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
                if mstSet[parent[u2]] is False:
                    mstSet[parent[u2]] = True
                    key[parent[u2]] = graph[parent[u2], parent[parent[u2]]]
        elif u in middle_joint_ids and parent[u] in right_joint_ids:
            # form loop in the tree, but we can do nothing
            u2 = None
        # middle case
        else:
            u2 = None

        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u,v] > 0 and mstSet[v] == False and key[v] > graph[u,v]:
                key[v] = graph[u,v]
                parent[v] = u
            if u2 is not None and graph[u2,v] > 0 and mstSet[v] == False and key[v] > graph[u2,v]:
                key[v] = graph[u2, v]
                parent[v] = u2

    return parent, key


def getMSTcost(joint_pos, joint_pos_cartesian, joint_pred, bone_pred, volume):
    n_joint = len(joint_pos)
    cost_matrix = np.zeros((n_joint, n_joint), dtype=np.float32)
    # fill upper triangular matrix
    for r in range(n_joint):
        for c in range(r + 1, n_joint):
            pos_start = joint_pos[r]
            pos_end = joint_pos[c]
            num_step = np.round((np.linalg.norm(pos_end - pos_start)) / 1.0).astype(int)
            pos_sample = pos_start[np.newaxis, :] + (pos_end - pos_start)[np.newaxis, :] * np.linspace(0, 1, num_step)[:, np.newaxis]
            pos_sample = np.round(pos_sample).astype(int)
            pos_sample = [tuple(row) for row in pos_sample]
            pos_sample = unique_rows(pos_sample)

            cost_matrix[r, c] = -np.log(bone_pred[pos_sample[:, 0], pos_sample[:, 1], pos_sample[:, 2]] + 1e-8).sum()
            cost_matrix[r, c] += 500 * np.sum(volume[pos_sample[:, 0], pos_sample[:, 1], pos_sample[:, 2]] == 0)

    cost_matrix = cost_matrix + cost_matrix.transpose()
    init_id = getInitId(joint_pos_cartesian)
    parent, key = primMST_symmetry(cost_matrix, init_id, joint_pos_cartesian)
    return joint_pos_cartesian, cost_matrix, parent, key


def mst_generate(res_folder, best_thred, sigma, size, visualize=True, out_folder='results/mst_volNet/',
                 mesh_folder='model_resource_data/obj_fixed/'):
    '''
    Generate skeleton as a MST problem.
    :param res_folder: folde that contains predicted joints, bones, voxelized input and
                       transformation information between vox-space coordinates and original (cartesian) coordinates.
    :param best_thred: minimal threshold to NMS
    :param sigma: sigma for soft NMS
    :param size: gaussian kernel size for soft NMS
    :param visualize: visualize result or not with Open3D library
    :param out_folder: folder to output final results
    '''
    joint_pred_list = glob.glob(res_folder + 'joint_pred_*.npy')
    for i in range(len(joint_pred_list)):
        joint_pred_file = joint_pred_list[i]
        #joint_pred_file = res_folder + 'joint_pred_6178.npy'
        model_id = joint_pred_file.split('_')[-1][:-4]
        print(model_id)
        joint_pred = np.load(joint_pred_file)
        bone_pred = np.load(joint_pred_file.replace('joint_', 'bone_'))
        input = np.load(joint_pred_file.replace('joint_pred_', 'input_')).astype(np.float32)
        erode_input = binary_erosion(input, structure=None, iterations=1).astype(np.float32)
        mesh = Mesh_obj(os.path.join(mesh_folder, '{}.obj'.format(model_id)))

        joint_pred = np.clip(joint_pred, 0.0, 1.0)
        bone_pred = np.clip(bone_pred, 0.0, 1.0)
        joint_pred = joint_pred * erode_input
        bone_pred = bone_pred * erode_input
        ts_filename = joint_pred_file.replace('joint_pred_', 'ts_').replace('.npy', '.txt')
        trans, scl, center_trans = load_ts(ts_filename)

        joint_pred, bone_pred = flip(joint_pred, bone_pred, trans, center_trans, scl)
        joint_pred = nms_soft(joint_pred, best_thred, size=size, sigma=sigma)
        joint_pred, _ = flip(joint_pred, None, trans, center_trans, scl)

        joint_pos = np.argwhere(joint_pred > 1e-10)
        joint_pos_cartesian = joint_pos - 3 + center_trans
        joint_pos_cartesian = joint_pos_cartesian / 82 * scl + trans
        #contain_flag = filter_joints_between_legs(joint_pos_cartesian, mesh.v)
        #joint_pos_cartesian = joint_pos_cartesian[contain_flag]
        #joint_pos = joint_pos[contain_flag]

        # make extracted joints symmetric
        joint_pos_cartesian_left = joint_pos_cartesian[joint_pos_cartesian[:, 0] < -2e-2]
        joint_pos_cartesian_middle = joint_pos_cartesian[np.abs(joint_pos_cartesian[:, 0]) < 2e-2]
        joint_pos_cartesian_right = joint_pos_cartesian_left * np.array([[-1, 1, 1]])
        joint_pos_cartesian = np.concatenate((joint_pos_cartesian_left, joint_pos_cartesian_middle, joint_pos_cartesian_right), axis=0)
        joint_pos = np.round((joint_pos_cartesian - trans) / scl * 82 - center_trans + 3).astype(int)

        if len(joint_pos) in [0, 1, 2]:
            print('too few joints extracted. Try to reduce the threshold.')
            continue
        if len(joint_pos) > 100:
            print('too many joints extracted. Try to increase the threshold.')
            continue
        joint_pos_cartesian, cost_matrix, parent, key = getMSTcost(joint_pos, joint_pos_cartesian, joint_pred, bone_pred, input)

        skel = Skel()
        for i in range(len(parent)):
            if parent[i] == -1:
                skel.root = TreeNode('joint_{}'.format(i), tuple(joint_pos_cartesian[i]))
                break
        loadSkel_recur(skel.root, i, joint_pos_cartesian, parent)

        if joint_pos_cartesian is not None:
            if visualize:
                img = show_obj_skel(mesh, skel.root)
                cv2.imwrite(os.path.join(out_folder, 'mst_{0}.jpg').format(model_id), img[:, 300:-300, ::-1])
        if parent:
            skel.save(os.path.join(out_folder, 'mst_{0}.txt').format(model_id))


if __name__ == '__main__':
    mesh_folder = 'model_resource_data/obj/'
    folder_name = 'volNet/'
    out_folder = 'results/mst_{0}'.format(folder_name)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    best_thred = 0.014
    print(folder_name, best_thred)
    mst_generate('results/{0}'.format(folder_name),
                 best_thred=best_thred, sigma=6.5, size=17, visualize=True, out_folder=out_folder,
                 mesh_folder=mesh_folder)
