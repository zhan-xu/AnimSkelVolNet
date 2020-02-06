from __future__ import print_function, absolute_import

import numpy as np
import json
import torch
import h5py
import pdb
import scipy.ndimage as ndimage
import torch.utils.data as data
from scipy.ndimage import convolve


class Heatmap3D_sdf(data.Dataset):
    def __init__(self, h5path, json_file, subset, kde, input_feature):
        self.h5path = h5path  # data files
        self.kde = kde
        self.input_feature = input_feature
        with open(json_file, 'r') as train_anno_file:
            annos = json.load(train_anno_file)
        h5data = h5py.File(h5path, "r")
        if subset == 'train':
            self.annos = [anno for anno in annos if anno['subset'] =='train']
            self.data = h5data['train_data']
            self.vert = h5data['train_vert']
            self.joint_label = h5data['train_label_joint']
            self.bone_label = h5data['train_label_bone']
            self.curvature = h5data['train_curvature']
            self.sd = h5data['train_sd']
        elif subset == 'val':
            self.annos = [anno for anno in annos if anno['subset']=='val']
            self.data = h5data['val_data']
            self.vert = h5data['val_vert']
            self.joint_label = h5data['val_label_joint']
            self.bone_label = h5data['val_label_bone']
            self.curvature = h5data['val_curvature']
            self.sd = h5data['val_sd']
        elif subset == 'test':
            self.annos = [anno for anno in annos if anno['subset'] == 'test']
            self.data = h5data['test_data']
            self.vert = h5data['test_vert']
            self.joint_label = h5data['test_label_joint']
            self.bone_label = h5data['test_label_bone']
            self.curvature = h5data['test_curvature']
            self.sd = h5data['test_sd']
        self.struct = np.ones((3, 3, 3)).astype(bool)

    def __getitem__(self, index):
        model = self.data[index].astype(np.float32)
        model = torch.from_numpy(model)
        model = model.unsqueeze(0)

        mask = self.data[index]
        mask = (mask < 0)
        mask = ndimage.binary_dilation(mask, structure=self.struct, iterations=2).astype(np.float32)
        mask = torch.from_numpy(mask)

        target_joint = self.joint_label[index].astype(np.float32)
        target_joint = torch.from_numpy(target_joint)

        target_bone = self.bone_label[index].astype(np.float32)
        target_bone = torch.from_numpy(target_bone)

        if 'vertex_kde' in self.input_feature:
            vert = self.vert[index].astype(np.float32)
            g_vert = self.make_gaussian(self.kde * self.annos[index]['avg_edge'])
            vert = convolve(vert.astype(np.float32), g_vert, mode='constant', cval=0)
            vert = torch.from_numpy(vert)
            model = torch.cat((model, vert.unsqueeze(0)), dim=0)

        if 'curvature' in self.input_feature:
            curvature = self.curvature[index].astype(np.float32)
            curvature = torch.from_numpy(curvature)
            model = torch.cat((model, curvature), dim=0)

        if 'sd' in self.input_feature:
            sd = self.sd[index].astype(np.float32)
            sd = torch.from_numpy(sd)
            model = torch.cat((model, sd), dim=0)

        # Meta info
        meta = {'index': index, 'min_5_fs': self.annos[index]['min_5_fs'], 'name': self.annos[index]['name'],
                'translate': self.annos[index]['translate'], 'scale': self.annos[index]['scale'],
                'center_trans': self.annos[index]['center_trans']}

        return model, mask, target_joint, target_bone, meta

    def __len__(self):
        return len(self.data)

    def make_gaussian(self, sigma):
        size = 6 * sigma + 1
        x0 = y0 = z0 = size // 2
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        z = y[..., np.newaxis]
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))
        return g
