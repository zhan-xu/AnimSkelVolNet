import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['V2V_HG', 'v2v_hg']


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size, input_plane):
        super(Pool3DBlock, self).__init__()
        self.stride_conv = nn.Sequential(
            nn.Conv3d(input_plane, input_plane, kernel_size=pool_size, stride=pool_size, padding=0),
            nn.BatchNorm3d(input_plane),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.stride_conv(x)
        return y


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        # assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class HG(nn.Module):
    def __init__(self, front=False):
        super(HG, self).__init__()
        if front:
            input_channel = 8
        else:
            input_channel = 10
        self.encoder_pool1 = Pool3DBlock(2, input_channel)
        self.encoder_res1 = Res3DBlock(input_channel, 16)
        self.encoder_pool2 = Pool3DBlock(2, 16)
        self.encoder_res2 = Res3DBlock(16, 24)
        self.encoder_pool3 = Pool3DBlock(2, 24)
        self.encoder_res3 = Res3DBlock(24, 36)

        self.mid_res = Res3DBlock(40, 36)

        self.decoder_res3 = Res3DBlock(36, 36)
        self.decoder_upsample3 = Upsample3DBlock(36, 24, 2, 2)
        self.decoder_res2 = Res3DBlock(24, 24)
        self.decoder_upsample2 = Upsample3DBlock(24, 16, 2, 2)
        self.decoder_res1 = Res3DBlock(16, 16)
        self.decoder_upsample1 = Upsample3DBlock(16, 8, 2, 2)

        self.skip_res1 = Res3DBlock(input_channel, 8)
        self.skip_res2 = Res3DBlock(16, 16)
        self.skip_res3 = Res3DBlock(24, 24)

    def forward(self, x, c):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        c = c.repeat(1, 4, 11, 11, 11)
        x = torch.cat((x, c), dim=1)
        x = self.mid_res(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x


class V2V_HG(nn.Module):
    def __init__(self, input_channels, n_stack):
        super(V2V_HG, self).__init__()
        self.input_channels = input_channels
        self.n_stack = n_stack
        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 8, 5),
            Res3DBlock(8, 8)
        )

        self.hg_1 = HG(front=True)

        self.joint_output_1 = nn.Sequential(
            Res3DBlock(8, 4),
            Basic3DBlock(4, 4, 1),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(4, 1, kernel_size=1, stride=1, padding=0)
        )
        self.bone_output_1 = nn.Sequential(
            Res3DBlock(8, 4),
            Basic3DBlock(4, 4, 1),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(4, 1, kernel_size=1, stride=1, padding=0)
        )

        if n_stack > 1:
            self.hg_list = nn.ModuleList([HG(front=False) for i in range(1, n_stack)])
            self.joint_output_list = nn.ModuleList([nn.Sequential(
                Res3DBlock(8, 4), Basic3DBlock(4, 4, 1), nn.Dropout3d(p=0.2),
                nn.Conv3d(4, 1, kernel_size=1, stride=1, padding=0)) for i in range(1, n_stack)])
            self.bone_output_list = nn.ModuleList([nn.Sequential(
                Res3DBlock(8, 4), Basic3DBlock(4, 4, 1), nn.Dropout3d(p=0.2),
                nn.Conv3d(4, 1, kernel_size=1, stride=1, padding=0)) for i in range(1, n_stack)])
        self._initialize_weights()

    def forward(self, x_in, c):
        x = self.front_layers(x_in)
        x_hg_1 = self.hg_1(x, c)
        x_joint_out1 = self.joint_output_1(x_hg_1)
        x_bone_out1 = self.bone_output_1(x_hg_1)

        x_joint_out = [x_joint_out1]
        x_bone_out = [x_bone_out1]

        for i in range(1, self.n_stack):
            x_in = torch.cat((x, x_joint_out1, x_bone_out1), dim=1)
            x_hg = self.hg_list[i-1](x_in, c)
            x_joint = self.joint_output_list[i-1](x_hg)
            x_bone = self.bone_output_list[i-1](x_hg)
            x_joint_out.append(x_joint)
            x_bone_out.append(x_bone)

        return x_joint_out, x_bone_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


def v2v_hg(**kwargs):
    model = V2V_HG(input_channels=kwargs['input_channels'], n_stack=kwargs['n_stack'])
    return model
