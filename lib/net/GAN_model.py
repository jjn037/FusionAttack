import torch
import torch.nn as nn
from torch.nn import init, Parameter
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import random
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from lib.net.pointnet2_msg import Atten_Fusion_Conv, Feature_Gather


def get_norm_layer():
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    return norm_layer


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)

    return x


class Generator_img(nn.Module):
    def __init__(self, num_channels=3, ngf=100):
        super(Generator_img, self).__init__()
        self.model = nn.Sequential(  # input is (nc) x 32 x 32
            nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 3 x 32 x 32
            nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # sequence += [nn.AvgPool2d((12, 40), stride=(12, 40))]
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=padw)]

        # if use_sigmoid:
        #     sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        h = 12
        w = 40
        self.fc = nn.Linear(h * w, 1)
        self.use_sigmoid = use_sigmoid

    def forward(self, input):
        x = self.model(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.use_sigmoid:
            x = F.sigmoid(x)

        return x


def define_Dimg(input_nc=3, ndf=64, use_sigmoid=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer()

    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    return init_net(net, init_type, init_gain, gpu_ids)


class Perturbation(nn.Module):
    def __init__(self, batch_size=2, npoints=16384, std=0.0000001):
        super(Perturbation, self).__init__()
        self.pert = Parameter(torch.FloatTensor(batch_size, npoints, 3), requires_grad=True)
        self.truncated_normal_(self.pert, mean=0, std=std)
        self.num = batch_size

    def truncated_normal_(self, tensor, mean=0, std=0.001):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def forward(self, input):
        if self.training or input.size(0) == self.num:
            return input + self.pert
        else:
            idx = random.sample(range(self.num), input.size(0))
            return input + self.pert[idx]

    def get_norm(self):
        return torch.mean(reduce_sum(self.pert ** 2))


class Generator_pts(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super(Generator_pts, self).__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 3, 1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        # print(l_xyz[0].size(), l_features[0].size())

        feature = self.conv1(l_features[0])
        feature = self.bn1(feature)
        feature = self.drop1(feature)
        feature = self.conv2(feature)
        # print('feature', feature.size())
        adv_pts = feature.transpose(1, 2).contiguous()
        # adv_pts = feature.transpose(1, 2)
        return adv_pts


class Generator_fusimg(nn.Module):
    def __init__(self, num_channels=3, ngf=100):
        super(Generator_fusimg, self).__init__()
        self.model = nn.Sequential(  # input is (nc) x 32 x 32
            nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 3 x 32 x 32
            nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=True),

        )
        self.tanh = nn.Tanh()

    def forward(self, input):
        # temp = self.model(input)
        return self.tanh(self.model(input)), self.model(input)


class Generator_fuspts(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super(Generator_fuspts, self).__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 3, 1)
        IMG_FEATURES_CHANNEL = 12
        self.final_fusion_img_point = Atten_Fusion_Conv(3, 128, 128)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, img_feature=None, xy=None):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]

        size_range = [1280.0, 384.0]
        xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        # print(l_xyz[0].size(), l_features[0].size())

        img_fusion_gather_feature = Feature_Gather(img_feature.detach(), xy)
        # print(l_features[0].size(), img_fusion_gather_feature.size(), xy.size(), img_feature.size())
        # print(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
        l_feature = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)


        feature = self.conv1(l_feature)
        feature = self.bn1(feature)
        feature = self.drop1(feature)
        feature = self.conv2(feature)
        # print('feature', feature.size())
        adv_pts = feature.transpose(1, 2).contiguous()
        return adv_pts