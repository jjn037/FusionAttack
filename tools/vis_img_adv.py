# Created by silver at 2019/9/16 15:21
# Email: xiwuchencn[at]gmail[dot]com
import _init_path
import os
import numpy as np
import pickle
import torch
from torch.nn.functional import grid_sample

import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.datasets.kitti_dataset import KittiDataset
import argparse

from lib.datasets.kitti_rcnn_dataset import interpolate_img_by_xy

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type = str, default = './log/data_adv')
parser.add_argument('--class_name', type = str, default = 'Car')
parser.add_argument('--split', type = str, default = 'val')
parser.add_argument("--adv_ckpt_dir", type = str, default = None)
parser.add_argument("--adv_iter", type = int, default = 100)
parser.add_argument('--noise', action = 'store_true', default = False)
parser.add_argument('--fusion', action = 'store_true', default = False)
args = parser.parse_args()

# import cv2
from lib.config import cfg
from lib.net.GAN_model import Generator_img, Generator_fusimg
from lib.net.train_functions import reduce_sum
from PIL import Image


class GTDatabaseGenerator(KittiDataset):
    def __init__(self, root_dir, split = 'val', classes = args.class_name):
        super().__init__(root_dir, split = split)
        self.gt_database = None
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes
        # self.velodyne_rgb_dir = os.path.join(root_dir, 'KITTI/object/training/velodyne_rgb')
        # # if not os.path.exists(self.velodyne_rgb_dir):
        # os.makedirs(self.velodyne_rgb_dir, exist_ok = True)
        if args.fusion:
            self.generator = Generator_fusimg(num_channels=3, ngf=100)
        else:
            self.generator = Generator_img(num_channels=3, ngf=100)
        self.generator.cuda()
        print("==> Loading generator")
        aimg_ckpt = os.path.join(args.adv_ckpt_dir, 'checkpoint_Gimg_iter_%d.pth' % args.adv_iter)
        checkpoint = torch.load(aimg_ckpt)
        self.generator.load_state_dict(checkpoint['model_state'])
        self.generator.eval()
        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.clamp_max = (1. - img_mean) / img_std
        self.clamp_min = - img_mean / img_std

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def vis_img(self, sample_id):
        gt_database = []
        sample_id = int(sample_id)
        print('process gt sample (id=%06d)' % sample_id)

        # (H,W,3)
        img, h, w = self.get_image_rgb_with_normal(sample_id, vis=True)
        if args.noise:
            img = img + args.pert
            img = torch.from_numpy(img).unsqueeze(0).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
            img_r = torch.zeros_like(img).cuda()
            for j in range(3):
                img_r[:, j, :, :] = torch.clamp(img[:, j, :, :], min=self.clamp_min[j], max=self.clamp_max[j])

            pert_dist = np.sum(args.pert ** 2)
            pert_dist_r = torch.mean(reduce_sum((img_r - img) ** 2))
            img_gen = img_r.permute((0, 2, 3, 1)).squeeze(0).cpu().numpy()
        else:
            img_ori = torch.from_numpy(img).unsqueeze(0).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
            if args.fusion:
                img_pert, _ = self.generator(img_ori)
            else:
                img_pert = self.generator(img_ori)
            pert_dist = torch.mean(reduce_sum(img_pert ** 2))
            adv_img = img_ori + img_pert
            adv_img_r = torch.zeros_like(img_ori).cuda()
            for j in range(3):
                adv_img_r[:, j, :, :] = torch.clamp(adv_img[:, j, :, :], min=self.clamp_min[j], max=self.clamp_max[j])
            pert_dist_r = torch.mean(reduce_sum((adv_img_r - img_ori) ** 2))
            img_gen = adv_img_r.permute((0, 2, 3, 1)).squeeze(0).cpu().numpy()
        img_gen = ((img_gen * self.std + self.mean) * 255)

        img_gen = img_gen.astype(np.uint8)
        img_gen = img_gen[:h, :w, :]
        img_save = Image.fromarray(img_gen)
        img_file = os.path.join(args.save_dir, '%06d_adv.png' % sample_id)
        img_save.save(img_file)

        print('pert img dist: %f' % pert_dist)
        print('pert img dist refined: %f' % pert_dist_r)

        input('Pause: ')


if __name__ == '__main__':
    dataset = GTDatabaseGenerator(root_dir = '../data/', split = args.split)
    os.makedirs(args.save_dir, exist_ok = True)
    if args.noise:
        args.pert = np.random.normal(0.0, 0.2, size=(384, 1280, 3))
        pert_file = os.path.join(args.save_dir, 'pert.npy')
        np.save(pert_file, args.pert)

    with torch.no_grad():
        while True:
            idx = input('sample id:')
            if idx == '':
                break
            dataset.vis_img(idx)

    # gt_database = pickle.load(open('gt_database/train_gt_database.pkl', 'rb'))
    # print(gt_database.__len__())
    # import pdb
    # pdb.set_trace()
