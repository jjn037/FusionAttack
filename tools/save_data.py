import _init_path
import numpy as np
import os
import argparse

import lib.utils.calibration as calibration
from lib.config import cfg, cfg_from_file, cfg_from_list
from PIL import Image
import scipy.io as sio
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import joblib

parser = argparse.ArgumentParser(description = "arg parser")
parser.add_argument('--cfg_file', type = str, default = 'cfgs/LI_Fusion_with_attention_use_ce_loss.yaml',
                    help = 'specify the config for training')
parser.add_argument('--out_dir', type = str, default = None)
parser.add_argument('--set', dest = 'set_cfgs', default = None, nargs = argparse.REMAINDER,
                    help = 'set extra config keys if needed')
parser.add_argument('--cca_suffix', type = str, default = None)
parser.add_argument('--cca_n', type=int, default=512)
parser.add_argument('--cca_mi', type=int, default=1000)
parser.add_argument('--ridge', action = 'store_true', default = False)
parser.add_argument('--lamda', type=float, default=0.1)
parser.add_argument('--ridge_suffix', type = str, default = None)
args = parser.parse_args()

class save_kitti(object):
    def __init__(self, out_dir, save_choice=False, save_np=False, save_mat=False, n_samples=0, load=False):
        root_dir = os.path.join('../', 'data')
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'training')
        split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', 'train.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')

        self.out_dir = out_dir
        self.choice_dir = os.path.join(self.out_dir, 'choice')
        os.makedirs(self.choice_dir, exist_ok=True)
        self.save_choice = save_choice
        self.save_np = save_np
        self.save_mat = save_mat
        self.n_samples = n_samples
        self.load = load
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.npoints = 16384

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_image_rgb(self, idx, vis=False):
        """
        return img with normalization in rgb mode
        :param idx:
        :return: imback(H,W,3)
        """
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file).convert('RGB')
        im = np.array(im).astype(np.float)
        im = im / 255.0
        im -= self.mean
        im /= self.std
        # print(im.shape)
        # ~[-2,2]
        # im = im[:, :, ::-1]
        # make same size padding with 0
        imback = np.zeros([384, 1280, 3], dtype = np.float)
        imback[:im.shape[0], :im.shape[1], :] = im

        if vis:
            return imback, im.shape[0], im.shape[1]
        else:
            return imback  # (H,W,3) RGB mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype = np.float32).reshape(-1, 4)

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

    def save_all(self):
        if self.load:
            print('loading...')
            img_file = os.path.join(self.out_dir, 'img_all.npy')
            pts_file = os.path.join(self.out_dir, 'pts_all.npy')
            img_all_np = np.load(img_file)
            pts_all_np = np.load(pts_file)
            print('loaded')
        else:
            img_all = []
            pts_all = []
            i = 0
            for sample_id in self.sample_id_list:
                if i >= self.n_samples > 0:
                    break
                if i % 100 == 0:
                    print('processing: %d' % i)
                calib = self.get_calib(sample_id)
                img = self.get_image_rgb(sample_id)
                img_shape = self.get_image_shape(sample_id)
                pts_lidar = self.get_lidar(sample_id)

                # get valid point (projected points should be in image)
                pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
                # pts_intensity = pts_lidar[:, 3]

                pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
                pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

                pts_rect = pts_rect[pts_valid_flag][:, 0:3]

                # pts_intensity = pts_intensity[pts_valid_flag]
                # pts_origin_xy = pts_img[pts_valid_flag]

                if self.save_choice:
                    if self.npoints < len(pts_rect):
                        pts_depth = pts_rect[:, 2]
                        pts_near_flag = pts_depth < 40.0
                        far_idxs_choice = np.where(pts_near_flag == 0)[0]
                        near_idxs = np.where(pts_near_flag == 1)[0]
                        near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice),
                                                            replace=False)

                        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                            if len(far_idxs_choice) > 0 else near_idxs_choice
                        np.random.shuffle(choice)
                    else:
                        choice = np.arange(0, len(pts_rect), dtype=np.int32)
                        if self.npoints > len(pts_rect):
                            extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                            choice = np.concatenate((choice, extra_choice), axis=0)
                        np.random.shuffle(choice)
                    choice_file = os.path.join(self.choice_dir, '%06d.npy' % sample_id)
                    np.save(choice_file, choice)
                else:
                    choice_file = os.path.join(self.choice_dir, '%06d.npy' % sample_id)
                    choice = np.load(choice_file)
                ret_pts_rect = pts_rect[choice, :]

                img_all.append(img.reshape(1, -1))
                pts_all.append(ret_pts_rect.reshape(1, -1))

                i += 1

            print('saving...')
            img_all_np = np.concatenate(img_all)
            pts_all_np = np.concatenate(pts_all)
            if self.save_np:
                img_file = os.path.join(self.out_dir, 'img_%d.npy' % self.n_samples)
                np.save(img_file, img_all_np)
                pts_file = os.path.join(self.out_dir, 'pts_%d.npy' % self.n_samples)
                np.save(pts_file, pts_all_np)
            if self.save_mat:
                img_mat = os.path.join(self.out_dir, 'img_%d.mat' % self.n_samples)
                sio.savemat(img_mat, {'img_all': img_all_np})
                pts_mat = os.path.join(self.out_dir, 'pts_%d.mat' % self.n_samples)
                sio.savemat(pts_mat, {'pts_all': pts_all_np})
            print('saved')

        if args.ridge:
            print('loading pca1...')
            pca1_file = os.path.join(self.out_dir, 'pca_img_0_%s.m' % args.cca_suffix)
            pca1 = joblib.load(pca1_file)
            print('ridge regression...')
            img_pca = pca1.transform(img_all_np)
            e = np.identity(3000)
            r1 = np.dot(img_pca.T, img_pca) + args.lamda * e
            p = np.dot(np.dot(np.linalg.inv(r1), img_pca.T), pts_all_np)
            ridge_file = os.path.join(self.out_dir, 'ridge_%s.npy' % args.ridge_suffix)
            np.save(ridge_file, p)
        else:
            if self.load:
                print('loading pca1...')
                pca1_file = os.path.join(self.out_dir, 'pca_img_0_n128mi1500.m')
                pca1 = joblib.load(pca1_file)
                img_all_np_pca = pca1.transform(img_all_np)
                print('loading pca2...')
                pca2_file = os.path.join(self.out_dir, 'pca_pts_0_n128mi1500.m')
                pca2 = joblib.load(pca2_file)
                pts_all_np_pca = pca2.transform(pts_all_np)
            else:
                print('PCA1...')
                pca1 = PCA(n_components=3000)
                img_all_np_pca = pca1.fit_transform(img_all_np)
                pca_file = os.path.join(self.out_dir, 'pca_img_%d_%s.m' % (self.n_samples, args.cca_suffix))
                joblib.dump(pca1, pca_file)
                print('PCA2...')
                pca2 = PCA(n_components=3000)
                pts_all_np_pca = pca2.fit_transform(pts_all_np)
                pca_file = os.path.join(self.out_dir, 'pca_pts_%d_%s.m' % (self.n_samples, args.cca_suffix))
                joblib.dump(pca2, pca_file)
            print('CCA...')
            cca = CCA(n_components=args.cca_n, max_iter=args.cca_mi)
            cca.fit(img_all_np_pca, pts_all_np_pca)
            cca_file = os.path.join(self.out_dir, 'cca2_%d_%s.m' % (self.n_samples, args.cca_suffix))
            joblib.dump(cca, cca_file)

        # cca = rcca.CCA(kernelcca=False, reg=0., numCC=2)
        # cca.train([img_all_np, pts_all_np])
        # cca_file = os.path.join(self.out_dir, 'rcca_%d.h5' % self.n_samples)
        # cca.save(cca_file)
        print('done')


if __name__ == '__main__':
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert args.out_dir is not None
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = save_kitti(args.out_dir, save_choice=False, save_np=False, save_mat=False, n_samples=0, load=True)
    dataset.save_all()