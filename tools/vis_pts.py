import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.net.point_rcnn import PointRCNN
from lib.net.GAN_model import Perturbation, Generator_pts, Generator_fusimg, Generator_fuspts
from lib.net.train_functions import reduce_sum
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
import tools.train_utils.train_utils as train_utils

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import logging
import re

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description = "arg parser")
parser.add_argument('--cfg_file', type = str, default = 'cfgs/default.yml', help = 'specify the config for evaluation')
parser.add_argument("--eval_mode", type = str, default = 'rpn', required = True, help = "specify the evaluation mode")

parser.add_argument('--eval_all', action = 'store_true', default = False, help = 'whether to evaluate all checkpoints')
parser.add_argument('--test', action = 'store_true', default = False, help = 'evaluate without ground truth')
parser.add_argument("--ckpt", type = str, default = None, help = "specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type = str, default = None,
                    help = "specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type = str, default = None,
                    help = "specify the checkpoint of rcnn if trained separated")
parser.add_argument("--afus_ckpt_dir", type = str, default = None)
parser.add_argument("--afus_epoch", type = int, default = 1)
parser.add_argument("--afus_iter", type = int, default = 100)
parser.add_argument('--gen_pert', action = 'store_true', default = True)

parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size for evaluation')
parser.add_argument('--workers', type = int, default = 4, help = 'number of workers for dataloader')
parser.add_argument("--extra_tag", type = str, default = 'default', help = "extra tag for multiple evaluation")
parser.add_argument('--output_dir', type = str, default = None, help = 'specify an output directory if needed')
parser.add_argument("--ckpt_dir", type = str, default = None,
                    help = "specify a ckpt directory to be evaluated if needed")

parser.add_argument('--save_result', action = 'store_true', default = False, help = 'save evaluation results to files')
parser.add_argument('--save_rpn_feature', action = 'store_true', default = False,
                    help = 'save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action = 'store_true', default = True,
                    help = 'sample to the same number of points')
parser.add_argument('--start_epoch', default = 0, type = int, help = 'ignore the checkpoint smaller than this epoch')
parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
parser.add_argument("--rcnn_eval_roi_dir", type = str, default = None,
                    help = 'specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type = str, default = None,
                    help = 'specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest = 'set_cfgs', default = None, nargs = argparse.REMAINDER,
                    help = 'set extra config keys if needed')

parser.add_argument('--model_type', type = str, default = 'base', help = 'model type')
parser.add_argument('--fusion', action = 'store_true', default = False)
parser.add_argument('--delta', action = 'store_true', default = False)

args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format, filename = log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def eval_one_epoch_joint(generator_pts, dataloader, result_dir, logger, generator_img=None):
    np.random.seed(666)

    # logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    generator_pts.eval()
    if generator_img is not None:
        generator_img.eval()

    cnt = 0
    cur_dist_pts = 0

    for data in dataloader:
        # input('Pause')
        cnt += 1
        sample_id, pts_rect, pts_features, pts_input = \
            data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
        if sample_id == 6:
            inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()

            input_data = {}
            if generator_img is not None:
                pts_origin_xy, img = data['pts_origin_xy'], data['img']
                pts_origin_xy = torch.from_numpy(pts_origin_xy).cuda(non_blocking=True).float()
                img = torch.from_numpy(img).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
                input_data['pts_origin_xy'] = pts_origin_xy
                # input_data['img'] = img
                _, img_pert_feature = generator_img(img)
            if args.gen_pert and not args.delta:
                if generator_img is not None:
                    pts_pert = generator_pts(inputs, img_pert_feature, pts_origin_xy)
                else:
                    pts_pert = generator_pts(inputs)
                input_data['pts_input'] = inputs + pts_pert
                cur_dist_pts = torch.mean(reduce_sum(pts_pert ** 2))
            else:
                input_data['pts_input'] = generator_pts(inputs)
                cur_dist_pts = torch.mean(reduce_sum((input_data['pts_input'] - inputs) ** 2))
            adv_file = os.path.join(result_dir, 'pts_adv.npy')
            np.save(adv_file, input_data['pts_input'].cpu().detach().numpy())
            logger.info('pert pts dist: %f' % cur_dist_pts)
            break

    logger.info('result is saved to: %s' % result_dir)
    return 1


def load_ckpt_based_on_args(generator_pts, logger, generator_img=None):
    # if args.ckpt is not None:
    #     train_utils.load_checkpoint(model, filename = args.ckpt, logger = logger)
    if args.afus_ckpt_dir is not None:
        if args.delta:
            logger.info("==> Loading Ppts")
            apts_ckpt = os.path.join(args.afus_ckpt_dir, 'checkpoint_Ppts_iter_%d.pth' % args.afus_iter)
            checkpoint = torch.load(apts_ckpt)
            generator_pts.load_state_dict(checkpoint['model_state'])
        else:
            logger.info("==> Loading Gpts")
            apts_ckpt = os.path.join(args.afus_ckpt_dir, 'checkpoint_Ppts_iter_%d.pth' % args.afus_iter)
            checkpoint = torch.load(apts_ckpt)
            generator_pts.load_state_dict(checkpoint['model_state'])
        if generator_img is not None:
            logger.info("==> Loading Gimg")
            aimg_ckpt = os.path.join(args.afus_ckpt_dir, 'checkpoint_Gimg_iter_%d.pth' % args.afus_iter)
            checkpoint = torch.load(aimg_ckpt)
            generator_img.load_state_dict(checkpoint['model_state'])
        logger.info("==> Done")


def eval_single_ckpt(root_result_dir):
    # root_result_dir = os.path.join(root_result_dir, 'eval')
    # # set epoch_id and output dir
    # num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    # epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    # root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
    # if args.test:
    #     root_result_dir = os.path.join(root_result_dir, 'test_mode')
    #
    # if args.extra_tag != 'default':
    #     root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    # os.makedirs(root_result_dir, exist_ok = True)

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger = logger)

    # create dataloader & network
    test_loader = create_dataloader(logger, root_result_dir)
    # # model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    # if args.model_type == 'base':
    #     model = PointRCNN(num_classes = test_loader.dataset.num_class, use_xyz = True, mode = 'TEST')
    input_channels = int(cfg.RPN.USE_INTENSITY) + 3 * int(cfg.RPN.USE_RGB)
    if args.fusion:
        generator_pts = Generator_fuspts(input_channels=input_channels, use_xyz=True)
        generator_img = Generator_fusimg(num_channels=3, ngf=100)
        generator_img.cuda()
    elif args.delta:
        generator_pts = Perturbation(batch_size=2)
        generator_img = None
    else:
        generator_pts = Generator_pts(input_channels=input_channels, use_xyz=True)
        generator_img = None
    # elif args.model_type == 'rpn_mscale':
    #     model = PointRCNN_mScale(num_classes = test_loader.dataset.num_class, use_xyz = True, mode = 'TEST')

    generator_pts.cuda()

    # copy important files to backup
    # backup_dir = os.path.join(root_result_dir, 'backup_files')
    # os.makedirs(backup_dir, exist_ok = True)
    # os.system('cp *.py %s/' % backup_dir)
    # os.system('cp ../lib/net/*.py %s/' % backup_dir)
    # os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    # load checkpoint
    load_ckpt_based_on_args(generator_pts, logger, generator_img=generator_img)

    # start evaluation
    eval_one_epoch_joint(generator_pts, test_loader, root_result_dir, logger, generator_img=generator_img)


def create_dataloader(logger, root_result_dir):
    mode = 'TEST' if args.test else 'EVAL'
    DATA_PATH = os.path.join('../', 'data')

    # create dataloader
    test_set = KittiRCNNDataset(root_dir = DATA_PATH, npoints = cfg.RPN.NUM_POINTS, split = cfg.TEST.SPLIT, mode = mode,
                                random_select = args.random_select,
                                rcnn_eval_roi_dir = args.rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir = args.rcnn_eval_feature_dir,
                                classes = cfg.CLASSES,
                                logger = logger,
                                save_dir=root_result_dir)

    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, pin_memory = True,
                             num_workers = args.workers, collate_fn = test_set.collate_batch)

    return test_loader


if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rpn', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rcnn_online':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
        assert args.rcnn_eval_roi_dir is not None and args.rcnn_eval_feature_dir is not None
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok = True)

    with torch.no_grad():
        eval_single_ckpt(root_result_dir)
