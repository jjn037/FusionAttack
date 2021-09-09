import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from collections import namedtuple
import numpy as np
from lib.net.point_rcnn import PointRCNN
from lib.net.GAN_model import Generator_fusimg, Generator_fuspts
import random


def model_joint_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def model_fn(model, data):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb
        ret_dict = model(input_data)

        tb_dict = { }
        disp_dict = { }
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']

            rpn_loss, rpn_loss_cls, rpn_loss_loc, rpn_loss_angle, rpn_loss_size, rpn_loss_iou = get_rpn_loss(model,
                                                                                                             rpn_cls,
                                                                                                             rpn_reg,
                                                                                                             rpn_cls_label,
                                                                                                             rpn_reg_label,
                                                                                                             tb_dict)
            rpn_loss = rpn_loss * cfg.TRAIN.RPN_TRAIN_WEIGHT
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()
            disp_dict['rpn_loss_cls'] = rpn_loss_cls.item()
            disp_dict['rpn_loss_loc'] = rpn_loss_loc.item()
            disp_dict['rpn_loss_angle'] = rpn_loss_angle.item()
            disp_dict['rpn_loss_size'] = rpn_loss_size.item()
            disp_dict['rpn_loss_iou'] = rpn_loss_iou.item()

        if cfg.RCNN.ENABLED:
            if cfg.USE_IOU_BRANCH:
                rcnn_loss,iou_loss,iou_branch_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()
                disp_dict['rcnn_iou_loss'] = iou_loss.item()
                disp_dict['iou_branch_loss'] = iou_branch_loss.item()
            else:
                rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()


        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight = weight, reduction = 'none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rpn_cls_flat)[fg_mask], torch.sigmoid(rpn_cls_flat)[fg_mask],
                                        rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope = cfg.RPN.LOC_SCOPE,
                                        loc_bin_size = cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size = MEAN_SIZE,
                                        get_xz_fine = cfg.RPN.LOC_XZ_FINE,
                                        use_cls_score = True,
                                        use_mask_score = False)

            loss_size = 3 * loss_size  # consistent with old codes

            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            rpn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
        else:
            # loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0
            loss_loc = loss_angle = loss_size = loss_iou = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({ 'rpn_loss_cls'  : rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                         'rpn_loss'      : rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                         'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item(),
                         'rpn_loss_iou'  : loss_iou.item() })

        # return rpn_loss
        return rpn_loss, rpn_loss_cls, loss_loc, loss_angle, loss_size, loss_iou

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']
        mask_score = ret_dict['mask_score']

        gt_iou_weight = ret_dict['gt_iou']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction = 'none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min = 1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim = 1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:

            if cfg.USE_IOU_BRANCH:
                iou_branch_pred = ret_dict['rcnn_iou_branch']
                iou_branch_pred_fg_mask = iou_branch_pred[fg_mask]
            else:
                iou_branch_pred_fg_mask = None

            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rcnn_cls_flat)[fg_mask], mask_score[fg_mask],
                                        rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope = cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size = cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size = anchor_size,
                                        get_xz_fine = True, get_y_by_bin = cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope = cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine = True,
                                        use_cls_score = True,
                                        use_mask_score = True,
                                        gt_iou_weight = gt_iou_weight[fg_mask],
                                        use_iou_branch = cfg.USE_IOU_BRANCH,
                                        iou_branch_pred = iou_branch_pred_fg_mask)


            loss_size = 3 * loss_size  # consistent with old codes
            # rcnn_loss_reg = loss_loc + loss_angle + loss_size
            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            if cfg.USE_IOU_BRANCH:
                iou_branch_loss = reg_loss_dict['iou_branch_loss']
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou  + iou_branch_loss
            else:
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = loss_iou = rcnn_loss_reg = iou_branch_loss = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        tb_dict['rcnn_loss_iou'] = loss_iou.item()
        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        if cfg.USE_IOU_BRANCH:
            tb_dict['iou_branch_loss'] = iou_branch_loss.item()
            # print('\n')
            # print('iou_branch_loss:',iou_branch_loss.item())
            return rcnn_loss, loss_iou, iou_branch_loss
        else:
            return rcnn_loss


    return model_fn


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input).cuda()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)

    return x


def L2_dist(x, y):
    return reduce_sum((x - y) ** 2)


def model_aimg_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])
    criterionGAN = GANLoss()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std

    def model_fn(model, generator, discriminator, data, optimizer, optimizer_D, it_g, tb_log=None,
                 cg=0.05, c_misclassify=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img_ori'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        for i in range(1):
            perturbation = generator(input_data['img_ori'])
            adv_img = input_data['img_ori'] + perturbation
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            optimizer_D.zero_grad()

            pred_real = discriminator(input_data['img_ori'].detach())
            # print(pred_real.shape)
            # pau = input('Pause: ')
            loss_D_real = criterionGAN(pred_real, True)
            # loss_D_real.backward(retain_graph=False)

            pred_fake = discriminator(input_data['img'].detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # loss_D_fake.backward(retain_graph=False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizer_D.step()

            # print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))
        tb_dict.update({'loss_D_fake': loss_D_fake.item(), 'loss_D_real': loss_D_real.item(),
                        'loss_D': loss_D.item()})
        disp_dict['D'] = loss_D.item()

        loss_perturb = 20
        if cfg.ATTACK.LOSS_TYPE == 0:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
            cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        else:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        tbw_dict = {}
        for i in range(1):
            optimizer.zero_grad()
            perturbation = generator(input_data['img_ori'])
            adv_img = input_data['img_ori'] + perturbation
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            pred_fake = discriminator(input_data['img'])
            loss_GAN = criterionGAN(pred_fake, True)
            loss_perturb = cg * torch.mean(L2_dist(input_data['img_ori'], input_data['img']))

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
            else:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)

            loss_total = loss_perturb + loss_GAN + loss_misclassify
            loss_total.backward()
            optimizer.step()

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_perturb': loss_perturb.item(), 'wloss_GAN': loss_GAN.item(),
                            'wloss_misclassify': loss_misclassify.item(), 'wloss_total': loss_total.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_perturb': loss_perturb.item(), 'loss_GAN': loss_GAN.item(),
                         'loss_misclassify': loss_misclassify.item(), 'loss_total': loss_total.item()})
        disp_dict['GAN'] = loss_GAN.item()
        disp_dict['Pimg'] = loss_perturb.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_aimg_fn_eval_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std

    def model_fn(model, generator, data):
        tb_dict = {}
        disp_dict = {}
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            perturbation = generator(img)
            pert_dist = torch.mean(reduce_sum(perturbation ** 2))
            tb_dict['pert_dist'] = pert_dist
            disp_dict['pert_dist'] = pert_dist
            input_data['img'] = img + perturbation
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(input_data['img'][:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb
        ret_dict = model(input_data)

        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']

            rpn_loss, rpn_loss_cls, rpn_loss_loc, rpn_loss_angle, rpn_loss_size, rpn_loss_iou = get_rpn_loss(model,
                                                                                                             rpn_cls,
                                                                                                             rpn_reg,
                                                                                                             rpn_cls_label,
                                                                                                             rpn_reg_label,
                                                                                                             tb_dict)
            rpn_loss = rpn_loss * cfg.TRAIN.RPN_TRAIN_WEIGHT
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()
            disp_dict['rpn_loss_cls'] = rpn_loss_cls.item()
            disp_dict['rpn_loss_loc'] = rpn_loss_loc.item()
            disp_dict['rpn_loss_angle'] = rpn_loss_angle.item()
            disp_dict['rpn_loss_size'] = rpn_loss_size.item()
            disp_dict['rpn_loss_iou'] = rpn_loss_iou.item()

        if cfg.RCNN.ENABLED:
            if cfg.USE_IOU_BRANCH:
                rcnn_loss,iou_loss,iou_branch_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()
                disp_dict['rcnn_iou_loss'] = iou_loss.item()
                disp_dict['iou_branch_loss'] = iou_branch_loss.item()
            else:
                rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()

        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight = weight, reduction = 'none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rpn_cls_flat)[fg_mask], torch.sigmoid(rpn_cls_flat)[fg_mask],
                                        rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope = cfg.RPN.LOC_SCOPE,
                                        loc_bin_size = cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size = MEAN_SIZE,
                                        get_xz_fine = cfg.RPN.LOC_XZ_FINE,
                                        use_cls_score = True,
                                        use_mask_score = False)

            loss_size = 3 * loss_size  # consistent with old codes

            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            rpn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
        else:
            # loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0
            loss_loc = loss_angle = loss_size = loss_iou = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({ 'rpn_loss_cls'  : rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                         'rpn_loss'      : rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                         'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item(),
                         'rpn_loss_iou'  : loss_iou.item() })

        # return rpn_loss
        return rpn_loss, rpn_loss_cls, loss_loc, loss_angle, loss_size, loss_iou

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']
        mask_score = ret_dict['mask_score']

        gt_iou_weight = ret_dict['gt_iou']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rcnn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rcnn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction = 'none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min = 1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim = 1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:

            if cfg.USE_IOU_BRANCH:
                iou_branch_pred = ret_dict['rcnn_iou_branch']
                iou_branch_pred_fg_mask = iou_branch_pred[fg_mask]
            else:
                iou_branch_pred_fg_mask = None

            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rcnn_cls_flat)[fg_mask], mask_score[fg_mask],
                                        rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope = cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size = cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size = anchor_size,
                                        get_xz_fine = True, get_y_by_bin = cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope = cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine = True,
                                        use_cls_score = True,
                                        use_mask_score = True,
                                        gt_iou_weight = gt_iou_weight[fg_mask],
                                        use_iou_branch = cfg.USE_IOU_BRANCH,
                                        iou_branch_pred = iou_branch_pred_fg_mask)


            loss_size = 3 * loss_size  # consistent with old codes
            # rcnn_loss_reg = loss_loc + loss_angle + loss_size
            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            if cfg.USE_IOU_BRANCH:
                iou_branch_loss = reg_loss_dict['iou_branch_loss']
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou  + iou_branch_loss
            else:
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = loss_iou = rcnn_loss_reg = iou_branch_loss = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        tb_dict['rcnn_loss_iou'] = loss_iou.item()
        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        if cfg.USE_IOU_BRANCH:
            tb_dict['iou_branch_loss'] = iou_branch_loss.item()
            # print('\n')
            # print('iou_branch_loss:',iou_branch_loss.item())
            return rcnn_loss, loss_iou, iou_branch_loss
        else:
            return rcnn_loss

    return model_fn


def model_afus_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])
    criterionGAN = GANLoss()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std

    def model_fn(model, data, generator, discriminator, pts_pert,
                 optimizer_Gimg, optimizer_Dimg, optimizer_Ppts,
                 it_g, tb_log=None, cg=0.05, c_misclassify=1.0, c_dpts=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img_ori'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        for i in range(1):
            img_pert = generator(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            optimizer_Dimg.zero_grad()

            pred_real = discriminator(input_data['img_ori'].detach())
            # print(pred_real.shape)
            # pau = input('Pause: ')
            loss_D_real = criterionGAN(pred_real, True)
            # loss_D_real.backward(retain_graph=False)

            pred_fake = discriminator(input_data['img'].detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # loss_D_fake.backward(retain_graph=False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizer_Dimg.step()

            # print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))
        tb_dict.update({'loss_D_fake': loss_D_fake.item(), 'loss_D_real': loss_D_real.item(),
                        'loss_D': loss_D.item()})
        disp_dict['D'] = loss_D.item()

        # loss_perturb = 20
        if cfg.ATTACK.LOSS_TYPE == 0:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
            cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        else:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        tbw_dict = {}
        for i in range(1):
            optimizer_Gimg.zero_grad()
            optimizer_Ppts.zero_grad()
            img_pert = generator(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            pred_fake = discriminator(input_data['img'])
            loss_GAN = criterionGAN(pred_fake, True)
            loss_pert_img = cg * torch.mean(L2_dist(input_data['img_ori'], input_data['img']))

            input_data['pts_input'] = pts_pert(input_data['pts_input_ori'])
            loss_pert_pts = c_dpts * pts_pert.get_norm()

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
            else:
                # rcnn_prob = torch.sigmoid(rcnn_cls_flat)
                # loss1 = - cls_label_flat * torch.clamp(torch.log(1. + confidence - rcnn_prob), max=0.) - \
                #         (1. - cls_label_flat) * torch.clamp(torch.log(1. - confidence + rcnn_prob), max=0.)
                # loss_misclassify = c_misclassify * (loss1 * cls_valid_mask).sum() / \
                #                    torch.clamp(cls_valid_mask.sum(), min=1.0)
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)

            loss_img = loss_pert_img + loss_GAN + loss_misclassify
            loss_pts = loss_pert_pts + loss_misclassify

            loss_img.backward(retain_graph=True)
            optimizer_Gimg.step()

            loss_pts.backward()
            optimizer_Ppts.step()

            # for name, parms in pts_pert.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause: ')
            # for name, parms in generator.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause: ')
            #     break

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_pert_img': loss_pert_img.item(), 'wloss_GAN': loss_GAN.item(),
                             'wloss_misclassify': loss_misclassify.item(), 'wloss_pert_pts': loss_pert_pts.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_pert_img': loss_pert_img.item(), 'loss_GAN': loss_GAN.item(),
                        'loss_misclassify': loss_misclassify.item(), 'loss_img': loss_img.item(),
                        'loss_pert_pts': loss_pert_pts.item(), 'loss_pts': loss_pts.item()})
        disp_dict['GAN'] = loss_GAN.item()
        disp_dict['Pimg'] = loss_pert_img.item()
        disp_dict['Ppts'] = loss_pert_pts.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_apts_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])

    def model_fn(model, data, pts_pert, optimizer_Ppts,
                 it_g, tb_log=None, c_misclassify=1.0, c_dpts=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        if cfg.ATTACK.LOSS_TYPE == 0:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
            cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        else:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        tbw_dict = {}
        for i in range(1):
            optimizer_Ppts.zero_grad()
            input_data['pts_input'] = pts_pert(input_data['pts_input_ori'])
            loss_pert_pts = c_dpts * pts_pert.get_norm()

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
            else:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)

            loss_pts = loss_pert_pts + loss_misclassify

            loss_pts.backward()
            optimizer_Ppts.step()

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_misclassify': loss_misclassify.item(), 'wloss_pert_pts': loss_pert_pts.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_misclassify': loss_misclassify.item(), 'loss_pert_pts': loss_pert_pts.item(),
                        'loss_pts': loss_pts.item()})
        disp_dict['Ppts'] = loss_pert_pts.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_aptsg_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])

    def model_fn(model, data, g_pts, optimizer_Gpts,
                 it_g, tb_log=None, c_misclassify=1.0, c_dpts=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        # loss_perturb = 20
        cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        # if cfg.ATTACK.LOSS_TYPE == 0:
        #     cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        #     cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        # else:
        #     cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        tbw_dict = {}
        for i in range(1):
            optimizer_Gpts.zero_grad()
            # input_data['pts_input'] = g_pts(input_data['pts_input_ori'])
            pts_pert = g_pts(input_data['pts_input_ori'])
            input_data['pts_input'] = input_data['pts_input_ori'] + pts_pert
            # loss_pert_pts = c_dpts * torch.mean(L2_dist(input_data['pts_input_ori'], input_data['pts_input']))
            loss_pert_pts = c_dpts * torch.mean(reduce_sum(pts_pert ** 2))

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
            elif cfg.ATTACK.LOSS_TYPE == 1:
                # rcnn_prob = torch.sigmoid(rcnn_cls_flat)
                # loss1 = - cls_label_flat * torch.clamp(torch.log(1. + confidence - rcnn_prob), max=0.) - \
                #         (1. - cls_label_flat) * torch.clamp(torch.log(1. - confidence + rcnn_prob), max=0.)
                # loss_misclassify = c_misclassify * (loss1 * cls_valid_mask).sum() / \
                #                    torch.clamp(cls_valid_mask.sum(), min=1.0)
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)
            elif cfg.ATTACK.LOSS_TYPE == 2:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(pos_rpn.sum(), min=1.0)
            else:
                # loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                #         neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))

            loss_pts = loss_pert_pts + loss_misclassify

            # loss_img.backward(retain_graph=True)
            # optimizer_Gimg.step()
            #
            loss_pts.backward()
            optimizer_Gpts.step()

            # for name, parms in g_pts.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause')
            #     break

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_misclassify': loss_misclassify.item(), 'wloss_pert_pts': loss_pert_pts.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_misclassify': loss_misclassify.item(),
                        'loss_pert_pts': loss_pert_pts.item(), 'loss_pts': loss_pts.item()})
        disp_dict['Ppts'] = loss_pert_pts.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_amis_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])
    criterionGAN = GANLoss()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std

    def model_fn(model, data, generator, discriminator, pts_pert,
                 optimizer_Gimg, optimizer_Dimg, optimizer_Ppts,
                 it_g, tb_log=None, cg=0.05, c_misclassify=1.0, c_dpts=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img_ori'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        for i in range(1):
            img_pert = generator(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            optimizer_Dimg.zero_grad()

            pred_real = discriminator(input_data['img_ori'].detach())
            # print(pred_real.shape)
            # pau = input('Pause: ')
            loss_D_real = criterionGAN(pred_real, True)
            # loss_D_real.backward(retain_graph=False)

            pred_fake = discriminator(input_data['img'].detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # loss_D_fake.backward(retain_graph=False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizer_Dimg.step()

            # print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))
        tb_dict.update({'loss_D_fake': loss_D_fake.item(), 'loss_D_real': loss_D_real.item(),
                        'loss_D': loss_D.item()})
        disp_dict['D'] = loss_D.item()

        # loss_perturb = 20
        if cfg.ATTACK.LOSS_TYPE == 0:
            cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
            cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        else:
            confidence = cfg.RCNN.SCORE_THRESH
            # confidence1 = cfg.RCNN.SCORE_THRESH - 0.1
            # confidence2 = cfg.RCNN.SCORE_THRESH + 0.1
        tbw_dict = {}
        for i in range(1):
            optimizer_Gimg.zero_grad()
            optimizer_Ppts.zero_grad()
            img_pert = generator(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            # pred_fake = discriminator(input_data['img'])
            # loss_GAN = criterionGAN(pred_fake, True)
            # loss_pert_img = cg * torch.mean(L2_dist(input_data['img_ori'], input_data['img']))
            dist_pert_img = torch.mean(L2_dist(input_data['img_ori'].detach(), input_data['img'].detach()))

            input_data['pts_input'] = pts_pert(input_data['pts_input_ori'])
            # loss_pert_pts = c_dpts * pts_pert.get_norm()
            dist_pert_pts = pts_pert.get_norm().detach()

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
                # loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)
            else:
                rcnn_prob = torch.sigmoid(rcnn_cls_flat)
                loss1 = - cls_label_flat * torch.clamp(torch.log(1. + confidence - rcnn_prob), max=0.) - \
                        (1. - cls_label_flat) * torch.clamp(torch.log(1. - confidence + rcnn_prob), max=0.)
                loss_misclassify = c_misclassify * (loss1 * cls_valid_mask).sum() / \
                                   torch.clamp(cls_valid_mask.sum(), min=1.0)

            # loss_img = loss_pert_img + loss_GAN + loss_misclassify
            # loss_pts = loss_pert_pts + loss_misclassify

            loss_misclassify.backward()
            for name, parms in pts_pert.named_parameters():
                print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
                pau = input('Pause: ')
            for name, parms in generator.named_parameters():
                print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
                pau = input('Pause: ')
                break
            for name, parms in model.rpn.backbone_net.SA_modules[0].mlps.named_parameters():
                print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
                pau = input('Pause: ')
                break
            optimizer_Ppts.step()
            optimizer_Gimg.step()

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_misclassify': loss_misclassify.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_misclassify': loss_misclassify.item(), 'dist_pert_img': dist_pert_img.item(),
                        'dist_pert_pts': dist_pert_pts.item()})
        # disp_dict['GAN'] = loss_GAN.item()
        disp_dict['Pimg'] = dist_pert_img.item()
        disp_dict['Ppts'] = dist_pert_pts.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_acombg_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])
    criterionGAN = GANLoss()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std

    def model_fn(model, data, g_img, d_img, g_pts, optimizer_Gimg, optimizer_Dimg, optimizer_Gpts,
                 it_g, tb_log=None, cg=0.05, c_misclassify=1.0, c_dpts=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img_ori'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        for i in range(1):
            img_pert = g_img(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            optimizer_Dimg.zero_grad()

            pred_real = d_img(input_data['img_ori'].detach())
            # print(pred_real.shape)
            # pau = input('Pause: ')
            loss_D_real = criterionGAN(pred_real, True)
            # loss_D_real.backward(retain_graph=False)

            pred_fake = d_img(input_data['img'].detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # loss_D_fake.backward(retain_graph=False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizer_Dimg.step()

            # print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))
        tb_dict.update({'loss_D_fake': loss_D_fake.item(), 'loss_D_real': loss_D_real.item(),
                        'loss_D': loss_D.item()})
        disp_dict['D'] = loss_D.item()

        # loss_perturb = 20
        cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        # if cfg.ATTACK.LOSS_TYPE == 0:
        #     cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        #     cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        # else:
        #     cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        tbw_dict = {}
        for i in range(1):
            optimizer_Gimg.zero_grad()
            optimizer_Gpts.zero_grad()
            img_pert = g_img(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            pred_fake = d_img(input_data['img'])
            loss_GAN = criterionGAN(pred_fake, True)
            loss_pert_img = cg * torch.mean(L2_dist(input_data['img_ori'], input_data['img']))

            # input_data['pts_input'] = g_pts(input_data['pts_input_ori'])
            pts_pert = g_pts(input_data['pts_input_ori'])
            input_data['pts_input'] = input_data['pts_input_ori'] + pts_pert
            # loss_pert_pts = c_dpts * torch.mean(L2_dist(input_data['pts_input_ori'], input_data['pts_input']))
            loss_pert_pts = c_dpts * torch.mean(reduce_sum(pts_pert ** 2))

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
            elif cfg.ATTACK.LOSS_TYPE == 1:
                # rcnn_prob = torch.sigmoid(rcnn_cls_flat)
                # loss1 = - cls_label_flat * torch.clamp(torch.log(1. + confidence - rcnn_prob), max=0.) - \
                #         (1. - cls_label_flat) * torch.clamp(torch.log(1. - confidence + rcnn_prob), max=0.)
                # loss_misclassify = c_misclassify * (loss1 * cls_valid_mask).sum() / \
                #                    torch.clamp(cls_valid_mask.sum(), min=1.0)
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)
            elif cfg.ATTACK.LOSS_TYPE == 2:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(pos_rpn.sum(), min=1.0)
            else:
                # loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                #         neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))

            loss_img = loss_pert_img + loss_GAN + loss_misclassify
            loss_pts = loss_pert_pts + loss_misclassify

            loss_img.backward(retain_graph=True)
            optimizer_Gimg.step()

            loss_pts.backward()
            optimizer_Gpts.step()

            # loss_misclassify.backward()
            # optimizer_Gimg.step()
            # optimizer_Gpts.step()
            # for name, parms in g_pts.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause')
            #     break
            # for name, parms in g_img.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause')
            #     break

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_pert_img': loss_pert_img.item(), 'wloss_GAN': loss_GAN.item(),
                             'wloss_misclassify': loss_misclassify.item(), 'wloss_pert_pts': loss_pert_pts.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_pert_img': loss_pert_img.item(), 'loss_GAN': loss_GAN.item(),
                        'loss_misclassify': loss_misclassify.item(), 'loss_img': loss_img.item(),
                        'loss_pert_pts': loss_pert_pts.item(), 'loss_pts': loss_pts.item()})
        disp_dict['GAN'] = loss_GAN.item()
        disp_dict['Pimg'] = loss_pert_img.item()
        disp_dict['Ppts'] = loss_pert_pts.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_afusg_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['tb_dict', 'disp_dict', 'it_g'])
    criterionGAN = GANLoss()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std

    def model_fn(model, data, g_img, d_img, g_pts, optimizer_Gimg, optimizer_Dimg, optimizer_Gpts,
                 it_g, tb_log=None, cg=0.05, c_misclassify=1.0, c_dpts=1.0):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            # np.save('img.npy', data['img'])
            # input('Pause: ')
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img_ori'] = img
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb

        tb_dict = {}
        disp_dict = {}

        for i in range(1):
            img_pert, _ = g_img(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            optimizer_Dimg.zero_grad()

            pred_real = d_img(input_data['img_ori'].detach())
            # print(pred_real.shape)
            # pau = input('Pause: ')
            loss_D_real = criterionGAN(pred_real, True)
            # loss_D_real.backward(retain_graph=False)

            pred_fake = d_img(input_data['img'].detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # loss_D_fake.backward(retain_graph=False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizer_Dimg.step()

            # print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))
        tb_dict.update({'loss_D_fake': loss_D_fake.item(), 'loss_D_real': loss_D_real.item(),
                        'loss_D': loss_D.item()})
        disp_dict['D'] = loss_D.item()

        # loss_perturb = 20
        cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        # if cfg.ATTACK.LOSS_TYPE == 0:
        #     cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        #     cfd_rcnn = np.log(cfg.RCNN.SCORE_THRESH / (1. - cfg.RCNN.SCORE_THRESH))
        # else:
        #     cfd_rpn = np.log(cfg.RPN.SCORE_THRESH / (1. - cfg.RPN.SCORE_THRESH))
        tbw_dict = {}
        for i in range(1):
            optimizer_Gimg.zero_grad()
            optimizer_Gpts.zero_grad()
            img_pert, img_pert_feature = g_img(input_data['img_ori'])
            adv_img = input_data['img_ori'] + img_pert
            input_data['img'] = torch.zeros_like(input_data['img_ori']).cuda()
            for j in range(3):
                input_data['img'][:, j, :, :] = torch.clamp(adv_img[:, j, :, :],
                                                            min=clamp_min[j], max=clamp_max[j])
            pred_fake = d_img(input_data['img'])
            loss_GAN = criterionGAN(pred_fake, True)
            loss_pert_img = cg * torch.mean(L2_dist(input_data['img_ori'], input_data['img']))

            # input_data['pts_input'] = g_pts(input_data['pts_input_ori'])
            pts_pert = g_pts(input_data['pts_input_ori'], img_pert_feature, input_data['pts_origin_xy'])
            input_data['pts_input'] = input_data['pts_input_ori'] + pts_pert
            # loss_pert_pts = c_dpts * torch.mean(L2_dist(input_data['pts_input_ori'], input_data['pts_input']))
            loss_pert_pts = c_dpts * torch.mean(reduce_sum(pts_pert ** 2))

            ret_dict = model(input_data, attack=True)

            rpn_cls = ret_dict['rpn_cls']
            rpn_cls_flat = rpn_cls.view(-1)
            rpn_cls_label_flat = rpn_cls_label.view(-1)
            pos_rpn = (rpn_cls_label_flat > 0).float()
            neg_rpn = (rpn_cls_label_flat == 0).float()
            valid_mask_rpn = pos_rpn + neg_rpn

            rcnn_cls = ret_dict['rcnn_cls']
            rcnn_cls_flat = rcnn_cls.view(-1)
            rcnn_cls_label = ret_dict['cls_label'].float()
            rcnn_cls_label_flat = rcnn_cls_label.view(-1)
            pos_rcnn = (rcnn_cls_label_flat > 0).float()
            neg_rcnn = (rcnn_cls_label_flat == 0).float()
            valid_mask_rcnn = pos_rcnn + neg_rcnn

            if cfg.ATTACK.LOSS_TYPE == 0:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0) +
                                                    loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))
            elif cfg.ATTACK.LOSS_TYPE == 1:
                # rcnn_prob = torch.sigmoid(rcnn_cls_flat)
                # loss1 = - cls_label_flat * torch.clamp(torch.log(1. + confidence - rcnn_prob), max=0.) - \
                #         (1. - cls_label_flat) * torch.clamp(torch.log(1. - confidence + rcnn_prob), max=0.)
                # loss_misclassify = c_misclassify * (loss1 * cls_valid_mask).sum() / \
                #                    torch.clamp(cls_valid_mask.sum(), min=1.0)
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                        neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(valid_mask_rpn.sum(), min=1.0)
            elif cfg.ATTACK.LOSS_TYPE == 2:
                loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.)
                loss_misclassify = c_misclassify * loss1.sum() / torch.clamp(pos_rpn.sum(), min=1.0)
            else:
                # loss1 = pos_rpn * torch.clamp(rpn_cls_flat - cfd_rpn, min=0.) + \
                #         neg_rpn * torch.clamp(cfd_rpn - rpn_cls_flat, min=0.)
                loss2 = pos_rcnn * torch.clamp(rcnn_cls_flat - cfd_rcnn, min=0.) + \
                        neg_rcnn * torch.clamp(cfd_rcnn - rcnn_cls_flat, min=0.)
                loss_misclassify = c_misclassify * (loss2.sum() / torch.clamp(valid_mask_rcnn.sum(), min=1.0))

            loss_img = loss_pert_img + loss_GAN + loss_misclassify
            loss_pts = loss_pert_pts + loss_misclassify

            loss_img.backward(retain_graph=True)
            optimizer_Gimg.step()

            loss_pts.backward()
            optimizer_Gpts.step()

            # loss_misclassify.backward()
            # optimizer_Gimg.step()
            # optimizer_Gpts.step()
            # for name, parms in g_pts.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause')
            #     break
            # for name, parms in g_img.named_parameters():
            #     print('name: ', name, ' grad_requirs: ', parms.requires_grad, '\ngrad_value: ', parms.grad)
            #     pau = input('Pause')
            #     break

            # print('loss_perturb: ', loss_perturb.item())

            tbw_dict.update({'wloss_pert_img': loss_pert_img.item(), 'wloss_GAN': loss_GAN.item(),
                             'wloss_misclassify': loss_misclassify.item(), 'wloss_pert_pts': loss_pert_pts.item()})

            # print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))

            it_g += 1
            if tb_log is not None:
                for key, val in tbw_dict.items():
                    tb_log.add_scalar('train_' + key, val, it_g)
        tb_dict.update({'loss_pert_img': loss_pert_img.item(), 'loss_GAN': loss_GAN.item(),
                        'loss_misclassify': loss_misclassify.item(), 'loss_img': loss_img.item(),
                        'loss_pert_pts': loss_pert_pts.item(), 'loss_pts': loss_pts.item()})
        disp_dict['GAN'] = loss_GAN.item()
        disp_dict['Pimg'] = loss_pert_img.item()
        disp_dict['Ppts'] = loss_pert_pts.item()
        disp_dict['Cmis'] = loss_misclassify.item()

        return ModelReturn(tb_dict, disp_dict, it_g)

    return model_fn


def model_advdef_fn_decorator(logger):
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std
    input_channels = int(cfg.RPN.USE_INTENSITY) + 3 * int(cfg.RPN.USE_RGB)
    generator_pts = Generator_fuspts(input_channels=input_channels, use_xyz=True)
    generator_img = Generator_fusimg(num_channels=3, ngf=100)
    generator_pts.cuda()
    generator_img.cuda()
    logger.info("==> Loading Gpts")
    apts_ckpt = '/nfs/volume-456-1/liubingyu/EPNet/tools/log/Car/afusg/ckpt/checkpoint_Gpts_iter_3000.pth'
    checkpoint = torch.load(apts_ckpt)
    generator_pts.load_state_dict(checkpoint['model_state'])
    logger.info("==> Loading Gimg")
    aimg_ckpt = '/nfs/volume-456-1/liubingyu/EPNet/tools/log/Car/afusg/ckpt/checkpoint_Gimg_iter_3000.pth'
    checkpoint = torch.load(aimg_ckpt)
    generator_img.load_state_dict(checkpoint['model_state'])
    generator_pts.eval()
    generator_img.eval()
    logger.info('**********************Attack model loaded**********************')

    def model_fn(model, data):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            with torch.no_grad():
                img_pert, img_pert_feature = generator_img(img)
                input_data['img'] = img + img_pert
                for j in range(3):
                    input_data['img'][:, j, :, :] = torch.clamp(input_data['img'][:, j, :, :],
                                                                min=clamp_min[j], max=clamp_max[j])
                pts_pert = generator_pts(input_data['pts_input_ori'], img_pert_feature, pts_origin_xy)
                input_data['pts_input'] = input_data['pts_input_ori'] + pts_pert
            input_data['pts_origin_xy'] = pts_origin_xy
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb
        ret_dict = model(input_data)

        tb_dict = { }
        disp_dict = { }
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']

            rpn_loss, rpn_loss_cls, rpn_loss_loc, rpn_loss_angle, rpn_loss_size, rpn_loss_iou = get_rpn_loss(model,
                                                                                                             rpn_cls,
                                                                                                             rpn_reg,
                                                                                                             rpn_cls_label,
                                                                                                             rpn_reg_label,
                                                                                                             tb_dict)
            rpn_loss = rpn_loss * cfg.TRAIN.RPN_TRAIN_WEIGHT
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()
            disp_dict['rpn_loss_cls'] = rpn_loss_cls.item()
            disp_dict['rpn_loss_loc'] = rpn_loss_loc.item()
            disp_dict['rpn_loss_angle'] = rpn_loss_angle.item()
            disp_dict['rpn_loss_size'] = rpn_loss_size.item()
            disp_dict['rpn_loss_iou'] = rpn_loss_iou.item()

        if cfg.RCNN.ENABLED:
            if cfg.USE_IOU_BRANCH:
                rcnn_loss,iou_loss,iou_branch_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()
                disp_dict['rcnn_iou_loss'] = iou_loss.item()
                disp_dict['iou_branch_loss'] = iou_branch_loss.item()
            else:
                rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()


        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight = weight, reduction = 'none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rpn_cls_flat)[fg_mask], torch.sigmoid(rpn_cls_flat)[fg_mask],
                                        rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope = cfg.RPN.LOC_SCOPE,
                                        loc_bin_size = cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size = MEAN_SIZE,
                                        get_xz_fine = cfg.RPN.LOC_XZ_FINE,
                                        use_cls_score = True,
                                        use_mask_score = False)

            loss_size = 3 * loss_size  # consistent with old codes

            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            rpn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
        else:
            # loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0
            loss_loc = loss_angle = loss_size = loss_iou = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({ 'rpn_loss_cls'  : rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                         'rpn_loss'      : rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                         'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item(),
                         'rpn_loss_iou'  : loss_iou.item() })

        # return rpn_loss
        return rpn_loss, rpn_loss_cls, loss_loc, loss_angle, loss_size, loss_iou

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']
        mask_score = ret_dict['mask_score']

        gt_iou_weight = ret_dict['gt_iou']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction = 'none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min = 1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim = 1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:

            if cfg.USE_IOU_BRANCH:
                iou_branch_pred = ret_dict['rcnn_iou_branch']
                iou_branch_pred_fg_mask = iou_branch_pred[fg_mask]
            else:
                iou_branch_pred_fg_mask = None

            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rcnn_cls_flat)[fg_mask], mask_score[fg_mask],
                                        rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope = cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size = cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size = anchor_size,
                                        get_xz_fine = True, get_y_by_bin = cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope = cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine = True,
                                        use_cls_score = True,
                                        use_mask_score = True,
                                        gt_iou_weight = gt_iou_weight[fg_mask],
                                        use_iou_branch = cfg.USE_IOU_BRANCH,
                                        iou_branch_pred = iou_branch_pred_fg_mask)


            loss_size = 3 * loss_size  # consistent with old codes
            # rcnn_loss_reg = loss_loc + loss_angle + loss_size
            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            if cfg.USE_IOU_BRANCH:
                iou_branch_loss = reg_loss_dict['iou_branch_loss']
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou  + iou_branch_loss
            else:
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = loss_iou = rcnn_loss_reg = iou_branch_loss = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        tb_dict['rcnn_loss_iou'] = loss_iou.item()
        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        if cfg.USE_IOU_BRANCH:
            tb_dict['iou_branch_loss'] = iou_branch_loss.item()
            # print('\n')
            # print('iou_branch_loss:',iou_branch_loss.item())
            return rcnn_loss, loss_iou, iou_branch_loss
        else:
            return rcnn_loss


    return model_fn


def model_advdef2_fn_decorator(logger):
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    clamp_max = (1. - img_mean) / img_std
    clamp_min = - img_mean / img_std
    input_channels = int(cfg.RPN.USE_INTENSITY) + 3 * int(cfg.RPN.USE_RGB)
    generator_pts = Generator_fuspts(input_channels=input_channels, use_xyz=True)
    generator_img = Generator_fusimg(num_channels=3, ngf=100)
    generator_pts.cuda()
    generator_img.cuda()
    logger.info("==> Loading Gpts")
    apts_ckpt = '/nfs/volume-456-1/liubingyu/EPNet/tools/log/Car/afusg/ckpt/checkpoint_Gpts_iter_3000.pth'
    checkpoint = torch.load(apts_ckpt)
    generator_pts.load_state_dict(checkpoint['model_state'])
    logger.info("==> Loading Gimg")
    aimg_ckpt = '/nfs/volume-456-1/liubingyu/EPNet/tools/log/Car/afusg/ckpt/checkpoint_Gimg_iter_3000.pth'
    checkpoint = torch.load(aimg_ckpt)
    generator_img.load_state_dict(checkpoint['model_state'])
    generator_pts.eval()
    generator_img.eval()
    logger.info('**********************Attack model loaded**********************')

    def model_fn(model, data):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input_ori': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['pts_origin_xy'] = pts_origin_xy
            p_adv = random.random()
            if p_adv >= 0.5:
                with torch.no_grad():
                    img_pert, img_pert_feature = generator_img(img)
                    input_data['img'] = img + img_pert
                    for j in range(3):
                        input_data['img'][:, j, :, :] = torch.clamp(input_data['img'][:, j, :, :],
                                                                    min=clamp_min[j], max=clamp_max[j])
                    pts_pert = generator_pts(input_data['pts_input_ori'], img_pert_feature, pts_origin_xy)
                    input_data['pts_input'] = input_data['pts_input_ori'] + pts_pert
            else:
                input_data['img'] = img
                input_data['pts_input'] = input_data['pts_input_ori']

        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb
        ret_dict = model(input_data)

        tb_dict = { }
        disp_dict = { }
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']

            rpn_loss, rpn_loss_cls, rpn_loss_loc, rpn_loss_angle, rpn_loss_size, rpn_loss_iou = get_rpn_loss(model,
                                                                                                             rpn_cls,
                                                                                                             rpn_reg,
                                                                                                             rpn_cls_label,
                                                                                                             rpn_reg_label,
                                                                                                             tb_dict)
            rpn_loss = rpn_loss * cfg.TRAIN.RPN_TRAIN_WEIGHT
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()
            disp_dict['rpn_loss_cls'] = rpn_loss_cls.item()
            disp_dict['rpn_loss_loc'] = rpn_loss_loc.item()
            disp_dict['rpn_loss_angle'] = rpn_loss_angle.item()
            disp_dict['rpn_loss_size'] = rpn_loss_size.item()
            disp_dict['rpn_loss_iou'] = rpn_loss_iou.item()

        if cfg.RCNN.ENABLED:
            if cfg.USE_IOU_BRANCH:
                rcnn_loss,iou_loss,iou_branch_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()
                disp_dict['rcnn_iou_loss'] = iou_loss.item()
                disp_dict['iou_branch_loss'] = iou_branch_loss.item()
            else:
                rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()


        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight = weight, reduction = 'none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rpn_cls_flat)[fg_mask], torch.sigmoid(rpn_cls_flat)[fg_mask],
                                        rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope = cfg.RPN.LOC_SCOPE,
                                        loc_bin_size = cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size = MEAN_SIZE,
                                        get_xz_fine = cfg.RPN.LOC_XZ_FINE,
                                        use_cls_score = True,
                                        use_mask_score = False)

            loss_size = 3 * loss_size  # consistent with old codes

            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            rpn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
        else:
            # loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0
            loss_loc = loss_angle = loss_size = loss_iou = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({ 'rpn_loss_cls'  : rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                         'rpn_loss'      : rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                         'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item(),
                         'rpn_loss_iou'  : loss_iou.item() })

        # return rpn_loss
        return rpn_loss, rpn_loss_cls, loss_loc, loss_angle, loss_size, loss_iou

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']
        mask_score = ret_dict['mask_score']

        gt_iou_weight = ret_dict['gt_iou']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction = 'none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min = 1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim = 1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:

            if cfg.USE_IOU_BRANCH:
                iou_branch_pred = ret_dict['rcnn_iou_branch']
                iou_branch_pred_fg_mask = iou_branch_pred[fg_mask]
            else:
                iou_branch_pred_fg_mask = None

            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rcnn_cls_flat)[fg_mask], mask_score[fg_mask],
                                        rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope = cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size = cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size = anchor_size,
                                        get_xz_fine = True, get_y_by_bin = cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope = cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine = True,
                                        use_cls_score = True,
                                        use_mask_score = True,
                                        gt_iou_weight = gt_iou_weight[fg_mask],
                                        use_iou_branch = cfg.USE_IOU_BRANCH,
                                        iou_branch_pred = iou_branch_pred_fg_mask)


            loss_size = 3 * loss_size  # consistent with old codes
            # rcnn_loss_reg = loss_loc + loss_angle + loss_size
            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            if cfg.USE_IOU_BRANCH:
                iou_branch_loss = reg_loss_dict['iou_branch_loss']
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou  + iou_branch_loss
            else:
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = loss_iou = rcnn_loss_reg = iou_branch_loss = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        tb_dict['rcnn_loss_iou'] = loss_iou.item()
        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        if cfg.USE_IOU_BRANCH:
            tb_dict['iou_branch_loss'] = iou_branch_loss.item()
            # print('\n')
            # print('iou_branch_loss:',iou_branch_loss.item())
            return rcnn_loss, loss_iou, iou_branch_loss
        else:
            return rcnn_loss


    return model_fn
