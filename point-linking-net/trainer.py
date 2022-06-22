from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from torch import nn

import torch as t
from utils import array_tool as at
import numpy as np
from utils.config import option
from utils.vis_tool import Visualizer
from torchnet.meter import ConfusionMeter, AverageValueMeter

from data.dataset import inverse_normalize
from utils.vis_tool import visdom_bbox
LossTuple = namedtuple('LossTuple',
                       ['pt_pexist_loss',
                        'pt_cls_loss',
                        'pt_offset_loss',
                        'pt_link_loss',
                        'pt_loss',
                        'nopt_loss',
                        'add_pexist_nopt_loss',
                        'center_exist_loss',
                        'center_pt_loss',
                        'center_nopt_loss',
                        'center_offset_loss', 
                        'total_loss',
                        ])

def gt_convert(bboxes, labels, H, W, grid_size, classes):
#need to optimize the situation of repetitive elements
    gt_ps = list()
    gt_ps_d = list()
    gt_cs = list()
    gt_cs_d = list()
    gt_labels = list()
    gt_linkcs_x = list()
    gt_linkcs_y = list()
    gt_linkps_x = list()
    gt_linkps_y = list()  
    #print(bboxes) 
    bboxes = bboxes / W * grid_size
    bboxes = bboxes[0]
    for which, b in enumerate(bboxes):
        #print(b)
        x0 = int(b[0])
        y0 = int(b[1])
        x0_d = b[0] - x0
        y0_d = b[1] - y0

        x1 = int( b[2])
        y1 =  int(b[1])
        x1_d = b[2] - x1
        y1_d = b[1] - y1

        x2 = int(b[0])
        y2 = int(b[3])
        x2_d = b[0] - x2
        y2_d = b[3] - y2

        x3 = int(b[2])
        y3 = int(b[3])
        x3_d = b[2] - x3
        y3_d = b[3] - y3

        xc = int((b[0] + b[2]) / 2)
        yc = int((b[1] + b[3]) / 2)
        xc_d = (b[0] + b[2]) / 2 - xc
        yc_d = (b[1] + b[3]) / 2 - yc

        x0_ = [x0, y0]
        x1_ = [x1, y1]
        x2_ = [x2, y2]
        x3_ = [x3, y3]
        xc_ = [xc, yc]
        gt_ps.append([x0_, x1_, x2_, x3_])
        gt_ps_d.append([[x0_d, y0_d], [x1_d, y1_d], [x2_d, y2_d], [x3_d, y3_d]])
        gt_cs.append(xc_)
        gt_cs_d.append([xc_d, yc_d])
        gt_label = np.zeros((classes)).tolist()
        gt_label[labels[0][which]] = 1
        gt_labels.append(gt_label)

        gt_linkc_x = np.zeros((grid_size)).tolist()
        gt_linkc_x[xc] = 1
        gt_linkcs_x.append(gt_linkc_x)
        gt_linkc_y = np.zeros((grid_size)).tolist()
        gt_linkc_y[yc] = 1
        gt_linkcs_y.append(gt_linkc_y)

        gt_linkp_x = np.zeros((4, grid_size)).tolist()
        gt_linkp_y = np.zeros((4, grid_size)).tolist()
        for i, p in enumerate(gt_ps[which][0: 4]):
            gt_linkp_x[i][p[0]] = 1
            gt_linkp_y[i][p[1]] = 1
        gt_linkps_x.append(gt_linkp_x)
        gt_linkps_y.append(gt_linkp_y)
        
    return np.array(gt_ps), t.Tensor(gt_ps_d), t.Tensor(gt_cs), t.Tensor(gt_cs_d), t.Tensor(gt_labels), t.Tensor(gt_linkcs_x), t.Tensor(gt_linkcs_y), t.Tensor(gt_linkps_x), t.Tensor(gt_linkps_y)


class Trainer(nn.Module):
    def __init__(self, point_link):
        super(Trainer, self).__init__()

        self.point_link = point_link
        self.grid_size = 14
        self.B = 2 # means the number of predict point in each grid
        self.optimizer = self.point_link.get_optimizer()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.cls_loss = nn.CrossEntropyLoss()
        self.classes = 20
        self.w_class = 1
        self.w_coord = 5
        self.w_link = 1
        self.w_pt = 1
        self.w_nopt = 0.05
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}
        self.vis = Visualizer(env=option.env)

    def compute_loss(self, out_four, bboxes, labels, H, W):
        """

        Args:
            out_four:
            bboxes:
            labels:
            H:
            W:

        Returns: losses of four branches

        """
        loss = t.empty(4)
        gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y, gt_linkps_x, gt_linkps_y = \
            gt_convert(bboxes, labels, H, W, self.grid_size, self.classes)
        #if len(bboxes[0])<3:
            #print("bboxes, labels", bboxes, labels)
            #print("gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels", gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels)
        
        total_loss = 0
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss_pt = 0
        loss_nopt = 0
        center_exist_loss = 0
        center_pt_loss = 0
        center_nopt_loss = 0
        center_offset_loss = 0
        gt_point = np.zeros((448, 448, 3), np.int32)
        predict_exist = np.zeros((3, 448, 448), np.int32)
        gt_point.fill(255)
        predict_exist.fill(255)

        gt_center = np.zeros((3, 448, 448), np.int32)
        center_exist = np.zeros((3, 448, 448), np.int32)
        gt_center.fill(255)
        center_exist.fill(255)
        
        for i_x in range(14):
            for i_y in range(14):
                predict_exist[0][i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 255 * out_four.detach().numpy()[0][i_x, i_y, 0, 0]
                predict_exist[1][i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 255 * out_four.detach().numpy()[0][i_x, i_y, 0, 0]
                predict_exist[2][i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 255 * out_four.detach().numpy()[0][i_x, i_y, 0, 0]
        #predict_exist[0] = out_four[0][:, :, 0, 0].detach().numpy()
        #predict_exist[1] = out_four[0][:, :, 0, 0].detach().numpy()
        #predict_exist[2] = out_four[0][:, :, 0, 0].detach().numpy()
        predict_exist[0][0*32: 0*32+32, 0*32: 0*32+32] = 1
        predict_exist[1][0*32: 0*32+32, 0*32: 0*32+32] = 1
        predict_exist[2][0*32: 0*32+32, 0*32: 0*32+32] = 1

        for i_x in range(14):
            for i_y in range(14):
                center_exist[0][i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 255 * out_four.detach().numpy()[0][i_x, i_y, 2, 0]
                center_exist[1][i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 255 * out_four.detach().numpy()[0][i_x, i_y, 2, 0]
                center_exist[2][i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 255 * out_four.detach().numpy()[0][i_x, i_y, 2, 0]

        center_exist[0][0*32: 0*32+32, 0*32: 0*32+32] = 1
        center_exist[1][0*32: 0*32+32, 0*32: 0*32+32] = 1
        center_exist[2][0*32: 0*32+32, 0*32: 0*32+32] = 1
 
        for direction in range(4):
            out = out_four[direction]
            for i_x in range(14):
                for i_y in range(14):
                    for j in range(2 * self.B):
                        if j < self.B:  #point
                            if [i_x, i_y] in gt_ps[:, direction].tolist():
                                gt_point[i_x*32: i_x*32+32, i_y*32: i_y*32+32, :] = 0
                                index_tup = np.where(gt_ps[:, direction] == [i_x, i_y])
                                which = index_tup[0][0]
                                loss1 += (out[i_x, i_y, j, 0] - 1) ** 2
                                #print("====== check ====")
                                #print("[i_x, i_y]", [i_x, i_y], which, [out[i_x, i_y, j, 1: 1 + self.classes], gt_labels[which], "check label"])
                                loss2 += self.w_class * self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes],
                                                                     gt_labels[which])
                                #print("loss2", self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes],
                                                                     #gt_labels[which]))
                                loss3 += self.w_coord * self.mse_loss(
                                    out[i_x, i_y, j, 1 + self.classes: 3 + self.classes], gt_ps_d[which][direction])
                                loss4 += self.w_link * self.mse_loss(
                                    out[i_x, i_y, j, 3 + self.classes: 3 + self.classes + self.grid_size],
                                    gt_linkcs_x[which]) + \
                                        self.mse_loss(
                                        out[i_x, i_y, j, 3 + self.classes + self.grid_size: 3 + self.classes + 2 * self.grid_size],
                                        gt_linkcs_y[which])
                            else:
                                loss_nopt += out[i_x, i_y, j, 0] ** 2
                        if j >= self.B:   #center point
                            if [i_x, i_y] in gt_cs.tolist():
                                gt_center[:, i_x*32: i_x*32+32, i_y*32: i_y*32+32] = 0
                                index_tup = np.where(gt_cs.numpy() == [i_x, i_y])
                                which = index_tup[0][0]
                                loss1 += (out[i_x, i_y, j, 0] - 1) ** 2
                                center_exist_loss += (out[i_x, i_y, 2, 0] - 1) ** 2
                                center_pt_loss += (out[i_x, i_y, 2, 0] - 1) ** 2
                                #center_exist_loss += self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes],
                                                                     #gt_labels[which])
                                loss2 += self.w_class * self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes],
                                                                     gt_labels[which])
                                center_offset_loss += self.w_coord * self.mse_loss(
                                    out[i_x, i_y, 2, 1 + self.classes: 3 + self.classes], gt_cs_d[which])
                                center_exist_loss += self.w_coord * self.mse_loss(
                                    out[i_x, i_y, 2, 1 + self.classes: 3 + self.classes], gt_cs_d[which])
                                loss3 += self.w_coord * self.mse_loss(
                                    out[i_x, i_y, j, 1 + self.classes: 3 + self.classes], gt_cs_d[which])
                                loss4 += self.w_link * self.mse_loss(
                                    out[i_x, i_y, j, 3 + self.classes: 3 + self.classes + self.grid_size],
                                    gt_linkps_x[which][direction]) + \
                                        self.mse_loss(
                                        out[i_x, i_y, j, 3 + self.classes + self.grid_size: 3 + self.classes + 2 * self.grid_size],
                                        gt_linkcs_y[which])
                            else:
                                loss_nopt += out[i_x, i_y, 2, 0] ** 2
                                center_nopt_loss += out[i_x, i_y, 2, 0] ** 2
                                center_exist_loss += self.w_nopt * out[i_x, i_y, 2, 0] ** 2
        # print("losses:")
        # print(center_exist_loss)
        loss_pt = loss1 + loss2 + loss3 + loss4
        total_loss = self.w_pt * loss_pt + self.w_nopt * loss_nopt
        losses = [loss1, loss2, loss3, loss4, loss_pt, loss_nopt, loss1 + self.w_nopt * loss_nopt, center_exist_loss, center_pt_loss, center_nopt_loss, center_offset_loss, total_loss]
        #print("losses:  ", losses)
        gt_point = gt_point.transpose([2, 0, 1])
        #predict_exist = predict_exist.transpose([0, 2, 0])
        predict_exist = 255 - predict_exist*255
        center_exist = 255 - center_exist*255
        #print(gt_grid.shape)
        #print("get_grid_type:", type(gt_grid[i]))
        # print('_____________lossTuple_____________')
        # print(LossTuple(*losses))
        # print('________________end________________')
        return LossTuple(*losses)

    def check_loss(self, out_four, bboxes, labels, H, W):
        gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y, gt_linkps_x, gt_linkps_y = \
            gt_convert(bboxes, labels, H, W, self.grid_size, self.classes)
        return 

    def forward(self, imgs, bboxes, labels):
        train_gt_img = visdom_bbox(inverse_normalize(at.tonumpy(imgs[0])),
                                     at.tonumpy(bboxes[0]),
                                     at.tonumpy(labels[0]))
        #print(train_gt_img.shape)
        self.vis.img('train_gt_img', train_gt_img)
        _, _, H, W = imgs.shape
        img_size = (H, W)
        out_four = self.point_link(imgs)
        # print('_________out_four_________')
        # print(out_four)
        # print('___________end____________')
        #print("bbox in forward", bboxes)
        loss = self.compute_loss(out_four, bboxes, labels, H, W)

        return loss

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            print("load model")
            self.point_link.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.point_link.load_state_dict(state_dict)
            return self
        if parse_opt:
            option._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def train_step(self, imgs, bboxes, labels):
        self.optimizer.zero_grad()
        
        losses = self.forward(imgs, bboxes, labels)
        #print("bbox in train_step", bboxes.shape, bboxes)
        #print("========losses.total_loss===========")
        #print(losses.total_loss)
        losses.center_exist_loss.backward()
        #losses.pt_link_loss.backward()
        #losses.add_pexist_nopt_loss.backward()
        #losses.center_exist_loss.backward()
        # self.optimizer.zero_grad()
        self.optimizer.step()
        self.update_meters(losses)

        return losses


    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.point_link.state_dict()
        save_dict['config'] = option._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/workdirs/2206_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_
            save_path += '.pth'
            '''tmp'''
            # save_path = 'checkpoints/workdirs/0711_0.05w_0.0001_no_sigmoid%s.pth' % timestr
            # for k_, v_ in kwargs.items():
            #     save_path += '_%s' % v_
            # # save_path += '.pth'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path



