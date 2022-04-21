import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


eps = 1e-5
def KL_div(p, q):  # p指的老师老师的预测(经过softmax)，q是学生的预测
    p = p + eps  # 老师
    q = q + eps
    #log_p = p * torch.log(p / q)
    log_p = -1*p*torch.log(q)  # shape=(feature_shape,num_classes +1 )

    return torch.sum(log_p)

class YOLO_KD_Loss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0, focal_loss=False, alpha=0.25, gamma=2):
        super(YOLO_KD_Loss, self).__init__()
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.focal_loss = focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    # def KDLoss(self, pred, target):

    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   求出预测框左上角右下角
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   求出真实框左上角右下角
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        # ----------------------------------------------------#
        #   计算中心的差距
        # ----------------------------------------------------#
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # ----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   计算对角线距离
        # ----------------------------------------------------#
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou

    # ---------------------------------------------------#
    #   平滑标签
    # ---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, input_t, targets=None):
        # ----------------------------------------------------#
        #   l 代表使用的是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets 真实框的标签情况 [batch_size, num_gt, 5]
        #   input是学生网络的输出，input_t是教师网络的输出，两个的shape是一样的
        # ----------------------------------------------------#
        # --------------------------------#
        #   获得图片数量，特征层的高和宽
        # --------------------------------#
        bs = input.size(0)  # batch_size  input.size(1)=3*(5+num_classes)
        in_h = input.size(2)
        in_w = input.size(3)

        # -----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点 实际就是原图缩小了多少倍 32倍
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点 16倍
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点  8倍
        #   stride_h = stride_w = 32、16、8
        #             a_h,  a_w
        #   (anchor([[ 12.,  16.],  # 从这开始是52*52的
        #        [ 19.,  36.],
        #        [ 40.,  28.],
        #        [ 36.,  75.],   # 这开始是26*26的
        #        [ 76.,  55.],
        #        [ 72., 146.],
        #        [142., 110.],   # 从这往下是13*13的
        #        [192., 243.],
        #        [459., 401.]]), 9)
        # -----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3 * (5+num_classes), 13, 13 => bs, 3, 5 + num_classes, 13, 13 => batch_size, 3, 13, 13, 5 + num_classes

        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #   anchors_mask=[[6,7,8], [3,4,5], [0,1,2]]
        #   bbox_attrs = 5+num_classes
        #   in_h和in_w是特征层的尺寸
        #   input将reshape成(batchsize,3,13,13,5+num_classes)
        # -----------------------------------------------#

        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()
        prediction_t = input_t.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # -----------------------------------------------#
        #   先验框的中心位置的调整参数
        # prediction[...,0]=prediction[:,:,:,:,0]
        # x,y是经过sigmoid后的0到1之间的数
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # ------------------------------------------------
        #   获取教师网络的中心位置
        # ------------------------------------------------
        x_t = torch.sigmoid(prediction[..., 0])
        y_t = torch.sigmoid(prediction[..., 1])

        # -----------------------------------------------#
        #   先验框的宽高调整参数
        # -----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]

        # -----------------------------------------------#
        #   获取教师网络的宽高
        # -----------------------------------------------#
        w_t = prediction[..., 2]
        h_t = prediction[..., 3]

        # -----------------------------------------------#
        #   获得置信度，是否有物体
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])

        # -----------------------------------------------
        #  获得教师网络的置信度，判断是否有物体
        # -----------------------------------------------
        conf_t = torch.sigmoid(prediction_t[..., 4])
        #conf_t = F.softmax(prediction_t[..., 4])
        # -----------------------------------------------#
        #   种类置信度 prediction[..., 5:]取的是类别，shape(batchsize,3,13,13,num_classes)
        # -----------------------------------------------#
        #pred_cls = torch.sigmoid(prediction[..., 5:])
        #pred_cls = F.softmax(prediction[...,5:], dim=1)  # dim=1按照行做归一化
        pred_cls = F.log_softmax(prediction[..., 5:], dim=1)
        # -----------------------------------------------
        #   获得教师网络种类
        # -----------------------------------------------
        #pred_cls_t = torch.sigmoid(prediction_t[..., 5:])
        pred_cls_t = F.softmax(prediction_t[..., 5:],dim=1)
        # -----------------------------------------------#
        #   获得网络应该有的预测结果
        # -----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        # ----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        #  教师网络预测进行解码
        noobj_mask_t, pred_boxes_t = self.get_ignore(l, x_t, y_t, h_t, w_t, targets, scaled_anchors, in_h, in_w, noobj_mask)
        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()

            noobj_mask_t = noobj_mask_t.cuda()
        # --------------------------------------------------------------------------#
        #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
        #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        #   使用iou损失时，大中小目标的回归损失不存在比例失衡问题，故弃用
        # --------------------------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale

        loss = 0
        loss_t = 0
        Lreg = 0
        Lcls = 0
        hard_student_conf_loss = 0
        Toall_loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)  # 有多少正样本
        if n != 0:
            # ---------------------------------------------------------------#
            #   计算预测结果和真实结果的ciou
            # ----------------------------------------------------------------#
            ciou = self.box_ciou(pred_boxes, y_true[..., :4])

            # loss_loc    = torch.mean((1 - ciou)[obj_mask] * box_loss_scale[obj_mask])
            loss_loc = torch.mean((1 - ciou)[obj_mask])  # 边界回归
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))  # 分类回归(只是判断有没有目标)

            ciou_t = self.box_ciou(pred_boxes_t, y_true[..., :4])  # 教师网络CIOU
            loss_loc_t = torch.mean((1 - ciou_t)[obj_mask])  # 教师网络的边界框回归
            loss_cls_t = torch.mean(self.BCELoss(pred_cls_t[obj_mask], y_true[..., 5:][obj_mask]))  # 教师网络分类回归

            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            loss_t += loss_loc_t * self.box_ratio + loss_cls_t * self.cls_ratio

            # --------------------------------------------------
            #               知识蒸馏之边界框回归
            # --------------------------------------------------
            if loss_loc > loss_loc_t:
                Lb_loss = loss_loc
            else:
                Lb_loss = 0
            Lreg += loss_loc + 0.5*Lb_loss


        if self.focal_loss:
            ratio = torch.where(obj_mask, torch.ones_like(conf) * self.alpha,
                                torch.ones_like(conf) * (1 - self.alpha)) * torch.where(obj_mask,
                                                                                        torch.ones_like(conf) - conf,
                                                                                        conf) ** self.gamma
            loss_conf = torch.mean((self.BCELoss(conf, obj_mask.type_as(conf)) * ratio)[noobj_mask.bool() | obj_mask])
            loss_conf_t = torch.mean((self.BCELoss(conf_t, obj_mask.type_as(conf_t)) * ratio)[noobj_mask_t.bool() | obj_mask])# 教师置信度回归
        else:
            loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])  # 置信度回归
            loss_conf_t = torch.mean(self.BCELoss(conf_t, obj_mask.type_as(conf_t))[noobj_mask_t.bool() | obj_mask])  # 教师置信度回归
        #loss += loss_conf * self.balance[l] * self.obj_ratio
        loss_t += loss_conf_t * self.balance[l] * self.obj_ratio

        # ---------------------------------------------------
        #                conf知识蒸馏
        # --------------------------------------------------
        hard_student_conf_loss += loss_conf * self.balance[l] * self.obj_ratio
        #conf_soft_loss = F.cross_entropy(conf, conf_t, size_average=False)
        #soft_loss = KL_div(pred_cls_t, pred_cls)
        soft_loss = nn.KLDivLoss()(pred_cls, pred_cls_t)
        Lcls += 0.3*hard_student_conf_loss + 0.7*soft_loss
        #return loss, loss_t

        Toall_loss += Lreg + Lcls
        return Toall_loss

    def calculate_iou(self, _box_a, _box_b):
        # -----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        # -----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        # -----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        # -----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        # -----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        # -----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # -----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        # -----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        # -----------------------------------------------------------#
        #   计算交的面积
        # -----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        # -----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        # -----------------------------------------------------------#
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        # -----------------------------------------------------------#
        #   求IOU
        # -----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def get_target(self, l, targets, anchors, in_h, in_w):
        # 特征图索引:0,1,2,
        # targets:列表形式，长度batchsize,每个元素又是5列，边界框信息和类别
        # scaled_anchors：缩放到特征层的anchor
        # in_h, in_w:特征层大小
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)
        # -----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        # -----------------------------------------------------#

        # 创建一个(batshzie,3,13,13)全1矩阵
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   让网络更加去关注小目标
        #   创建一个(batshzie,3,13,13)全0矩阵
        # -----------------------------------------------------#
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   y_true用来存放ground truth 信息
        # -----------------------------------------------------#
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])

            # -------------------------------------------------------#
            #   计算出正样本在特征层上的中心点?
            # -------------------------------------------------------#
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # -------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            # -------------------------------------------------------#
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            # -------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4
            # -------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            # -------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   best_ns:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            # -------------------------------------------------------#
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                # ----------------------------------------#
                #   判断这个先验框是当前特征点的哪一个先验框
                #  [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                # ----------------------------------------#

                k = self.anchors_mask[l].index(best_n)
                # ----------------------------------------#
                #   获得真实框属于哪个网格点
                #  floor 返回一个新张量，包含输入input张量每个元素的floor，即取不大于元素的最大整数。
                # ----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                # ----------------------------------------#
                #   取出真实框的种类
                # ----------------------------------------#
                c = batch_target[t, 4].long()
                # ----------------------------------------#
                #   noobj_mask代表无目标的特征点
                # ----------------------------------------#
                noobj_mask[b, k, j, i] = 0
                # ----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                #   y_true[b, k, j, i, 0] 意思是取第几各个batch的第k个锚框(每个层有三个)，在特征层上第j行第i列网格点
                # ----------------------------------------#
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]  # 前几行这是box的信息
                y_true[b, k, j, i, 4] = 1  # bbox由5个信息组成，前四个是坐标信息，第5个是Pc，是否由目标
                y_true[b, k, j, i, c + 5] = 1  # c + 5指定获得的第几个类，5是前5个维度(x,y,w,h,Pc),从下个维度开始是类
                # ----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                # ----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # -----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        # -----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            # -------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            # -------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                # -------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                # -------------------------------------------------------#
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4]
                # -------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                # -------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                # -------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                # -------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
