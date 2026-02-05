import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import *

import math
import numpy as np

def cosine_decay(current_epoch, decay_start_epoch, solver_epoch):
    return 0.5 * (1 + math.cos((current_epoch - decay_start_epoch) / (solver_epoch - decay_start_epoch) * math.pi))

# 指数衰减函数
def exponential_decay(current_epoch, decay_start_epoch, solver_epoch, beta):
    # 计算自变量的归一化值
    t = (current_epoch - decay_start_epoch) / (solver_epoch - decay_start_epoch)
    return math.exp(-beta * t)

# 线性衰减函数
def linear_decay(current_epoch, decay_start_epoch, solver_epoch, beta):
    # 计算自变量的归一化值
    t = (current_epoch - decay_start_epoch) / (solver_epoch - decay_start_epoch)
    return max(1 - beta * t, 0)

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    ) # [64, 100]
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )


    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False) # size_average=False表示小批量样本上的各样本kl损失这和
        * (temperature**2)
        / target.shape[0]
    )

    return alpha * tckd_loss + beta * nckd_loss, alpha * tckd_loss, beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

##########################################################################################

def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1/dim*torch.ones_like(w1, device=w1.device, dtype=w1.dtype) # [batch,N,1]
    K = torch.exp(-cost / reg)
    Kt= K.transpose(2, 1)
    for i in range(max_iter):
        v=w2/(torch.bmm(Kt,u)+1e-8) #[batch,N,1]
        u=w1/(torch.bmm(K,v)+1e-8)  #[batch,N,1]

    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow
        

def wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix=None, sinkhorn_lambda=25, sinkhorn_iter=30):
    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)
    # 代价矩阵
    cost_matrix = F.relu(cost_matrix) + 1e-8 # [100, 100]
    cost_matrix = cost_matrix.to(pred_student.device)
    
    # flow shape [bxnxn]  [64,99,99]
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return ws_distance


def wkd_logit_loss_with_speration(logits_student, logits_teacher, gt_label, temperature, gamma, cost_matrix=None, sinkhorn_lambda=0.05, sinkhorn_iter=10):
        
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

    # N*class
    N, c = logits_student.shape
    s_i = F.log_softmax(logits_student, dim=1)
    t_i = F.softmax(logits_teacher, dim=1)
    s_t = torch.gather(s_i, 1, label) # 按照label筛选出s_i中的目标类概率
    t_t = torch.gather(t_i, 1, label).detach()
    loss_t = - (t_t * s_t).mean() # 目标类
    # 剩下非目标类,mask非目标类处为0,目标类处为1
    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_student = logits_student[mask].reshape(N, -1) # 取出所有的非目标类logit，再reshape
    logits_teacher = logits_teacher[mask].reshape(N, -1) # 取出所有的非目标类logit，再reshape
    
    cost_matrix = cost_matrix.repeat(N, 1, 1) # 将[100,100]转化成[64,100,100]
    gd_mask = mask.unsqueeze(1) * mask.unsqueeze(2) # [64, 100, 100]
    cost_matrix = cost_matrix[gd_mask].reshape(N, c-1, c-1) # 取出非目标类的cost_matrix [64, 99]，重复64次
        
    # N*class
    loss_wkd = wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix, sinkhorn_lambda, sinkhorn_iter)

    return loss_t + gamma * loss_wkd, loss_t, gamma * loss_wkd


def adaptive_avg_std_pool2d(input_tensor, out_size=(1, 1), eps=1e-5):
    def start_index(a, b, c):
        return int(np.floor(a * c / b))
    def end_index(a, b, c):
        return int(np.ceil((a+1) * c / b))

    b, c, isizeH, isizeW = input_tensor.shape
    if len(out_size) == 2:
        osizeH, osizeW = out_size
    else:
        osizeH = osizeW = out_size

    avg_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # cov_pooled_tensor = torch.zeros((b, c*c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    cov_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # block_list = []
    for oh in range(osizeH):
        istartH = start_index(oh, osizeH, isizeH)
        iendH = end_index(oh, osizeH, isizeH)
        kH = iendH - istartH
        for ow in range(osizeW):
            istartW = start_index(ow, osizeW, isizeW)
            iendW = end_index(ow, osizeW, isizeW)
            kW = iendW - istartW

            # avg pool2d
            input_block = input_tensor[:, :, istartH:iendH, istartW:iendW]
            avg_pooled_tensor[:, :, oh, ow] = input_block.mean(dim=(-1, -2))
            # diagonal cov pool2d
            cov_pooled_tensor[:, :, oh, ow] = torch.sqrt(input_block.var(dim=(-1, -2)) + eps)
    
    return avg_pooled_tensor, cov_pooled_tensor

# res8x4, f_s: torch.Size([64, 256, 8, 8])    res32x4, f_t: torch.Size([64, 256, 8, 8])
def wkd_feature_loss(f_s, f_t, eps=1e-5, grid=1):
    if grid == 1:
        f_s_avg, f_t_avg = f_s.mean(dim=(-1,-2)), f_t.mean(dim=(-1,-2))
        f_s_std, f_t_std = torch.sqrt(f_s.var(dim=(-1,-2)) + eps), torch.sqrt(f_t.var(dim=(-1,-2)) + eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / f_s.size(0) # torch的mse_loss带有平方
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / f_s.size(0)
    elif grid > 1:
        f_s_avg, f_s_std = adaptive_avg_std_pool2d(f_s, out_size=(grid, grid), eps=eps)
        f_t_avg, f_t_std = adaptive_avg_std_pool2d(f_t, out_size=(grid, grid), eps=eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / (grid**2 * f_s.size(0))
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / (grid**2 * f_s.size(0))
    # 95.5241, 77.5059
    return mean_loss, cov_loss

def kl_feature_loss(f_s, f_t, eps):
    # 计算 f_s 和 f_t 的均值和标准差
    f_s_avg, f_t_avg = f_s.mean(dim=(-1, -2)), f_t.mean(dim=(-1, -2))  # 均值
    f_s_std, f_t_std = torch.sqrt(f_s.var(dim=(-1, -2)) + eps), torch.sqrt(f_t.var(dim=(-1, -2)) + eps)  # sqrt是求平方根，标准差
    # f_s_std, f_t_std = f_s.var(dim=(-1, -2)), f_t.var(dim=(-1, -2))  # sqrt是求平方根，标准差

    # 提取均值和方差
    mu_s, mu_t = f_s_avg, f_t_avg # [64, 256]
    delta_s, delta_t = f_s_std, f_t_std # [64, 256]

    # 根据公式计算 KL 散度损失, 57925.2070
    kl_loss = 0.5 * torch.sum(
        ((mu_t - mu_s) / delta_s) ** 2 + (delta_t / delta_s) ** 2 - 2 * (torch.log(delta_t) - torch.log(delta_s)) - 1)

    # 根据公式计算对称 KL 散度损失
    # kl_sym_loss = 0.5 * torch.sum(((mu_t - mu_s) ** 2) * ((1 / delta_t ** 2) + (1 / delta_s ** 2)) +
    #                               (delta_s / delta_t) ** 2 + (delta_t / delta_s) ** 2 - 2)

    # 对 KL 损失进行归一化, 905.0814
    kl_loss = kl_loss / f_s.size(0)

    return kl_loss


class WKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(WKD, self).__init__(student, teacher)
        self.cfg = cfg

        self.ce_loss_weight = cfg.WKD.LOSS.CE_WEIGHT
        self.wkd_logit_loss_weight = cfg.WKD.LOSS.WKD_LOGIT_WEIGHT
        self.wkd_feature_loss_weight = cfg.WKD.LOSS.WKD_FEAT_WEIGHT
        self.loss_cosine_decay_epoch = cfg.WKD.LOSS.COSINE_DECAY_EPOCH
        self.kl_weight = cfg.WKD.LOSS.KL_WEIGHT

        self.enable_wkdl = self.wkd_logit_loss_weight > 0
        self.enable_wkdf = self.wkd_feature_loss_weight > 0

        # 以下是获取梯度
        self.grad_accumulator = {
            "grad_in": 0,
            "grad_out": 0,
        }
        def hook_fun(module, grad_input, grad_output):
            # import pdb
            # pdb.set_trace()
            self.grad_accumulator["grad_out"] = grad_output[0]

        if self.student.__class__.__name__ in ['VGG', 'MobileNetV2']:
            self.student.classifier.register_backward_hook(hook_fun)
        elif self.student.__class__.__name__ in ['ShuffleNet', 'ShuffleNetV2']:
            self.student.linear.register_backward_hook(hook_fun)
        else: # resnet or wrn
            self.student.fc.register_backward_hook(hook_fun)


        # WKD-L: WD for logits distillation  这里是初始化代价矩阵
        if self.enable_wkdl:
            self.temperature = cfg.WKD.TEMPERATURE
            self.sinkhorn_lambda = cfg.WKD.SINKHORN.LAMBDA
            self.sinkhorn_iter = cfg.WKD.SINKHORN.ITER

            if cfg.WKD.COST_MATRIX == "fc":
                print("Using fc weight of teacher model as category prototype")
                self.prototype = self.teacher.fc.weight
                # caluate cosine similarity
                proto_normed = F.normalize(self.prototype, p=2, dim=-1)
                cosine_sim = proto_normed.matmul(proto_normed.transpose(-1, -2))
                self.dist = 1 - cosine_sim
            else:
                print("Using "+cfg.WKD.COST_MATRIX+" as cost matrix")
                path_gd = cfg.WKD.COST_MATRIX_PATH
                self.dist = torch.load(path_gd).cuda().detach() # [100, 100], 移动距离
                
            if cfg.WKD.COST_MATRIX_SHARPEN != 0:
                print("Sharpen ", cfg.WKD.COST_MATRIX_SHARPEN)
                sim = torch.exp(-cfg.WKD.COST_MATRIX_SHARPEN*self.dist)
                self.dist = 1 - sim

        # WKD-F: WD for feature distillation
        if self.enable_wkdf:
            self.wkd_feature_mean_cov_ratio = cfg.WKD.MEAN_COV_RATIO
            self.eps = cfg.WKD.EPS

            feat_s_shapes, feat_t_shapes = get_feat_shapes(self.student, self.teacher, cfg.WKD.INPUT_SIZE)

            self.hint_layer = cfg.WKD.HINT_LAYER
            self.projector = cfg.WKD.PROJECTOR
            self.spatial_grid = cfg.WKD.SPATIAL_GRID
            if self.projector == "bottleneck":
                self.conv_reg = ConvRegBottleNeck(
                    feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer], c_hidden=256, use_relu=True, use_bn=True
                )
            elif self.projector == "conv1x1":
                self.conv_reg = ConvReg(
                    feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer], use_relu=True, use_bn=True
                )
            else:
                raise NotImplementedError(f"Unknown projector type: {self.projector}")

        self.teacher = self.teacher.eval()
        self.student = self.student.eval()

        # ablation for kl, wkd 可选：kl, dkd, wkd, dwkd, wkd_kl, dkwd_dkd
        self.distiller_name = "dwkd_dkd"

    def get_learnable_parameters(self):
        student_params = [v for k, v in self.student.named_parameters()]
        if self.enable_wkdf:
            return student_params + list(self.conv_reg.parameters())
        else:
            return student_params

    def get_extra_parameters(self):
        return 0

    def forward_train(self, image, target, **kwargs):
        with torch.cuda.amp.autocast():
            logits_student, feats_student = self.student(image)
            with torch.no_grad():
                logits_teacher, feats_teacher = self.teacher(image)

        logits_student = logits_student.to(torch.float32)
        logits_teacher = logits_teacher.to(torch.float32)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)  # loss_ce=4.6764

        # 从loss_cosine_decay_epoch=150开始，后续wkd的权重依次余弦衰减，最后1个epoch时wkd的权重已经为0，此时相当于只有交叉熵损失
        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
            # cosine decay
            # logits蒸馏
            self.wkd_logit_loss_weight_1 = 0.5 * self.wkd_logit_loss_weight * (1 + math.cos(
                (kwargs['epoch'] - decay_start_epoch) / (self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi))
            # feature蒸馏
            self.wkd_feature_loss_weight_1 = 0.5 * self.wkd_feature_loss_weight * (1 + math.cos(
                (kwargs['epoch'] - decay_start_epoch) / (self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi))
        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight
            self.wkd_feature_loss_weight_1 = self.wkd_feature_loss_weight

        if self.enable_wkdl:
            # (1)计算wkd      仅计算WKD,不乘以权重：0.0461
            loss_wkd_logit = wkd_logit_loss(logits_student, logits_teacher, self.temperature,
                                                   self.dist.repeat(logits_student.shape[0], 1, 1), self.sinkhorn_lambda,
                                                   self.sinkhorn_iter)

            # (2)计算dwkd, twkd, nwkd        22.7164, 4.6507, 18.0657
            loss_wkd_logit_dwkd, loss_twkd, loss_nwkd = wkd_logit_loss_with_speration(logits_student, logits_teacher,
                                                                                      target, self.temperature,
                                                                                      self.wkd_logit_loss_weight_1,
                                                                                      self.dist, self.sinkhorn_lambda,
                                                                                      self.sinkhorn_iter)

            # (3)计算 KL 散度  # 仅计算KL，不乘权重：KL=0.0831
            loss_kl = F.kl_div(
                input=F.log_softmax(logits_student / self.temperature, dim=-1),
                target=F.softmax(logits_teacher / self.temperature, dim=-1),
                reduction='batchmean'
            )

            # (4)仅计算DKD，不乘权重：dkd_loss=7.8047, loss_tckd=3.6631, loss_nckd=3.5354
            loss_dkd, loss_tckd, loss_nckd = dkd_loss(logits_student, logits_teacher, target, 1.0, 2.0, self.temperature)

            # 统一观察：kl,dkd,tckd,nckd,  wkd,dwkd,twkd,nwkd
            losses_vis = {
                "loss_kl": loss_kl,
                "loss_dkd": loss_dkd,
                "loss_tckd": loss_tckd,
                "loss_nckd": loss_nckd,

                "loss_wkd": loss_wkd_logit,
                "loss_dwkd": loss_wkd_logit_dwkd,
                "loss_twkd": loss_twkd,
                "loss_nwkd": loss_nwkd,
            }

            if self.distiller_name == "kl":
                losses_dict = {
                    "loss_ce": loss_ce, # 4.6764
                    "loss_kd": loss_kl * 10.0, # 0.0831
                }
            elif self.distiller_name == "dkd":
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_dkd, # 7.1985
                }
            elif self.distiller_name == "wkd":
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_wkd_logit * 10.0, # 0.0461
                }
            elif self.distiller_name == "dwkd":
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_wkd_logit_dwkd, # 22.7164
                }
            elif self.distiller_name == "wkd_kl":
                g_t = 0.5 * (1 + math.cos(kwargs['epoch'] / self.cfg.SOLVER.EPOCHS * math.pi)) # 从第0个epoch开始上升
                # g_t = 0.5 * (1 + math.cos((kwargs['epoch'] - decay_start_epoch)/(self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi)) # 从第decay_start_epoch个epoch开始上升
                loss_wkd_kl = loss_wkd_logit + (1 - g_t) * loss_kl * self.kl_weight
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_wkd_kl,
                }
            elif self.distiller_name == "dwkd_dkd":
                g_t = 0.5 * (1 + math.cos(kwargs['epoch'] / self.cfg.SOLVER.EPOCHS * math.pi)) # 从第0个epoch开始上升
                # g_t = 0.5 * (1 + math.cos((kwargs['epoch'] - decay_start_epoch)/(self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi)) # 从第decay_start_epoch个epoch开始上升
                loss_dwkd_dkd = loss_wkd_logit_dwkd + (1 - g_t) * loss_dkd * self.kl_weight
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_dwkd_dkd,
                }
            else:
                print("distiller method not implemented！")
                raise NotImplementedError

            return logits_student, logits_teacher, losses_dict, losses_vis

        if self.enable_wkdf:
            # WD for feature distillation
            if self.enable_wkdf:
                f_t = feats_teacher["feats"][self.hint_layer].to(torch.float32)
                f_s = feats_student["feats"][self.hint_layer].to(torch.float32)
                f_s = self.conv_reg(f_s)

                feat_loss_type = "use_wkd_kl_feat_loss" # use_wkd_feat_loss, use_kl_feat_loss, use_wkd_kl_feat_loss = 0, 1, 0
                if feat_loss_type == "use_wkd_feat_loss":
                    mean_loss, cov_loss = wkd_feature_loss(f_s, f_t, self.eps, grid=self.spatial_grid)
                    loss_wkd_feat = self.wkd_feature_mean_cov_ratio * mean_loss + cov_loss # 459.6024 = 4 * 95.5241 + 77.5049
                    loss_feat = self.wkd_feature_loss_weight_1 * loss_wkd_feat # 36.7582 = 0.08 * 459.6024

                elif feat_loss_type == "use_kl_feat_loss":
                    loss_kl_feat = kl_feature_loss(f_s, f_t, 1.0)
                    loss_feat = 0.04 * loss_kl_feat # 45.2541 = 0.05 * 905.0814
                    # loss_feat = F.mse_loss(f_s, f_t) * 8.0

                elif feat_loss_type == "use_wkd_kl_feat_loss":
                    mean_loss, cov_loss = wkd_feature_loss(f_s, f_t, self.eps, grid=self.spatial_grid)
                    loss_wkd_feat = self.wkd_feature_mean_cov_ratio * mean_loss + cov_loss # 459.6024 = 4 * 95.5241 + 77.5049
                    loss_wkd_feat = self.wkd_feature_loss_weight_1 * loss_wkd_feat # 36.7582 = 0.08 * 459.6024

                    loss_kl_feat = kl_feature_loss(f_s, f_t, 1.0)
                    loss_kl_feat = 0.04 * loss_kl_feat # 45.2541 = 0.05 * 905.0814

                    g_t = 0.5 * (1 + math.cos(kwargs['epoch'] / self.cfg.SOLVER.EPOCHS * math.pi))
                    loss_feat = loss_wkd_feat + (1 - g_t) * loss_kl_feat * self.kl_weight

                else:
                    print("The feature loss is not implemented")
                    raise NotImplementedError

                losses_dict = {
                    "loss_ce": loss_ce,  # 4.6764
                    "loss_kd": loss_feat,  # 0.0831
                }

                # 统一观察：kl,dkd,tckd,nckd,  wkd,dwkd,twkd,nwkd
                losses_vis = {
                    "loss_kl": torch.zeros([1], device="cuda"),
                    "loss_dkd": torch.zeros([1], device="cuda"),
                    "loss_tckd": torch.zeros([1], device="cuda"),
                    "loss_nckd": torch.zeros([1], device="cuda"),

                    "loss_wkd": torch.zeros([1], device="cuda"),
                    "loss_dwkd": torch.zeros([1], device="cuda"),
                    "loss_twkd": torch.zeros([1], device="cuda"),
                    "loss_nwkd": torch.zeros([1], device="cuda"),
                }
                return logits_student, logits_teacher, losses_dict, losses_vis



