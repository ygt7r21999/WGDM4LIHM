import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append('../')

import torch
from torch import optim
from skimage.io import imsave
from ASM_Prop import *
from lossfunc import *
from utils import *

device = torch.device("cuda")
lossTypeList = ['L2', 'LPIPS', 'L2_tv', 'L2_ctv', 'LPIPS_ctv']
lossType = lossTypeList[1]
# lossType = lossTypeList[4]

objTypeList = ['Amp', 'Pha', 'Combined']
objType = objTypeList[1]

savedir = './results/' + objType + '-' + lossType + '/'
saveEpoch = 10
# saveEpoch = 100

# 定义存储路径
savedir_abs = savedir + "abs"
savedir_pha = savedir + "pha"
os.makedirs(savedir_abs, exist_ok=savedir_abs)
os.makedirs(savedir_pha, exist_ok=savedir_pha)



"""Amp"""
# Int_o_gt = datai("I0.bmp").to(device)
# Amp_o_gt = torch.sqrt(Int_o_gt)
# Pha_o_gt = torch.zeros_like(Amp_o_gt).to(device)
"""Pha"""
Int_o_gt = datai("I0.bmp").to(device)
Amp_o_gt = torch.ones_like(Int_o_gt)
Pha_o_gt = datai("P0.bmp").to(device)
"""Combined"""
# Int_o_gt = datai("I0.bmp").to(device)
# Amp_o_gt = torch.sqrt(Int_o_gt)
# Pha_o_gt = datai("P0.bmp").to(device)

datao(Amp_o_gt)
datao(Pha_o_gt)

psize = 1.34e-3  # unit:mm
dist = 1.21
lamda = 0.534e-3

# 前向传播
Uo_gt_r = Amp_o_gt*torch.cos(Pha_o_gt)
Uo_gt_i = Amp_o_gt*torch.sin(Pha_o_gt)
Ui_gt = ASM_Prop_gpu(Uo_gt_r, Uo_gt_i, psize, dist, lamda, device)

# -----------------------------------------------------------------------
Amp_i_gt = torch.abs(Ui_gt)
datao(Amp_i_gt)

# 超参数设置
Max_iter = 5000
alpha = 0.01
lr = alpha

# 目标变量初始化
# 初始化1：全1
# Uo_pred_r = torch.ones_like(Amp_i_gt)
# Uo_pred_i = torch.ones_like(Amp_i_gt)
# 初始化2：逆衍射
Ui_idiff = ASM_Prop_gpu(Amp_i_gt*torch.cos(torch.zeros_like(Amp_i_gt)),\
                        Amp_i_gt*torch.sin(torch.zeros_like(Amp_i_gt)), psize, -dist, lamda, device)
Uo_pred_r = torch.real(Ui_idiff)
Uo_pred_i = torch.imag(Ui_idiff)

Uo_pred_r.requires_grad = True
Uo_pred_i.requires_grad = True

# # 自动微分族定义
LPIPSLoss = PerceptualLoss().to(device)
optimizer = optim.Adam([Uo_pred_r, \
                        Uo_pred_i], lr=lr)

for iter in range(Max_iter):
    # 1.Uo_pred前向传播
    Ui_pred = ASM_Prop_gpu(Uo_pred_r, Uo_pred_i, psize, dist, lamda, device)

    # 2.loss计算
    loss = losscompute(Amp_i_gt, Ui_pred, Uo_pred_r, Uo_pred_i, lossType, LPIPSLoss)

    # 3.Uo_pred更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (iter+1) % saveEpoch == 0:
        print(iter+1, loss.data)
        Uo_pred_Amp = (Uo_pred_r**2 + Uo_pred_i**2)**0.5
        Uo_pred_Pha = torch.atan2(Uo_pred_i, Uo_pred_r)
        # datao(Uo_pred_Amp)
        imsave(savedir_abs + "/abs_"+str(iter+1)+".png",datas(Uo_pred_Amp))
        imsave(savedir_pha + "/pha_"+str(iter+1)+".png",datas(Uo_pred_Pha))

