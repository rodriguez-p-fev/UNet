import sys
import os
import os.path as osp
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from UNet import UNet
from tools import BCELoss
use_fp16 = True

def set_model(LOAD_MODEL, MODEL_PATH, DEVICE):
    model = UNet()
    if(LOAD_MODEL):
        print("model loading")
        state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(state_dict)
    model.to(device=DEVICE)
    model.train()
    criteria = BCELoss
    return model, criteria

def set_optimizer(model, lr_start, weight_decay):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = weight_decay
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': lr_start},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': lr_start},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': weight_decay},
        ]
    optim = torch.optim.Adam(
        params_list,
        lr=lr_start,
        weight_decay=weight_decay,
    )
    return optim

def train_batch_scaler(model, data, scaler, optim, criteria, DEVICE):
    model.train()
    im, lb = data
    im = im.to(device=DEVICE)
    lb = lb.float().unsqueeze(1).to(device=DEVICE)
    with torch.cuda.amp.autocast(enabled=use_fp16):
        logits = model(im)
        loss = criteria(logits, lb)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    #torch.cuda.synchronize()
    return loss
    
def train_batch(model, data, optim, criteria, DEVICE):
    model.train()
    im, lb = data
    im = im.to(device=DEVICE)
    lb = lb.float().unsqueeze(1).to(device=DEVICE)
    logits = model(im)
    optim.zero_grad()
    loss, acc = criteria(logits, lb)
    loss.backward()
    optim.step()
    return loss, acc
