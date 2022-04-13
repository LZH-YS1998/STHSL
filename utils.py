import numpy as np
import torch
import torch.nn.functional as F
from Params import args
import random

def cal_loss_r(preds, labels, mask):
    loss = torch.sum(torch.square(preds - labels) * mask) / torch.sum(mask)
    return loss

def cal_metrics_r(preds, labels, mask):
    loss = np.sum(np.square(preds - labels) * mask) / np.sum(mask)
    sqLoss = np.sum(np.sum(np.square(preds - labels) * mask, axis=0), axis=0)
    absLoss = np.sum(np.sum(np.abs(preds - labels) * mask, axis=0), axis=0)
    tstNums = np.sum(np.sum(mask, axis=0), axis=0)
    posMask = mask * np.greater(labels, 0.5)
    apeLoss = np.sum(np.sum(np.abs(preds - labels) / (labels + 1e-8) * posMask, axis=0), axis=0)
    posNums = np.sum(np.sum(posMask, axis=0), axis=0)
    return loss, sqLoss, absLoss, tstNums, apeLoss, posNums

def cal_metrics_r_mask(preds, labels, mask, mask_sparsity):
    loss = np.sum(np.square(preds - labels) * mask) / np.sum(mask)
    sqLoss = np.sum(np.sum(np.square(preds - labels) * mask * mask_sparsity, axis=0), axis=0)
    absLoss = np.sum(np.sum(np.abs(preds - labels) * mask * mask_sparsity, axis=0), axis=0)
    tstNums = np.sum(np.sum(mask * mask_sparsity, axis=0), axis=0)
    posMask = mask * mask_sparsity * np.greater(labels, 0.5)
    apeLoss = np.sum(np.sum(np.abs(preds - labels) / (labels + 1e-8) * posMask, axis=0), axis=0)
    posNums = np.sum(np.sum(posMask, axis=0), axis=0)
    return loss, sqLoss, absLoss, tstNums, apeLoss, posNums

def Informax_loss(DGI_pred, DGI_labels):
    BCE_loss = torch.nn.BCEWithLogitsLoss()
    loss = BCE_loss(DGI_pred, DGI_labels)
    return loss

def infoNCEloss(q, k):
    T = args.t
    q = q.expand_as(k)
    q = q.permute(0, 3, 4, 2, 1)
    k = k.permute(0, 3, 4, 2, 1)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    pos_sim = torch.sum(torch.mul(q, k), dim=-1)
    neg_sim = torch.matmul(q, k.transpose(-1, -2))
    pos = torch.exp(torch.div(pos_sim, T))
    neg = torch.sum(torch.exp(torch.div(neg_sim, T)), dim=-1)
    denominator = neg + pos
    return torch.mean(-torch.log(torch.div(pos, denominator)))


def seed_torch(seed=523):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def makePrint(name, ep, reses):
    ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
    for metric in reses:
        val = reses[metric]
        ret += '%s = %.4f, ' % (metric, val)
    ret = ret[:-2] + '  '
    return ret