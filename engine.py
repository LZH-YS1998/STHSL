import torch
import torch.optim as optim
from model import *
import numpy as np
import utils
from Params import args
from DataHandler import DataHandler


class trainer():
    def __init__(self, device):
        self.handler = DataHandler()
        self.model = STHSL()
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = utils.cal_loss_r
        self.metrics = utils.cal_metrics_r

    def sampleTrainBatch(self, batIds, st, ed):
        batch = ed - st
        idx = batIds[0: batch]
        label = self.handler.trnT[:, idx, :]
        label = np.transpose(label, [1, 0, 2])
        retLabels = (label >= 0) * 1
        mask = retLabels
        retLabels = label

        feat_list = []
        for i in range(batch):
            feat_one = self.handler.trnT[:, idx[i] - args.temporalRange: idx[i], :]
            feat_one = np.expand_dims(feat_one, axis=0)
            feat_list.append(feat_one)
        feat_batch = np.concatenate(feat_list, axis=0)
        return self.handler.zScore(feat_batch), retLabels, mask

    def sampTestBatch(self, batIds, st, ed, tstTensor, inpTensor):
        batch = ed - st
        idx = batIds[0: batch]
        label = tstTensor[:, idx, :]
        label = np.transpose(label, [1, 0, 2])
        retLabels = label
        mask = 1 * (label > 0)

        feat_list = []
        for i in range(batch):
            if idx[i] - args.temporalRange < 0:
                temT = inpTensor[:, idx[i] - args.temporalRange:, :]
                temT2 = tstTensor[:, :idx[i], :]
                feat_one = np.concatenate([temT, temT2], axis=1)
            else:
                feat_one = tstTensor[:, idx[i] - args.temporalRange: idx[i], :]
            feat_one = np.expand_dims(feat_one, axis=0)
            feat_list.append(feat_one)
        feats = np.concatenate(feat_list, axis=0)
        return self.handler.zScore(feats), retLabels, mask


    def train(self):
        self.model.train()
        ids = np.random.permutation(list(range(args.temporalRange, args.trnDays)))
        epochLoss, epochPreLoss, epochAcc = [0] * 3
        num = len(ids)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = ids[st: ed]
            bt = ed - st

            Infomax_L1 = torch.ones(bt, args.offNum, args.areaNum)
            Infomax_L2 = torch.zeros(bt, args.offNum, args.areaNum)
            Infomax_labels = torch.Tensor(torch.cat((Infomax_L1, Infomax_L2), -1)).to(args.device)

            tem = self.sampleTrainBatch(batIds, st, ed)
            feats, labels, mask = tem
            mask = torch.Tensor(mask).to(args.device)
            self.optimizer.zero_grad()

            idx = np.random.permutation(args.areaNum)
            DGI_feats = torch.Tensor(feats[:, idx, :, :]).to(args.device)
            feats = torch.Tensor(feats).to(args.device)
            labels = torch.Tensor(labels).to(args.device)

            out_local, eb_local, eb_global, Infomax_pred, out_global = self.model(feats, DGI_feats)
            out_local = self.handler.zInverse(out_local)
            out_global = self.handler.zInverse(out_global)
            loss = (utils.Informax_loss(Infomax_pred, Infomax_labels) * args.ir) + (utils.infoNCEloss(eb_global, eb_local) * args.cr) + \
                   self.loss(out_local, labels, mask) + self.loss(out_global, labels, mask)

            loss.backward()
            self.optimizer.step()
            print('Step %d/%d: preLoss = %.4f         ' % (i, steps, loss), end='\r')
            epochLoss += loss
        epochLoss = epochLoss / steps
        return epochLoss, loss.item()


    def eval(self, iseval, isSparsity):
        self.model.eval()
        if iseval:
            ids = np.array(list(range(self.handler.valT.shape[1])))
        else:
            ids = np.array(list(range(self.handler.tstT.shape[1])))
        epochLoss, epochPreLoss, = [0] * 2

        num = len(ids)
        if isSparsity:
            epochSqLoss1, epochAbsLoss1, epochTstNum1, epochApeLoss1, epochPosNums1 = [np.zeros(4) for i in range(5)]
            epochSqLoss2, epochAbsLoss2, epochTstNum2, epochApeLoss2, epochPosNums2 = [np.zeros(4) for i in range(5)]
            epochSqLoss3, epochAbsLoss3, epochTstNum3, epochApeLoss3, epochPosNums3 = [np.zeros(4) for i in range(5)]
            epochSqLoss4, epochAbsLoss4, epochTstNum4, epochApeLoss4, epochPosNums4 = [np.zeros(4) for i in range(5)]
            epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]
        else:
            epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]

        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = ids[st: ed]

            if iseval:
                tem = self.sampTestBatch(batIds, st, ed, self.handler.valT, self.handler.trnT)
            else:
                tem = self.sampTestBatch(batIds, st, ed, self.handler.tstT, np.concatenate([self.handler.trnT, self.handler.valT], axis=1))
            feats, labels, mask = tem
            idx = np.random.permutation(args.areaNum)
            shuf_feats = feats[:, idx, :, :]
            feats = torch.Tensor(feats).to(args.device)
            shuf_feats = torch.Tensor(shuf_feats).to(args.device)
            out_local, eb_local, eb_global, DGI_pred, out_global = self.model(feats, shuf_feats)

            if isSparsity:
                output = self.handler.zInverse(out_global)
                _, sqLoss1, absLoss1, tstNums1, apeLoss1, posNums1 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask1)
                _, sqLoss2, absLoss2, tstNums2, apeLoss2, posNums2 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask2)
                _, sqLoss3, absLoss3, tstNums3, apeLoss3, posNums3 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask3)
                _, sqLoss4, absLoss4, tstNums4, apeLoss4, posNums4 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask4)

                loss, sqLoss, absLoss, tstNums, apeLoss, posNums = self.metrics(output.cpu().detach().numpy(), labels, mask)
                epochSqLoss += sqLoss
                epochAbsLoss += absLoss
                epochTstNum += tstNums
                epochApeLoss += apeLoss
                epochPosNums += posNums

                epochSqLoss1 += sqLoss1
                epochAbsLoss1 += absLoss1
                epochTstNum1 += tstNums1
                epochApeLoss1 += apeLoss1
                epochPosNums1 += posNums1

                epochSqLoss2 += sqLoss2
                epochAbsLoss2 += absLoss2
                epochTstNum2 += tstNums2
                epochApeLoss2 += apeLoss2
                epochPosNums2 += posNums2

                epochSqLoss3 += sqLoss3
                epochAbsLoss3 += absLoss3
                epochTstNum3 += tstNums3
                epochApeLoss3 += apeLoss3
                epochPosNums3 += posNums3

                epochSqLoss4 += sqLoss4
                epochAbsLoss4 += absLoss4
                epochTstNum4 += tstNums4
                epochApeLoss4 += apeLoss4
                epochPosNums4 += posNums4
                epochLoss += loss
                print('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, loss), end='\r')
            else:
                loss, sqLoss, absLoss, tstNums, apeLoss, posNums = self.metrics(output.cpu().detach().numpy(), labels, mask)
                epochSqLoss += sqLoss
                epochAbsLoss += absLoss
                epochTstNum += tstNums
                epochApeLoss += apeLoss
                epochPosNums += posNums
                epochLoss += loss
                print('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, loss), end='\r')
        epochLoss = epochLoss / steps
        ret = dict()

        if isSparsity == False:
            for i in range(args.offNum):
                ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
                ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
                ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]
            ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
            ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
            ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)
            ret['epochLoss'] = epochLoss
        else:
            ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
            ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
            ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)
            for i in range(args.offNum):
                ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
                ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
                ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]

            ret['RMSE_mask_1'] = np.sqrt(np.sum(epochSqLoss1) / np.sum(epochTstNum1))
            ret['MAE_mask_1'] = np.sum(epochAbsLoss1) / np.sum(epochTstNum1)
            ret['MAPE_mask_1'] = np.sum(epochApeLoss1) / np.sum(epochPosNums1)

            ret['RMSE_mask_2'] = np.sqrt(np.sum(epochSqLoss2) / np.sum(epochTstNum2))
            ret['MAE_mask_2'] = np.sum(epochAbsLoss2) / np.sum(epochTstNum2)
            ret['MAPE_mask_2'] = np.sum(epochApeLoss2) / np.sum(epochPosNums2)

            ret['RMSE_mask_3'] = np.sqrt(np.sum(epochSqLoss3) / np.sum(epochTstNum3))
            ret['MAE_mask_3'] = np.sum(epochAbsLoss3) / np.sum(epochTstNum3)
            ret['MAPE_mask_3'] = np.sum(epochApeLoss3) / np.sum(epochPosNums3)

            ret['RMSE_mask_4'] = np.sqrt(np.sum(epochSqLoss4) / np.sum(epochTstNum4))
            ret['MAE_mask_4'] = np.sum(epochAbsLoss4) / np.sum(epochTstNum4)
            ret['MAPE_mask_4'] = np.sum(epochApeLoss4) / np.sum(epochPosNums4)
            ret['epochLoss'] = epochLoss

        return ret


def sampleTestBatch(batIds, st, ed, tstTensor, inpTensor, handler):
    batch = ed - st
    idx = batIds[0: batch]
    label = tstTensor[:, idx, :]
    label = np.transpose(label, [1, 0, 2])
    retLabels = label
    mask = handler.tstLocs * (label > 0)

    feat_list = []
    for i in range(batch):
        if idx[i] - args.temporalRange < 0:
            temT = inpTensor[:, idx[i] - args.temporalRange:, :]
            temT2 = tstTensor[:, :idx[i], :]
            feat_one = np.concatenate([temT, temT2], axis=1)
        else:
            feat_one = tstTensor[:, idx[i] - args.temporalRange: idx[i], :]
        feat_one = np.expand_dims(feat_one, axis=0)
        feat_list.append(feat_one)
    feats = np.concatenate(feat_list, axis=0)
    return handler.zScore(feats), retLabels, mask,


def test(model, handler):
    ids = np.array(list(range(handler.tstT.shape[1])))
    epochLoss, epochPreLoss, = [0] * 2
    epochSqLoss1, epochAbsLoss1, epochTstNum1, epochApeLoss1, epochPosNums1 = [np.zeros(4) for i in range(5)]
    epochSqLoss2, epochAbsLoss2, epochTstNum2, epochApeLoss2, epochPosNums2 = [np.zeros(4) for i in range(5)]
    epochSqLoss3, epochAbsLoss3, epochTstNum3, epochApeLoss3, epochPosNums3 = [np.zeros(4) for i in range(5)]
    epochSqLoss4, epochAbsLoss4, epochTstNum4, epochApeLoss4, epochPosNums4 = [np.zeros(4) for i in range(5)]
    epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]
    num = len(ids)

    steps = int(np.ceil(num / args.batch))
    for i in range(steps):
        st = i * args.batch
        ed = min((i + 1) * args.batch, num)
        batIds = ids[st: ed]

        tem = sampleTestBatch(batIds, st, ed, handler.tstT, np.concatenate([handler.trnT, handler.valT], axis=1), handler)
        feats, labels, mask = tem
        feats = torch.Tensor(feats).to(args.device)
        idx = np.random.permutation(args.areaNum)
        shuf_feats = feats[:, idx, :, :]

        out_local, eb_local, eb_global, DGI_pred, out_global = model(feats, shuf_feats)
        output = handler.zInverse(out_global)

        _, sqLoss1, absLoss1, tstNums1, apeLoss1, posNums1 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask1)
        _, sqLoss2, absLoss2, tstNums2, apeLoss2, posNums2 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask2)
        _, sqLoss3, absLoss3, tstNums3, apeLoss3, posNums3 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask3)
        _, sqLoss4, absLoss4, tstNums4, apeLoss4, posNums4 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask4)

        loss, sqLoss, absLoss, tstNums, apeLoss, posNums = utils.cal_metrics_r(output.cpu().detach().numpy(), labels, mask)
        epochSqLoss += sqLoss
        epochAbsLoss += absLoss
        epochTstNum += tstNums
        epochApeLoss += apeLoss
        epochPosNums += posNums

        epochSqLoss1 += sqLoss1
        epochAbsLoss1 += absLoss1
        epochTstNum1 += tstNums1
        epochApeLoss1 += apeLoss1
        epochPosNums1 += posNums1

        epochSqLoss2 += sqLoss2
        epochAbsLoss2 += absLoss2
        epochTstNum2 += tstNums2
        epochApeLoss2 += apeLoss2
        epochPosNums2 += posNums2

        epochSqLoss3 += sqLoss3
        epochAbsLoss3 += absLoss3
        epochTstNum3 += tstNums3
        epochApeLoss3 += apeLoss3
        epochPosNums3 += posNums3

        epochSqLoss4 += sqLoss4
        epochAbsLoss4 += absLoss4
        epochTstNum4 += tstNums4
        epochApeLoss4 += apeLoss4
        epochPosNums4 += posNums4

        epochLoss += loss
        print('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, loss), end='\r')
    ret = dict()

    ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
    ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
    ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)

    for i in range(args.offNum):
        ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
        ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
        ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]


    ret['RMSE_mask_1'] = np.sqrt(np.sum(epochSqLoss1) / np.sum(epochTstNum1))
    ret['MAE_mask_1'] = np.sum(epochAbsLoss1) / np.sum(epochTstNum1)
    ret['MAPE_mask_1'] = np.sum(epochApeLoss1) / np.sum(epochPosNums1)

    ret['RMSE_mask_2'] = np.sqrt(np.sum(epochSqLoss2) / np.sum(epochTstNum2))
    ret['MAE_mask_2'] = np.sum(epochAbsLoss2) / np.sum(epochTstNum2)
    ret['MAPE_mask_2'] = np.sum(epochApeLoss2) / np.sum(epochPosNums2)

    ret['RMSE_mask_3'] = np.sqrt(np.sum(epochSqLoss3) / np.sum(epochTstNum3))
    ret['MAE_mask_3'] = np.sum(epochAbsLoss3) / np.sum(epochTstNum3)
    ret['MAPE_mask_3'] = np.sum(epochApeLoss3) / np.sum(epochPosNums3)

    ret['RMSE_mask_4'] = np.sqrt(np.sum(epochSqLoss4) / np.sum(epochTstNum4))
    ret['MAE_mask_4'] = np.sum(epochAbsLoss4) / np.sum(epochTstNum4)
    ret['MAPE_mask_4'] = np.sum(epochApeLoss4) / np.sum(epochPosNums4)
    ret['epochLoss'] = epochLoss

    return ret