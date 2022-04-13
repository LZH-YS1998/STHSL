import torch
import torch.nn as nn
from Params import args

# Local
# Local Spatial cnn
class spa_cnn_local(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super(spa_cnn_local, self).__init__()
        self.spaConv1 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv2 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv3 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv4 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.drop = nn.Dropout(args.dropRateL)
        self.act_lr = nn.LeakyReLU()

    def forward(self, embeds):
        cate_1 = self.drop(self.spaConv1(embeds))
        cate_2 = self.drop(self.spaConv2(embeds))
        cate_3 = self.drop(self.spaConv3(embeds))
        cate_4 = self.drop(self.spaConv4(embeds))
        spa_cate = torch.cat([cate_1, cate_2, cate_3, cate_4], dim=-1)
        return self.act_lr(spa_cate + embeds)

# Local Temporal cnn
class tem_cnn_local(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(tem_cnn_local, self).__init__()
        self.temConv1 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv2 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv3 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv4 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.act_lr = nn.LeakyReLU()
        self.drop = nn.Dropout(args.dropRateL)

    def forward(self, embeds):
        cate_1 = self.drop(self.temConv1(embeds))
        cate_2 = self.drop(self.temConv2(embeds))
        cate_3 = self.drop(self.temConv3(embeds))
        cate_4 = self.drop(self.temConv4(embeds))
        tem_cate = torch.cat([cate_1, cate_2, cate_3, cate_4], dim=-1)
        return self.act_lr(tem_cate + embeds)


# Global
# Global Hypergraph
class Hypergraph(nn.Module):
    def __init__(self):
        super(Hypergraph, self).__init__()
        self.adj = nn.Parameter(torch.Tensor(torch.randn([args.temporalRange, args.hyperNum, args.areaNum * args.cateNum])), requires_grad=True)
        self.Conv = nn.Conv3d(args.latdim, args.latdim, kernel_size=1)
        self.act1 = nn.LeakyReLU()

    def forward(self, embeds):
        adj = self.adj
        tpadj = adj.transpose(2, 1)
        embeds_cate = embeds.transpose(2, 3).contiguous().view(embeds.shape[0], args.latdim, args.temporalRange, -1)
        hyperEmbeds = self.act1(torch.einsum('thn,bdtn->bdth', adj, embeds_cate))
        retEmbeds = self.act1(torch.einsum('tnh,bdth->bdtn', tpadj, hyperEmbeds))
        retEmbeds = retEmbeds.view(embeds.shape[0], args.latdim, args.temporalRange, args.areaNum, args.cateNum).transpose(2, 3)
        return retEmbeds

# Hypergraph Infomax AvgReadout
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, embeds):
        return torch.mean(embeds, 2)

# Hypergraph Infomax Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(args.latdim, args.latdim, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, score, h_pos, h_neg):
        score = torch.unsqueeze(score, 2)
        score = score.expand_as(h_pos)
        score = score.transpose(1, 4).contiguous()
        h_pos = h_pos.transpose(1, 4).contiguous()
        h_neg = h_neg.transpose(1, 4).contiguous()
        sc_pos = torch.squeeze(self.f_k(h_pos, score), -1)
        sc_neg = torch.squeeze(self.f_k(h_neg, score), -1)
        logits = torch.cat((sc_pos.mean(-1), sc_neg.mean(-1)), dim=2)
        return logits

# Global Hypergraph Infomax
class Hypergraph_Infomax(nn.Module):
    def __init__(self):
        super(Hypergraph_Infomax, self).__init__()
        self.Hypergraph = Hypergraph()
        self.readout = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator()

    def forward(self, eb_pos, eb_neg):
        h_pos = self.Hypergraph(eb_pos)
        c = self.readout(h_pos)
        score = self.sigm(c)
        h_neg = self.Hypergraph(eb_neg)
        ret = self.disc(score, h_pos, h_neg)
        return h_pos, ret

# Global Temporal cnn
class tem_cnn_global(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(tem_cnn_global, self).__init__()
        self.kernel_size = kernel_size
        self.temConv = nn.Conv3d(input_dim, output_dim, kernel_size=[1, kernel_size, 1], stride=1, padding=[0,0,0])
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(args.dropRateG)

    def forward(self, embeds):
        ret_flow = self.temConv(embeds)
        ret_drop = self.drop(ret_flow)
        return self.act(ret_drop)

# embedding transform
class Transform_3d(nn.Module):
    def __init__(self):
        super(Transform_3d, self).__init__()
        self.BN = nn.BatchNorm3d(args.latdim)
        self.Conv1 = nn.Conv3d(args.latdim, args.latdim, kernel_size=1)

    def forward(self, embeds):
        embeds_BN = self.BN(embeds)
        embeds1 = self.Conv1(embeds_BN)
        return embeds1


class STHSL(nn.Module):
    def __init__(self):
        super(STHSL, self).__init__()

        self.dimConv_in = nn.Conv3d(1, args.latdim, kernel_size=1, padding=0, bias=True)
        self.dimConv_local = nn.Conv2d(args.latdim, 1, kernel_size=1, padding=0, bias=True)
        self.dimConv_global = nn.Conv2d(args.latdim, 1, kernel_size=1, padding=0, bias=True)

        self.spa_cnn_local1 = spa_cnn_local(args.latdim, args.latdim)
        self.spa_cnn_local2 = spa_cnn_local(args.latdim, args.latdim)
        self.tem_cnn_local1 = tem_cnn_local(args.latdim, args.latdim)
        self.tem_cnn_local2 = tem_cnn_local(args.latdim, args.latdim)

        self.Hypergraph_Infomax = Hypergraph_Infomax()
        self.tem_cnn_global1 = tem_cnn_global(args.latdim, args.latdim, 9)
        self.tem_cnn_global2 = tem_cnn_global(args.latdim, args.latdim, 9)
        self.tem_cnn_global3 = tem_cnn_global(args.latdim, args.latdim, 9)
        self.tem_cnn_global4 = tem_cnn_global(args.latdim, args.latdim, 6)

        self.local_tra = Transform_3d()
        self.global_tra = Transform_3d()


    def forward(self, embeds_true, neg):
        embeds_in_global = self.dimConv_in(embeds_true.unsqueeze(1))
        DGI_neg = self.dimConv_in(neg.unsqueeze(1))
        embeds_in_local = embeds_in_global.permute(0, 3, 1, 2, 4).contiguous().view(-1, args.latdim, args.row, args.col, 4)
        spa_local1 = self.spa_cnn_local1(embeds_in_local)
        spa_local2 = self.spa_cnn_local2(spa_local1)
        spa_local2 = spa_local2.view(-1, args.temporalRange, args.latdim, args.areaNum, args.cateNum).permute(0, 2, 3, 1, 4)
        tem_local1 = self.tem_cnn_local1(spa_local2)
        tem_local2 = self.tem_cnn_local2(tem_local1)
        eb_local = tem_local2.mean(3)
        eb_tra_local = self.local_tra(tem_local2)
        out_local = self.dimConv_local(eb_local).squeeze(1)

        hy_embeds, Infomax_pred = self.Hypergraph_Infomax(embeds_in_global, DGI_neg)
        tem_global1 = self.tem_cnn_global1(hy_embeds)
        tem_global2 = self.tem_cnn_global2(tem_global1)
        tem_global3 = self.tem_cnn_global3(tem_global2)
        tem_global4 = self.tem_cnn_global4(tem_global3)
        eb_global = tem_global4.squeeze(3)
        eb_tra_global = self.global_tra(tem_global4)
        out_global = self.dimConv_global(eb_global).squeeze(1)
        return out_local, eb_tra_local, eb_tra_global, Infomax_pred, out_global