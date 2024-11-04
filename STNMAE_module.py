import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import partial
import layers
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = layers.GraphConvolution(input_dim, hidden_dim)
        self.gc2 = layers.GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = nn.PReLU()(self.gc1(x, adj))    #####################################后续试试
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(input_dim,  output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1, bias=False)
        )

    def forward(self, z):
        # print("Z的维度", z.size())
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta# beta注意力权重矩阵，(beta * z).sum(1)全局表示

    """STMGCN和SpatialMGCN前向传播部分代码上是一样的，但是实施的细节不同，因为project不同"""


class MGCN(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, dropout):
        super(MGCN, self).__init__()
        self.GCNA1 = GCN(input_dim, latent_dim, output_dim, dropout)  # 初始化GCN类，GCN1特征维度，隐藏层维度
        self.GCNA2 = GCN(input_dim, latent_dim, output_dim, dropout)
        self.GCNA3 = GCN(input_dim, latent_dim, output_dim, dropout)
        self.GCNA4 = GCN(input_dim, latent_dim, output_dim, dropout)

    def forward(self, X, features1, features2, adj1, adj2):
        emb1 = self.GCNA1(X, features1)
        emb2 = self.GCNA2(X, features2)
        emb3 = self.GCNA3(X,  adj1)
        emb4 = self.GCNA4(X,  adj2)
        return emb1, emb2, emb3, emb4

class stnmae_module(nn.Module):
    def __init__(
            self,
            input_dim,
            nclass,
            # latent_dim=64,
            # output_dim=32,
            # latent_dim=128,
            # output_dim=32,
            latent_dim=128,
            output_dim=64,
            train_dim=128,
            num_layers=2,
            dropout=0.2,
            num_graph=4,
            g_type='GCN',
            decode_type='GCN',
            remask_method="random",
            dorp_code=0.2,
            drop_en=0.2,
            alpha=0.1,
            mask_rate=0.8,
            remask_rate=0.8,
            device='cuda:0'
    ):#隐藏层维度
        super(stnmae_module, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.train_dim = train_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.nclass = nclass
        self.num_graph = num_graph
        self.g_type = g_type
        self.decode_type = decode_type
        self.remask_method = remask_method
        self.dorp_code = dorp_code
        self.drop_en = drop_en
        self.alpha = alpha
        self.mask_rate = mask_rate
        """掩码比例，到时候可以调调，0.8感觉有点高"""
        self.remask_rate = remask_rate  ###感觉仍然很高
        self.mask_dim = output_dim
        self.input_latent = output_dim
        self.en_hidden = latent_dim
        self.en_hidden2 = output_dim
        self.latent_p = output_dim
        self.output_dec = output_dim + output_dim
        self.output_dec1 = output_dim
        self.emb = output_dim + output_dim + output_dim

        self.GCN = GCN(input_dim, latent_dim, output_dim, dropout)
        self.loss_type = self.setup_loss_fn(loss_fn='sce', alpha_l=3)
        self.cluster_layer = Parameter(torch.Tensor(self.nclass,  output_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer)

        self.encoder = Encodeer_Model(self.input_dim, self.en_hidden,  self.en_hidden2, self.drop_en, self.device)
        self.mgcn = MGCN(self.input_dim, self.latent_dim,  self.output_dim, self.dropout)
        self.encode = self.Code(self.g_type, self.input_dim, self.latent_dim, self.output_dim, self.dorp_code)
        # self.decoder = self.Code(self.decode_type, self.output_dec, self.latent_dim, self.input_dim, self.dorp_code)  ###初始版本对应的解码器
        self.decoder = self.Code(self.decode_type, self.output_dec1, self.latent_dim, self.input_dim, self.dorp_code)
        self.encode_latent = self.Code(self.g_type, self.input_latent, self.latent_dim, self.output_dim, self.dorp_code)
        self.projector = nn.Sequential(nn.Linear(self.latent_p, self.train_dim),
                                       nn.PReLU(), nn.Linear(self.train_dim, self.latent_p))
        self.projector_generate = nn.Sequential(nn.Linear(self.latent_p, self.train_dim),
                                                nn.PReLU(), nn.Linear(self.train_dim, self.latent_p))
        self.predictor = nn.Sequential(nn.PReLU(), nn.Linear(self.latent_p, self.latent_p))
        self.encode_generate = self.Code(self.g_type, self.input_dim, self.latent_dim, self.output_dim, self.dorp_code)

        self.encode_generate.load_state_dict(self.encode.state_dict())
        self.projector_generate.load_state_dict(self.projector.state_dict())

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim)).to(self.device)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.mask_dim)).to(self.device)
        self.encoder_to_decoder = nn.Linear(self.output_dim, self.output_dim, bias=False).to(self.device)
        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def forward(self, X, adj, features1, features2, adj1, adj2):
        targets = None
        adj, X, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, X, self.mask_rate)
        Zf = self.encoder(X)
        Gf1, Gf2, Gs1, Gs2 = self.mgcn(X, features1, features2, adj1, adj2)
        """下面的代码是初始版本，勿删！！！！！！！！！"""
        # H0 = torch.cat((Zf, Gf1), 1)
        # H1 = torch.cat((Zf, Gf2), 1)
        # H2 = torch.cat((Zf, Gs1), 1)
        # H3 = torch.cat((Zf, Gs2), 1)
        # H = self.encode_latent(Zf, adj)
        # emb1 = torch.cat([H, (H0 + H1 + H2 + H3) / 4], dim=1).to(self.device)
        """这一版更符合画的模型架构图"""
        H0 = Gf1
        H1 = Gf2
        H2 = Gs1
        H3 = Gs2
        H = self.encode_latent(Zf, adj)
        emb1 = torch.cat([H, (Gf1 + Gf2 + Gs1 + Gs2) / 4, Zf], dim=1).to(self.device)

        linear = nn.Linear(self.emb, self.output_dim).to(self.device)
        emb = linear(emb1).to(self.device)

        with torch.no_grad():#不进行梯度计算生成目标特征表示
            X_target = self.encode_generate(X, adj)#通过encoder_ema对 drop_g2 和原始特征 x 进行编码
            x_target = self.projector_generate(X_target[keep_nodes])#否则对保留节点进行预测（未掩码的那部分保留节点）

        X_pred = self.projector(H[keep_nodes])
        x_pred = self.predictor(X_pred)
        loss_latent = sce_loss(x_pred, x_target, 1)#没有目标target则对未编码部分执行上面相应的操作

        # ---- attribute reconstruction ----
        latent_emb = self.encoder_to_decoder(H)
        loss_rec_all = 0
        if self.remask_method == "random":#随机重掩码
            for i in range(self.num_graph):
                locals()[f'G{i}'] = locals()[f'H{i}'].clone()
                # locals()[f'G{i}'], (remask_nodes, rekeep_nodes) = self.random_remask(adj, locals()[f'G{i}'], self.remask_rate)#进行重掩码
                locals()[f'G{i}'], _, _ = self.random_remask(adj, locals()[f'G{i}'], self.remask_rate)
                # print("H_i维度",  locals()[f'H_{i}'].size())
                locals()[f'recon{i}'] = self.decoder(locals()[f'G{i}'], adj)#解码器重建输入，即生成Z
                # x_init = X[mask_nodes]
                x_init = X[mask_nodes]
                x_rec = locals()[f'recon{i}'][mask_nodes]
                loss_rec = self.loss_type(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all#重构损失
        elif self.remask_method == "fixed":#固定损失
            for i in range(self.num_graph):
                locals()[f'G{i}'] = locals()[f'H{i}'].clone()
                locals()[f'G{i}'] = self.fixed_remask(locals()[f'G{i}'], mask_nodes)
                # locals()[f'H_{i}'], _, _ = self.random_remask(adj, locals()[f'H_{i}'], self.remask_rate)  # 进行重掩码
                # print("H_i维度",  locals()[f'H_{i}'].size())
                locals()[f'recon{i}'] = self.decoder(locals()[f'G{i}'], adj)# 解码器重建输入，即生成Z
                x_init = X[mask_nodes]
                x_rec = locals()[f'recon{i}'][mask_nodes]
                loss_rec = self.loss_type(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
            # rep = self.fixed_remask(latent_emb, mask_nodes)#对输入解码器进行掩码
            # x_rec = self.decoder2(rep, adj)#重构损失
            # x_init = X[mask_nodes]
            # loss_rec = self.loss_type(x_init, x_rec)#同上
        else:
            raise NotImplementedError

        q = 1.0 / ((1.0 + torch.sum((emb.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)  # 便于损失计算
        # print("q的值", q)
        return emb, q, loss_rec, loss_latent

    # def ema_update(self):
    #     momentum = 0.996
    #     def update(student, teacher):
    #         with torch.no_grad():
    #         # m = momentum_schedule[it]  # momentum parameter
    #             m = momentum
    #             for param_q, param_k in zip(student.parameters(), teacher.parameters()):
    #                 param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    #     update(self.encode, self.encode_generate)
    #     update(self.projector, self.projector_generate)

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            loss_type = nn.MSELoss()
        elif loss_fn == "sce":
            loss_type = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return loss_type

    def random_remask(self, adj, rep, remask_rate=0.5):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]
        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token
        # return rep, (remask_nodes, rekeep_nodes)
        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()
        return use_adj, out_x, (mask_nodes, keep_nodes)

    def Code(self, m_type, in_dim, num_hidden, out_dim, dropout) -> nn.Module:
        if m_type == "GCN":
            mod = GCN(in_dim, num_hidden, out_dim, dropout)
        elif m_type == "mlp":
            mod = nn.Sequential(nn.Linear(in_dim, num_hidden * 2), nn.PReLU(), nn.Dropout(0.2), nn.Linear(num_hidden * 2, out_dim))
        elif m_type == "linear":
            mod = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError
        return mod


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class Encodeer_Model(nn.Module):
    def __init__(self, input_dim, intermediate_dim, kan_dim, p_drop, device):
        super(Encodeer_Model, self).__init__()
        self.device = device
        self.full_block = full_block(input_dim, intermediate_dim,  p_drop).to(self.device)
        self.KAN = layers.KANLinear(intermediate_dim, kan_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.full_block(x)
        feat = self.KAN(x)
        return feat




