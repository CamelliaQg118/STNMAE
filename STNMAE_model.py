import numpy as np
import torch
import pandas as pd
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .STNMAE_module import stnmae_module
from tqdm import tqdm
from sklearn import metrics
import ST_NMAE
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
EPS = 1e-15


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def kl_loss(q, p):
    return F.kl_div(q, p, reduction="batchmean")


class stnmae_train:
    def __init__(
            self,
            X,
            adata,
            adj,
            adj1,
            adj2,
            features1,
            features2,
            n_clusters,
            dataset,
            epochs=800,
            dec_interval=3,
            lr=0.0001,
            decay=0.0001,
            rec_w=10,
            laten_w=6,
            kl_w=1,
            dec_tol=0.00,
            device='cuda:0',
    ):
        self.random_seed = 42
        ST_NMAE.fix_seed(self.random_seed)

        self.rec_w = rec_w
        self.laten_w = laten_w
        self.kl_w = kl_w
        self.n_clusters = n_clusters
        self.dataset = dataset
        self.device = device
        self.epochs = epochs
        self.dec_interval = dec_interval
        self.learning_rate = lr
        self.weight_decay = decay
        self.dec_tol = dec_tol
        self.adata = adata.copy()
        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.features1 = features1.to(self.device)
        self.features2 = features2.to(self.device)
        self.adj = adj.to(self.device)
        self.adj1 = adj1.to(self.device)
        self.adj2 = adj2.to(self.device)
        self.model = stnmae_module(self.input_dim, self.n_clusters).to(self.device)

    def train(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_clusters * 2, random_state=42)
        test_z, q = self.model_eval()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()
        list_rec = []
        list_latent = []
        list_kl = []
        epoch_max = 0
        ari_max = 0
        idx_max = []
        emb_max = []
        if self.dataset in ['Human_Breast_Cancer', 'DLPFC']:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                if epoch % self.dec_interval == 0:
                    emb, tmp_q = self.model_eval()
                    tmp_p = target_distribution(torch.Tensor(tmp_q))
                    y_pred = tmp_p.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)
                #
                # self.model.train()
                # self.optimizer.zero_grad()
                _, out_q, loss_rec, loss_latent = self.model(self.X, self.adj, self.features1, self.features2, self.adj1, self.adj2)
                loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss_total = self.rec_w * loss_rec + self.laten_w * loss_latent + self.kl_w * loss_kl
                loss_total.backward()
                self.optimizer.step()
                list_rec.append(loss_rec.detach().cpu().numpy())
                list_kl.append(loss_kl.detach().cpu().numpy())
                list_latent.append(loss_latent.detach().cpu().numpy())
                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
                idx = kmeans.labels_
                self.adata.obsm['STNMAE'] = emb
                labels = self.adata.obs['ground']
                labels = pd.to_numeric(labels, errors='coerce')
                labels = pd.Series(labels).fillna(0).to_numpy()
                idx = pd.Series(idx).fillna(0).to_numpy()
                ari_res = metrics.adjusted_rand_score(labels, idx)
                if ari_res > ari_max:
                    ari_max = ari_res
                    epoch_max = epoch
                    idx_max = idx
                    emb_max = emb
            print("epoch_max", epoch_max)
            print("ARI=======", ari_max)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx_max)
            print("NMI=======", nmi_res)
            self.adata.obs['STNMAE'] = idx_max.astype(str)
            self.adata.obsm['emb'] = emb_max
            return self.adata.obsm['emb'], self.adata.obs['STNMAE']
        else:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                if epoch % self.dec_interval == 0:
                    emb, tmp_q = self.model_eval()
                    tmp_p = target_distribution(torch.Tensor(tmp_q))
                    y_pred = tmp_p.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)
                _, out_q, loss_rec, loss_latent = self.model(self.X, self.adj, self.features1, self.features2,
                                                             self.adj1, self.adj2)
                loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss_total = self.rec_w * loss_rec + self.laten_w * loss_latent + self.kl_w * loss_kl
                loss_total.backward()
                self.optimizer.step()
                list_rec.append(loss_rec.detach().cpu().numpy())
                list_kl.append(loss_kl.detach().cpu().numpy())
                list_latent.append(loss_latent.detach().cpu().numpy())
                # print(' epoch: ', epoch, ' loss_rec = {:.5f}'.format(loss_rec), ' loss_latent = {:.5f}'.format(loss_latent),
                #       'loss_kl = {:.5f}'.format(loss_kl), ' loss_total = {:.5f}'.format(loss_total))
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.plot(list_rec, label='rec')
            # ax.plot(list_latent, label='latent')
            # ax.plot(list_kl, label='kl')
            # ax.legend()
            # plt.show()
            return emb

    def model_eval(self):
        self.model.eval()
        emb, q, loss_rec, loss_latent = self.model(self.X, self.adj, self.features1, self.features2, self.adj1, self.adj2)
        q = q.data.cpu().numpy()
        emb = emb.data.cpu().numpy()
        return emb, q

