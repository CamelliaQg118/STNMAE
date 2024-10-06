import scipy.sparse as sp
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import torch
import numba


def graph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC':
        n = 12
        adj = load_adj(adata, n)
        adj1 = load_adj1(adata)
        adj2 = load_adj2(adata, include_self=True, n=n)
        # features1 = load_features1(adata_X, k=12, mode="connectivity", metric="cosine")
        features1 = load_features1(adata_X, k=n, mode="connectivity", metric="cosine")
        features2 = load_features2(adata_X, k=n, mode="connectivity", metric="euclidean")

    elif dataset == 'MBO':
        n = 10
        adj = load_adj(adata, n)
        adj1 = load_adj1(adata)
        adj2 = load_adj2(adata, include_self=True, n=n)
        features1 = load_features1(adata_X, k=14, mode="connectivity", metric="cosine")
        features2 = load_features2(adata_X, k=14, mode="connectivity", metric="euclidean")

    elif dataset =='MOB_V2':
        n = 7
        adj = load_adj(adata, n)
        adj1 = load_adj1(adata)
        adj2 = load_adj2(adata, include_self=True, n=n)
        features1 = load_features1(adata_X, k=12, mode="connectivity", metric="cosine")
        features2 = load_features2(adata_X, k=12, mode="connectivity", metric="euclidean")

    elif dataset == 'Adult_Mouse_Brain_Section_1':
        n = 5
        adj = load_adj(adata, n)
        adj1 = load_adj1(adata)
        adj2 = load_adj2(adata, include_self=True, n=n)
        features1 = load_features1(adata_X, k=n, mode="connectivity", metric="cosine")
        features2 = load_features2(adata_X, k=n, mode="connectivity", metric="euclidean")
    elif dataset == 'ISH':
        n = 7
        adj = load_adj(adata, n)
        adj1 = load_adj1(adata)
        adj2 = load_adj2(adata, include_self=True, n=n)
        features1 = load_features1(adata_X, k=n, mode="connectivity", metric="cosine")
        features2 = load_features2(adata_X, k=n, mode="connectivity", metric="euclidean")
    else:
        n = 10
        adj = load_adj(adata, n)
        adj1 = load_adj1(adata)
        adj2 = load_adj2(adata, include_self=True, n=n)
        features1 = load_features1(adata_X, k=n, mode="connectivity", metric="cosine")
        features2 = load_features2(adata_X, k=n, mode="connectivity", metric="euclidean")
    return adata, adj, adj1, adj2, features1, features2


def load_adj(adata, n):
    adj = generate_adj(adata, include_self=False, n=n)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm = preprocess_adj(adj)
    return adj_norm


def generate_adj(adata, include_self=False, n=6):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj[i, n_neighbors] = 1
    if not include_self:
        x, y = np.diag_indices_from(adj)
        adj[x, y] = 0
    adj = adj + adj.T
    adj = adj > 0
    adj = adj.astype(np.int64)
    return adj


def preprocess_adj(adj):
    adj= adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_adj1(adata):
    wid = 1
    x = adata.obs['x']
    y = adata.obs['y']
    adj = calculate_adj_matrix(x, y)
    adj_1 = np.exp(-1*(adj**2)/(2*(wid**2)))
    adj_1 = sp.coo_matrix(adj_1)
    adj_1 = normalize_sparse_matrix(adj_1 + sp.eye(adj_1.shape[0]))
    adj_1 = sparse_mx_to_torch_sparse_tensor(adj_1)
    return adj_1


def calculate_adj_matrix(x, y):
    A = np.array([x, y]).T.astype(np.float32)
    adj = distance(A)
    return adj


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i]-t2[i])**2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=False, nogil=False)#######都改为false后性能降低了
def distance(A):
    n = A.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(A[i], A[j])
    return adj


def load_adj2(adata, include_self=False, n=6):
    assert 'spatial' in adata.obsm, 'AnnData object should provide spatial information'
    cosine_sim = cosine_similarity(adata.obsm['spatial'])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(-cosine_sim[i, :])[:n + 1]
        adj_mat[i, n_neighbors] = 1
    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)
    adj = norm_adj2(adj_mat)
    return adj


# def normalize_adj1(A, p):
#     # A = A + sp.eye(A.shape[0])无需自环处理，因为后续已经实现
#     degrees = np.power(np.array(A.sum(1)), p).flatten()
#     degrees[np.isinf(degrees)] = 0.
#     if sp.issparse(A):
#         D = sp.diags(degrees)
#     else:
#         D = np.diag(degrees)
#     normalized_D = D
#     adj_normalized = normalized_D.dot(A).dot(normalized_D)
#     return adj_normalized

def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def norm_adj2(adj):
    adj = sp.coo_matrix(adj)
    adj_m1 = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_m1.eliminate_zeros()
    adj_norm = preprocess_adj(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_m1 = adj_m1.tocoo()
    # shape = adj_m1.shape
    # values = adj_m1.data
    # indices = np.stack([adj_m1.row, adj_m1.col])
    # adj_label = torch.sparse_coo_tensor(indices, values, shape)
    # norm_value = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    return adj_norm



def load_features1(features, k=12, mode="connectivity", metric="cosine"):
    A = kneighbors_graph(features, k, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    fadj = sparse_mx_to_torch_sparse_tensor(fadj)
    return fadj


def load_features2(features, k=12, mode="connectivity", metric="euclidean"):
    A = kneighbors_graph(features, k, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    fadj = sparse_mx_to_torch_sparse_tensor(fadj)
    return fadj
