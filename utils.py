import os
import ot
import torch
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def adata_hvg(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def adata_hvg_process(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.scale(adata)
    return adata

def adata_hvg_slide(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_data(dataset, file_fold):
    if dataset == "DLPFC":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == 'MOB':
        savepath = '../Result/STNMAE/MOB_Stereo/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        counts_file = os.path.join(file_fold, 'RNA_counts.tsv')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0).T
        counts.index = [f'Spot_{i}' for i in counts.index]
        adata = sc.AnnData(counts)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()

        pos_file = os.path.join(file_fold, 'position.tsv')
        coor_df = pd.read_csv(pos_file, sep='\t')
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]

        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obs['x'] = coor_df['x'].tolist()
        adata.obs['y'] = coor_df['y'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))
    elif dataset == 'ISH':
        adata = sc.read(file_fold + '/STARmap_20180505_BY3_1k.h5ad')
        print(adata)
        adata.obs['x'] = adata.obs["X"]
        adata.obs['y'] = adata.obs["Y"]
        adata.layers['count'] = adata.X
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    else:
        platform = '10X'
        file_fold = os.path.join('../Data', platform, dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        df_meta = pd.read_csv(os.path.join('../Data', dataset,  'metadata.tsv'), sep='\t', header=None, index_col=0)
        adata.obs['layer_guess'] = df_meta['layer_guess']
        df_meta.columns = ['over', 'ground_truth']
        adata.obs['ground_truth'] = df_meta.iloc[:, 1]
        adata.var_names_make_unique()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    return adata, adata_X


def label_process_DLPFC(adata, df_meta):
    labels = df_meta["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    return adata


def refine_label(adata, radius=50, key='label'):  
    n_neigh = radius  
    new_type = []  
    old_type = adata.obs[key].values  
    
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean') 

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type
