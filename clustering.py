import scanpy as sc
import pandas as pd
import numpy as np
from sklearn import metrics
import ST_NMAE
import utils
from anndata import AnnData


def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=42):

    for res in np.arange(0.2, 2, increment):
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['leiden'].unique()) > n_clusters:
            break
    return res-increment


def leiden(adata, n_clusters, use_rep='emb', key_added='STNMAE', random_seed=42):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.leiden(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['leiden']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=42):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res-increment

def louvain(adata, n_clusters, use_rep='emb', key_added='STNMAE', random_seed=42):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)
    adata.obs[key_added] = adata.obs['louvain']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata



def mclust_R(adata, n_clusters, use_rep='STNMAE', key_added='STNMAE', random_seed=42):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    modelNames = 'EEE'
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


# def metric(stnmae_net, adata, df_meta, tool, n_clusters, random_seed):
#     if tool == 'mclust':
#         emb = stnmae_net.train()
#         adata.obsm['STNMAE'] = emb
#         adata.obs['ground_truth'] = df_meta['layer_guess']
#         adata = adata[~pd.isnull(adata.obs['ground_truth'])]
#         ST_NMAE.mclust_R(adata, n_clusters, use_rep='STNMAE', key_added='STNMAE', random_seed=random_seed)
#         radius = 50
#         new_type = utils.refine_label(adata, radius, key='STNMAE')
#         adata.obs['STNMAE'] = new_type
#         ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
#         NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
#     elif tool == 'leiden':
#         emb = stnmae_net.train()
#         adata.obsm['STNMAE'] = emb
#         adata.obs['ground_truth'] = df_meta['layer_guess']
#         adata = adata[~pd.isnull(adata.obs['ground_truth'])]
#         ST_NMAE.leiden(adata, n_clusters, use_rep='STNMAE', key_added='STNMAE', random_seed=random_seed)
#         radius = 50
#         new_type = utils.refine_label(adata, radius, key='STNMAE')
#         adata.obs['STNMAE'] = new_type
#         ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
#         NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
#
#     elif tool == 'louvain':
#         emb = stnmae_net.train()
#         adata.obsm['STNMAE'] = emb
#         adata.obs['ground_truth'] = df_meta['layer_guess']
#         adata = adata[~pd.isnull(adata.obs['ground_truth'])]
#         ST_NMAE.louvain(adata, n_clusters, use_rep='STNMAE', key_added='STNMAE', random_seed=random_seed)
#         radius = 50
#         new_type = utils.refine_label(adata, radius, key='STNMAE')
#         adata.obs['STNMAE'] = new_type
#         ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
#         NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STNMAE'])



