import scanpy as sc
import pandas as pd
import torch
import graph_construct
import utils
import os
import clustering
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import metrics
import ST_NMAE

random_seed = 42
ST_NMAE.fix_seed(random_seed)

# if __name__ == "__main__":

ARI_list = []
os.environ['R_HOME'] = 'D:/Software/Code/R/R-4.3.3/R-4.3.3'
os.environ['R_USER'] = 'C:/Users/29461/.conda/envs/SEDR/Lib/site-packages/rpy2'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

#######data load
dataset = 'DLPFC'
slice = '151510'
platform = '10X'
file_fold = os.path.join('../Data', platform, dataset, slice)
adata, adata_X = utils.load_data(dataset, file_fold)

df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
adata = utils.label_process_DLPFC(adata, df_meta)

savepath = '../Result/STNMAE/DLPFC/test/' + str(slice) + '/'
if not os.path.exists(savepath):
    os.mkdir(savepath)

n_clusters = 5 if slice in ['151669', '151670', '151671', '151672'] else 7
print('n_clusters', n_clusters)

adata, adj, adj1, adj2, features1, features2 = graph_construct.graph_build(adata, adata_X, dataset)

stnmae_net = ST_NMAE.stnmae_train(adata.obsm['X_pca'],  adata, adj, adj1, adj2, features1, features2, n_clusters,
                                  dataset, device=device)
emb, idx = stnmae_net.train()
adata.obs['STNMAE'] = idx
adata.obsm['STNMAE'] = emb
adata.obs['ground_truth'] = df_meta['layer_guess']
adata = adata[~pd.isnull(adata.obs['ground_truth'])]
new_type = utils.refine_label(adata, radius=50, key='STNMAE')
adata.obs['STNMAE'] = new_type
ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STNMAE'])
adata.uns["ARI"] = ARI
adata.uns["NMI"] = NMI
print('===== Project: {}_{} ARI score: {:.4f}'.format(str(dataset), str(slice), ARI))
print('===== Project: {}_{} NMI score: {:.4f}'.format(str(dataset), str(slice), NMI))
print(str(slice))
print(n_clusters)
ARI_list.append(ARI)

#map
plt.rcParams["figure.figsize"] = (3, 3)
title = "Manual annotation (" + dataset + "#" + slice + ")"
sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=300)
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
sc.pl.spatial(adata, color='ground_truth', ax=axes[0], show=False)
sc.pl.spatial(adata, color=['STNMAE'], ax=axes[1], show=False)
axes[0].set_title("Manual annotation (" + dataset + "#" + slice + ")")
axes[1].set_title('STNMAE_Clustering: (ARI=%.4f)' % ARI)

plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)  
plt.savefig(savepath + 'STNMAE.jpg', dpi=300) 


sc.pp.neighbors(adata, use_rep='STNMAE', metric='cosine')
sc.tl.umap(adata)
sc.pl.umap(adata, color='STNMAE', title='STNMAE', show=False)
plt.savefig(savepath + 'umap.jpg', bbox_inches='tight', dpi=300)

for ax in axes:
    ax.set_aspect(1)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)


title = 'STNMAE:{}_{} ARI={:.4f} NMI={:.4f}'.format(str(dataset), str(slice), adata.uns['ARI'], adata.uns['NMI'])
sc.pl.spatial(adata, img_key="hires", color=['STNMAE'], title=title, show=False)
plt.savefig(savepath + 'STNMAE_NMI_ARI.tif', bbox_inches='tight', dpi=300)
