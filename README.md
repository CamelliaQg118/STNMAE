# STNAME: Identifying spatial domains from spatial transcriptomics data with neighbor-aware multi-view masked graph autoencoder


The complete code will be made available after the article is published.
## Overview:

__STNAME__ (spatial transcriptomics neighbor-aware multi-view masked graph autoencoder) STNMAE, a neighborhood-aware multi-view graph autoencoder architecture for spatial domain identification, learns latent representations of gene expression profiles and spatial information by jointly training a features masking encoder, a multi-view autoencoder, and a target generator-mapper.STNMAE has been applied to seven spatial transcriptomics datasets across platforms like 10X Visium, Stereo-seq, and Slide-seqV2, proving its capability to deliver enhanced representations for a range of downstream analyses, such as clustering, visualization, trajectory inference, and differential gene analysis.

## Requirements:
 
STNMAE is implemented in the pytorch framework. Please run STNAME on CUDA. The following packages are required to be able to run everything in this repository (included are the versions we used):

```bash
python==3.11.0
torch==2.2.0
cudnn==11.8
numpy==1.25.2
scanpy==1.10.1
pandas==2.2.2
scipy==1.9.3
scikit-learn==1.1.3
anndata==0.10.7
R==4.3.3
ryp2==3.5.12
tqdm==4.66.2
matplotlib==3.8.4
seaborn==0.13.2
```
