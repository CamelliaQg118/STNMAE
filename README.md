# STNAME(DOI:)
An official source code for the paper "STNMAE: Identifying spatial domains from spatial transcriptomics data with neighbor-aware multi-view masked graph autoencoder," accepted by Interdisciplinary Sciences: Computational Life Sciences. Any communications or issues are welcome. Please contact qigao118@163.com. If you find this repository useful to your research or work, it is really appreciated to cite our paper.

## Overview:
![Fig1](https://github.com/user-attachments/assets/c78a1dbf-69e3-4d9c-8071-8edf4d5f7fa4)
STNMAE, a neighborhood-aware multi-view graph autoencoder architecture for spatial domain identification, learns latent representations of gene expression profiles and spatial information by jointly training a features masking encoder, a multi-view autoencoder, and a target generator-mapper. STNMAE performs downstream analysis, such as spatial domain identification, trajectory inference, DEGs. STNMAE has been applied to seven spatial transcriptomics datasets across platforms like 10X Visium, STARmap, Stereo-seq, Slide-seqV2.

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
