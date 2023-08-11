# SpaFormer
This is the official github repo for paper **Single Cells Are Spatial Tokens: Transformers for Spatial Transcriptomic Data Imputation** [[Arxiv]](https://arxiv.org/abs/2302.03038)

## Data Avalability
Preprocessed data will be provided in this repo after official release of the paper.

## Acknowledgement
Some impelementations in this Repo are based on [GraphMAE](https://arxiv.org/abs/2205.10803) and [scBERT](https://github.com/TencentAILabHealthcare/scBERT)

## Supplementary Reproduciability Information for our Paper

#### Baselines ####
To evaluate the effectiveness of SpaFormer, we compare it with the state-of-the-art spatial and non-spatial transcriptomic imputation models:
(1) scImpute[1] employs a probabilistic model to detect dropouts, and implements imputation through non-negative least squares regression.
(2) SAVER[2] uses negative binomial distribution to model the data and estimates a Gamma prior through Poisson Lasso regression. The posterior mean is used to output expression with uncertainty quantification from the posterior distribution.
(3) scVI[3] models dropouts with a ZINB distribution, and estimates the distributional parameters of each gene in each cell with a VAE model.
(4) DCA[4] is an autoencoder that predicts parameters of chosen distributions like ZINB to generate the imputed data. 
(5) GraphSCI[5] employs a graph autoencoder on a gene correlation graph. Meanwhile, it uses another autoencoder to reconstruct the gene expressions, taking graph autoencoder embeddings as additional input.
(6) scGNN[6] first builds a cell-cell graph based on gene expression similarity and then utilizes a graph autoencoder together with a standard autoencoder to refine graph structures and cell representation. Lastly, an imputation autoencoder is trained with a graph Laplacian smoothing term added to the reconstruction loss.
(7) gimVI[7] is a deep generative model for integrating spatial transcriptomics data and scRNA-seq data which can be used to impute spatial transcriptomic data. gimVI is based on scVI and employs alternative conditional distributions to address technology-specific covariate shift more effectively.
%(8) biCGNN[8] builds a spatial graph and a cell-wise kNN graph separately to enhance spatial transcriptomic imputation. It is specified in a non-archival manuscript.
(8) Sprod[9] is the latest published imputation method for spatial transcriptomics data. It first projects gene expressions to a latent space, then connects neighboring cells in the latent space, and prunes it with physical distance. Then a denoised matrix is learned by jointly minimizing the reconstruction error and a graph Laplacian smoothing term. 
(9) SpaGAT is a baseline model created by ourselves. It is the same bi-level masking autoencoder framework as SpaFormer, based on a graph neural network encoder with spatial graphs. Specifically, we implement a graph attention network (GAT) as the encoder. Since the graph attention network is a localized version of transformers, SpaGAT can be considered an ablation study for our SpaFormer model.

#### Hyperparameter Settings ####
For our own2 SpaGAT and SpaFormer, we first normalize the total RNA counts of each cell, and then apply log1p transform. We heavily conducted hyperparameters searching on the Lung dataset. However, we noticed that the performance is not sensitive to most hyperparameters, except for masking rate, autoencoder type, and positional encodings as we presented in ablation studies. To reproduce our results, the recommended hyperparameters are n\_layer=2, num\_heads=8, num\_hidden=128, latent\_dim=20, learning rate=1e-3, weight\_decay=5e-4. We used these default hyperparameters in the other two datasets. The source codes will been released on our github. For our own created baseline model SpaGAT, we used the same set of hyperparameters while replacing the transformer encoder with a graph attention network.

For baseline models, all the implementations are from the authors’ repo/software. Optimizers/trainers are provided by original implementations. Preprocessings are also consistent with the original methods. The detailed settings are as below:

ScImpute only involves two parameters. Parameter K denotes the potential number of cell populations, threshold t on dropout probabilities. We set t=0.5 and K=15. This is per the author’s instructions in their paper, i.e., a default threshold value of 0.5 is sufficient for most scRNA-seq data, and K should be chosen based on the clustering result of the raw data and the resolution level desired by the users, where K=15 is close to the ground-truth cell-type number.

SAVER is a statistical model and does not expose any hyperparameters in their implementation. Therefore, we run their default setting.

For scVI, we searched for combinations of n\_hidden=[128, 256], n\_layer=[1, 2], gene\_likelihood=[nb, zinb] on the Lung5 dataset. All settings are repeated five times and the best mean performance is achieved by n\_hidden=128, n\_layer=1, and gene\_likelihood=’nb’. Other parameters are per default. We then applied this set of hyperparameters to all three datasets and reported it in the main results.

For DCA, the original hyperparameter optimization mode is broken (there is a relevant unresolved issue on GitHub), and there are no other parameter instructions in the tutorial, therefore we used default parameters.

For GraphSCI, we tried hidden1=[16, 32, 64], hidden2=[32, 64]. Note that GraphSCI does not have other hyperparameters and the default number of hidden sizes is quite small. The performance reported in our paper was obtained from hidden1=32, and hidden2=64.

gimVI software does not expose hyperparameters such as n\_hidden and n\_layer, so we follow the default settings. Additionally, gimVI requires external scRNA-seq reference data. For the Lung5 dataset, we used data from GSE131907 [11] as a reference. For Kidney, we used data from GSE159115 [12], and for Liver we used data from GSE115469 [13].

scGNN provides two sets of preprocessing in their GitHub repo and we adopt the first one "Cell/Gene filtering without inferring LTMG" for CSV format. Then we follow the corresponding instructions to run their imputation method but get an out-of-memory error during EM algorithm.

Sprod provides batch mode for handling very big spatial datasets. We follow their instructions for dataset without a matching image and ran the batch mode with num\_of\_batches=30, however, it can not finish running within 48 hours even on the smallest Kidney dataset.
