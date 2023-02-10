import numpy as np
from scipy.stats import pearsonr as pe
import scanpy as sc
import logging

mergedicts = {}
mergedicts['T CD8 memory'] = 'lymphocyte'
mergedicts['T CD8 naive'] = 'lymphocyte'
mergedicts['T CD4 naive'] = 'lymphocyte'
mergedicts['T CD4 memory'] = 'lymphocyte'
mergedicts['Treg'] = 'lymphocyte'
mergedicts['B-cell'] = 'lymphocyte'
mergedicts['plasmablast'] = 'lymphocyte'
mergedicts['NK'] = 'lymphocyte'
mergedicts['monocyte'] = 'Mcell'
mergedicts['macrophage'] = 'Mcell'
mergedicts['mDC'] = 'Mcell'
mergedicts['pDC'] = 'Mcell'
mergedicts['tumors'] = 'tumors'
mergedicts['epithelial'] = 'epithelial'
mergedicts['mast'] = 'mast'
mergedicts['endothelial'] = 'endothelial'
mergedicts['fibroblast'] = 'fibroblast'
mergedicts['neutrophil'] = 'neutrophil'


def comprehensive_evaluate(pred, gex, mask, lbl, logger = None, emb = None):  
    pred_mean = pred.mean()
    gex_mean = gex.mean()
    
    logging.info(f'Pred mean {pred_mean} True mean {gex_mean}')
    logging.info(f'Overall rmse: {np.sqrt(np.mean((pred-gex)**2))}')
    pm = pred[mask.astype(bool)]
    tm = gex[mask.astype(bool)]
    mrmse = np.sqrt(np.mean((pm-tm)**2))
    mcos = np.sum(pm*tm)/np.sqrt(np.sum(pm*pm)*np.sum(tm*tm))
    mpe = pe(pm, tm)[0]
    logging.info(f'Masked rmse: {mrmse}')
    logging.info(f'Masked cosine: {mcos}')
    logging.info(f'Masked pearson: {mpe}')

    from sklearn.preprocessing import LabelEncoder
    if emb is not None:
        pred = emb
    lbl[lbl=='NotDet'] = 'nan'
    pred = pred[lbl!='nan']
    lbl = lbl[lbl!='nan']
    lbl[lbl.str.startswith('tumor')] = 'tumor'
    for key, v in mergedicts.items():
        lbl[lbl == key] = v
    gt = LabelEncoder().fit_transform(lbl)
    
    adata = sc.AnnData(pred)
    sc.pp.pca(adata, n_comps=10, random_state=1)
    sc.pp.neighbors(adata, metric='correlation')
    sc.tl.leiden(adata, resolution=0.26, random_state=1)

    for res in [0.3, 0.32, 0.34, 0.36]:
        if len(adata.obs['leiden'].unique()) >= gt.max()+1:
            break
        sc.tl.leiden(adata, resolution=res, random_state=1)
        
    pred_labels = adata.obs['leiden']
    logging.info(f'{len(pred_labels.unique())} clusters detected  |  Ground-truth labels have {gt.max()+1} clusters.')

    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.metrics import adjusted_rand_score
    #pred = TSNE(3, init='random', learning_rate='auto',  perplexity=10, n_jobs=16).fit_transform(pred)
    #kmeans = KMeans(n_clusters=18, random_state=200)
    #pred_labels = kmeans.fit_predict(pred)


    NMI_score = round(normalized_mutual_info_score(gt, pred_labels, average_method='arithmetic'), 3)
    ARI_score = round(adjusted_rand_score(gt, pred_labels), 3)

    logging.info(f'NMI: {NMI_score}')
    logging.info(f'ARI: {ARI_score}')
    if logger is not None:
        logger.note({'pred_mean': pred_mean, 'gex_mean': gex_mean, 'rmse': mrmse, 'cosine': mcos, 'pearson': mpe, 'nmi': NMI_score, 'ARI': ARI_score})
