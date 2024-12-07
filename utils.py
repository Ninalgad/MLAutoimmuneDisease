import numpy as np
import scanpy as sc

###############
# utils fct
###############


def log1p_normalization(arr):
    """  Apply log1p normalization to the given array """

    scale_factor = 100
    return np.log1p((arr / np.sum(arr, axis=1, keepdims=True)) * scale_factor)


def normalize_adata(adata: sc.AnnData) -> sc.AnnData:
    """
    Normalize and apply log1p transformation to the expression matrix of an AnnData object.
    (The function normalizes the gene expression by row)

    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object containing gene expression data.
    """

    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    filtered_adata.X = log1p_normalization(filtered_adata.X)

    return filtered_adata


def load_adata(expr_path, genes=None, barcodes=None, normalize=False):
    """
    Load AnnData object from a given path

    Parameters:
    -----------
    expr_path : str
        Path to the .h5ad file containing the AnnData object.
    genes : list, optional
        List of genes to retain. If None, all genes are kept.
    barcodes : list, optional
        List of barcodes (cells) to retain. If None, all cells are kept.
    normalize : bool, optional
        Whether to apply normalization (log1p normalization) to the data.

    Returns:
    --------
    pd.DataFrame
        Gene expression data as a DataFrame.
    """

    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()


def merge_dict(main_dict, new_dict, value_fn=None):
    """
    Merge new_dict into main_dict. If a key exists in both dicts, the values are appended.
    Else, the key-value pair is added.
    Expects value to be an array or list - if not, it is converted to a list.
    If value_fn is not None, it is applied to each item in each value in new_dict before merging.
    Args:
        main_dict: main dict
        new_dict: new dict
        value_fn: function to apply to each item in each value in new_dict before merging
    """

    if value_fn is None:
        def value_fn(x): return x

    for key, value in new_dict.items():
        if not isinstance(value, list):
            value = [value]
        value = [value_fn(v) for v in value]
        if key in main_dict:
            main_dict[key] = main_dict[key] + value
        else:
            main_dict[key] = value

    return main_dict


def post_collate_fn(batch):
    """ Post collate function to clean up batch """

    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)

    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)

    return batch


def merge_fold_results(arr):
    """ Merges results from multiple cross-validation folds, aggregating Pearson correlation across all folds. """
    aggr_dict = {}
    for dict in arr:
        for item in dict['pearson_corrs']:
            gene_name = item['name']
            correlation = item['pearson_corr']
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]

    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append({
            "name": key,
            "pearson_corrs": value,
            "mean": np.mean(value),
            "std": np.std(value)
        })
        all_corrs += value

    mean_per_split = [d['pearson_mean'] for d in arr]

    return {
        "pearson_corrs": aggr_results,
        "pearson_mean": np.mean(mean_per_split),
        "pearson_std": np.std(mean_per_split),
        "mean_per_split": mean_per_split
    }
