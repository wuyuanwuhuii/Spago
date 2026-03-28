"""
Utility functions

Some functions live here because otherwise managing their import
in other places would be overly difficult
"""
import os
import sys
import functools
import logging
from typing import *
import itertools
import collections
import gzip

import numpy as np
import pandas as pd
import scipy
import scanpy as sc
from anndata import AnnData
import mpl_scatter_density
import torch

import intervaltree as itree
import sortedcontainers
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch
SAVEFIG_DPI = 1200



DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
print(DATA_DIR)
assert os.path.isdir(DATA_DIR)
HG38_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.100.gtf.gz")
assert os.path.isfile(HG38_GTF)
HG19_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh37.87.gtf.gz")
assert os.path.isfile(HG19_GTF)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy's sparse matrix to torch's sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def plot_loss_history(history1,history2,history3,fname: str):
    """Constructing training loss curves"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(
        np.arange(len(history1)), history1, label="Train_G",
    )
    if len(history2):
        ax.plot(
        np.arange(len(history2)), history2, label="Train_D",)
    if len(history3):
        ax.plot(
        np.arange(len(history3)), history3, label="Test_G",)
    ax.legend()
    ax.set(
        xlabel="Epoch", ylabel="Loss",
    )
    #plt.show()
    fig.savefig(fname)
    return fig



def plot_auroc(
        truth,
        preds,
        title_prefix: str = "Receiver operating characteristic",
        fname: str = "",
):
    """
    Plot AUROC after flattening inputs
    """
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    fpr, tpr, _thresholds = metrics.roc_curve(truth, preds)
    auc = metrics.auc(fpr, tpr)
    logging.info(f"Found AUROC of {auc:.4f}")

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5))
    ax.plot(fpr, tpr)
    ax.set(
        xlim=(0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{title_prefix} (AUROC={auc:.2f})",
    )
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def plot_prc(
        truth,
        preds,
        title_prefix: str = "Receiver operating characteristic",
        fname: str = "",
):
    """
    Plot PRC after flattening inputs
    """
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    precision, recall, _thresholds = metrics.precision_recall_curve(truth, preds)
    auc = metrics.auc(recall,precision)
    logging.info(f"Found AUPRC of {auc:.4f}")

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(
        xlim=(0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="recall",
        ylabel="precision",
        title=f"{title_prefix} (PRC={auc:.2f})",
    )
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def rmse_value(truth,
               preds,
):
    'Calculate RMSE'
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    rmse=np.sqrt(metrics.mean_squared_error(truth, preds))
    logging.info(f"Found RMSE of {rmse:.4f}")

def plot_scatter_with_r(
    x: Union[np.ndarray, scipy.sparse.csr_matrix],
    y: Union[np.ndarray, scipy.sparse.csr_matrix],
    color=None,
    subset: int = 0,
    logscale: bool = False,
    density_heatmap: bool = False,
    density_dpi: int = 150,
    density_logstretch: int = 1000,
    title: str = "",
    xlabel: str = "Original norm counts",
    ylabel: str = "Inferred norm counts",
    xlim: Tuple[int, int] = None,
    ylim: Tuple[int, int] = None,
    one_to_one: bool = False,
    corr_func: Callable = scipy.stats.pearsonr,
    figsize: Tuple[float, float] = (7, 5),
    fname: str = "",
    ax=None,
):
    """
    Plot the given x y coordinates, appending Pearsons r
    Setting xlim/ylim will affect both plot and R2 calculation
    In other words, plot view mirrors the range for which correlation is calculated
    """
    assert x.shape == y.shape, f"Mismatched shapes: {x.shape} {y.shape}"
    if color is not None:
        assert color.size == x.size
    if one_to_one and (xlim is not None or ylim is not None):
        assert xlim == ylim
    if xlim:
        keep_idx = utils.ensure_arr((x >= xlim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    if ylim:
        keep_idx = utils.ensure_arr((y >= ylim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    # x and y may or may not be sparse at this point
    assert x.shape == y.shape
    if subset > 0 and subset < x.size:
        logging.info(f"Subsetting to {subset} points")
        random.seed(1234)
        # Converts flat index to coordinates
        indices = np.unravel_index(
            np.array(random.sample(range(np.product(x.shape)), k=subset)), shape=x.shape
        )
        x = utils.ensure_arr(x[indices])
        y = utils.ensure_arr(y[indices])
        if isinstance(color, (tuple, list, np.ndarray)):
            color = np.array([color[i] for i in indices])

    if logscale:
        x = np.log1p(x.cpu())
        y = np.log1p(y.cpu())

    # Ensure correct format
    x = x.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    logging.info(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    logging.info(
        f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}"
    )

    if ax is None:
        fig = plt.figure(dpi=300, figsize=figsize)
        if density_heatmap:
            # https://github.com/astrofrog/mpl-scatter-density
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    if density_heatmap:
        norm = None
        if density_logstretch:
            norm = ImageNormalize(
                vmin=0, vmax=100, stretch=LogStretch(a=density_logstretch)
            )
        ax.scatter_density(x, y, dpi=density_dpi, norm=norm, color="tab:blue")
    else:
        ax.scatter(x, y, alpha=0.2, c=color)

    if one_to_one:
        unit = np.linspace(*ax.get_xlim())
        ax.plot(unit, unit, linestyle="--", alpha=0.5, label="$y=x$", color="grey")
        ax.legend()
    ax.set(
        xlabel=xlabel + (" (log)" if logscale else ""),
        ylabel=ylabel + (" (log)" if logscale else ""),
        title=(title + f" ($r={pearson_r:.2f}$)").strip(),
    )
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)

    if fig is not None and fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")

    return fig






def ensure_arr(x) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def is_integral_val(x) -> bool:
    """
    Check if value(s) can be cast as integer without losing precision
    >>> is_integral_val(np.array([1., 2., 3.]))
    True
    >>> is_integral_val(np.array([1., 2., 3.5]))
    False
    """
    if isinstance(x, (np.ndarray, scipy.sparse.csr_matrix)):
        x_int = x.astype(int)
    else:
        x_int = int(x)
    residuals = x - x_int
    if isinstance(residuals, scipy.sparse.csr_matrix):
        residuals = ensure_arr(residuals[residuals.nonzero()])
    return np.all(np.isclose(residuals, 0))


def get_file_extension_no_gz(fname: str) -> str:
    """
    Get the filename extension (not gz)
    >>> get_file_extension_no_gz("foo.txt.gz")
    'txt'
    >>> get_file_extension_no_gz("foo.bar")
    'bar'
    >>> get_file_extension_no_gz("foo")
    ''
    """
    assert fname, f"Got empty input"
    retval = ""
    while fname and (not retval or retval == ".gz"):
        fname, ext = os.path.splitext(fname)
        if not ext:
            break  # Returns empty string
        if ext != ".gz":
            retval = ext
    return retval.strip(".")


def get_ad_reader(fname: str, ft_type: str) -> Callable:
    """Return the function that when called, returns an AnnData object"""
    # Modality is only used for reading h5 files
    ext = get_file_extension_no_gz(fname)
    if ext == "h5":
        pfunc = functools.partial(sc.read_10x_h5, gex_only=False)
        if not ft_type:
            return pfunc

        def helper(fname, pfunc, ft_type):
            a = pfunc(fname)
            return a[:, a.var["feature_types"] == ft_type]

        return functools.partial(helper, pfunc=pfunc, ft_type=ft_type)
    elif ext == "h5ad":
        return sc.read_h5ad
    elif ext == "csv":
        return sc.read_csv
    elif ext in ("tsv", "txt"):
        return sc.read_text
    else:
        raise ValueError("Could not determine reader for {fname}")


def sc_read_multi_files(
    fnames: List[str],
    reader: Callable = None,
    feature_type: str = "",
    transpose: bool = False,
    var_name_sanitization: Callable = None,
    join: str = "inner",
) -> AnnData:
    """Given a list of files, read the adata objects and concatenate"""
    # var name sanitization lets us make sure that variable name conventions are consistent
    assert fnames
    for fname in fnames:
        assert os.path.isfile(fname), f"File does not exist: {fname}"
    if reader is None:  # Autodetermine reader type
        parsed = [get_ad_reader(fname, feature_type)(fname) for fname in fnames]
    else:  # Given a fixed reader
        parsed = [reader(fname) for fname in fnames]
    if transpose:
        # h5 reading automatically transposes
        parsed = [
            p.T if get_file_extension_no_gz(fname) != "h5" else p
            for p, fname in zip(parsed, fnames)
        ]

    # Log and check genomes
    for f, p in zip(fnames, parsed):
        logging.info(f"Read in {f} for {p.shape} (obs x var)")
    genomes_present = set(
        g
        for g in itertools.chain.from_iterable(
            [p.var["genome"] for p in parsed if "genome" in p.var]
        )
        if g
    )

    # Build concatenated output
    assert len(genomes_present) <= 1, f"Got more than one genome: {genomes_present}"
    for fname, p in zip(fnames, parsed):  # Make variable names unique and ensure sparse
        if var_name_sanitization:
            p.var.index = pd.Index([var_name_sanitization(i) for i in p.var_names])
        p.var_names_make_unique()
        p.X = scipy.sparse.csr_matrix(p.X)
        p.obs["source_file"] = fname
    retval = parsed[0]
    if len(parsed) > 1:
        retval = retval.concatenate(*parsed[1:], join=join)
    return retval


def sc_read_10x_h5_ft_type(fname: str, ft_type: str) -> AnnData:
    """Read the h5 file, taking only features with specified ft_type"""
    assert fname.endswith(".h5")
    parsed = sc.read_10x_h5(fname, gex_only=False)
    parsed.var_names_make_unique()
    assert ft_type in set(
        parsed.var["feature_types"]
    ), f"Given feature type {ft_type} not in included types: {set(parsed.var['feature_types'])}"

    retval = parsed[
        :,
        [n for n in parsed.var_names if parsed.var.loc[n, "feature_types"] == ft_type],
    ]
    return retval


def extract_file(fname, overwrite: bool = False) -> str:
    """Extracts the file and return the path to extracted file"""
    out_fname = os.path.abspath(fname.replace(".gz", ""))
    if os.path.isfile(out_fname):  # If the file already
        # If the file already exists and we aren't overwriting, do nothing
        if not overwrite:
            return out_fname
        os.remove(out_fname)
    with open(out_fname, "wb") as sink, gzip.GzipFile(fname) as source:
        sink.write(source.read())
    return out_fname


def read_delimited_file(
    fname: str, delimiter: str = "\n", comment: str = "#"
) -> List[str]:
    """Read the delimited (typically newline) file into a list"""
    with open(fname) as source:
        contents = source.read().strip()
        retval = contents.split(delimiter)
    # Filter out comment entries
    retval = [item for item in retval if not item.startswith(comment)]
    return retval


@functools.lru_cache(maxsize=2, typed=True)
def read_gtf_gene_to_pos(
    fname: str = HG38_GTF,
    acceptable_types: List[str] = None,
    addtl_attr_filters: dict = None,
    extend_upstream: int = 0,
    extend_downstream: int = 0,
) -> Dict[str, Tuple[str, int, int]]:
    """
    Given a gtf file, read it in and return as a ordered dictionary mapping genes to genomic ranges
    Ordering is done by chromosome then by position
    """
    # https://uswest.ensembl.org/info/website/upload/gff.html
    gene_to_positions = collections.defaultdict(list)
    gene_to_chroms = collections.defaultdict(set)

    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            start = int(start)
            end = int(end)
            assert (
                start <= end
            ), f"Start {start} is not less than end {end} for {gene} with strand {strand}"
            if extend_upstream:
                if strand == "+":
                    start -= extend_upstream
                else:
                    end += extend_upstream
            if extend_downstream:
                if strand == "+":
                    end += extend_downstream
                else:
                    start -= extend_downstream

            gene_to_positions[gene].append(start)
            gene_to_positions[gene].append(end)
            gene_to_chroms[gene].add(chrom)

    slist = sortedcontainers.SortedList()
    for gene, chroms in gene_to_chroms.items():
        if len(chroms) != 1:
            logging.warn(
                f"Got multiple chromosomes for gene {gene}: {chroms}, skipping"
            )
            continue
        positions = gene_to_positions[gene]
        t = (chroms.pop(), min(positions), max(positions), gene)
        slist.add(t)

    retval = collections.OrderedDict()
    for chrom, start, stop, gene in slist:
        retval[gene] = (chrom, start, stop)
    return retval


@functools.lru_cache(maxsize=2)
def read_gtf_gene_symbol_to_id(
    fname: str = HG38_GTF,
    acceptable_types: List[str] = None,
    addtl_attr_filters: dict = None,
) -> Dict[str, str]:
    """Return a map from easily readable gene name to ENSG gene ID"""
    retval = {}
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            gene_id = attr_dict["gene_id"].strip('"')
            retval[gene] = gene_id

    return retval


@functools.lru_cache(maxsize=8)
def read_gtf_pos_to_features(
    fname: str = HG38_GTF,
    acceptable_types: Iterable[str] = [],
    addtl_attr_filters: dict = None,
) -> Dict[str, itree.IntervalTree]:
    """Return an intervaltree representation of the gtf file"""
    acceptable_types = set(acceptable_types)
    retval = collections.defaultdict(itree.IntervalTree)
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            line = line.decode()
            if line.startswith("#"):
                continue
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            start = int(start)
            end = int(end)
            if start >= end:
                continue
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [
                    [u.strip('"') for u in t.strip().split(" ", 1)]
                    for t in attrs.strip().split(";")
                    if t
                ]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            assert (
                "entry_type" not in attr_dict
                and "entry_class" not in attr_dict
                and "entry_strand" not in attr_dict
            )
            attr_dict["entry_type"] = entry_type
            attr_dict["entry_class"] = entry_class
            attr_dict["entry_strand"] = strand
            retval[chrom][int(start) : int(end)] = attr_dict
    return retval


def get_device(i: int = None) -> str:
    """Returns the i-th GPU if GPU is available, else CPU"""
    if torch.cuda.is_available() and isinstance(i, int):
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(d)
    else:
        d = torch.device("cpu")
    return d


def is_numeric(x) -> bool:
    """Return True if x is numeric"""
    try:
        x = float(x)
        return True
    except ValueError:
        return False


def is_all_unique(x: Iterable[Any]) -> bool:
    """
    Return whether the given iterable is all unique
    >>> is_all_unique(['x', 'y'])
    True
    >>> is_all_unique(['x', 'x', 'y'])
    False
    """
    return len(set(x)) == len(x)


def shifted_sigmoid(x, center: float = 0.5, slope: float = 25):
    """Compute a shifted sigmoid with configurable center and slope (steepness)"""
    return 1.0 / (1.0 + np.exp(slope * (-x + center)))


def unit_rescale(vals):
    """Rescale the given values to be between 0 and 1"""
    vals = np.array(vals).astype(float)
    denom = float(np.max(vals) - np.min(vals))
    retval = (vals - np.min(vals)) / denom
    assert np.alltrue(retval <= 1.0) and np.alltrue(retval >= 0.0)
    return retval


def split_df_by_col(df: pd.DataFrame, col: str) -> List[pd.DataFrame]:
    """Splits the dataframe into multiple dataframes by value of col"""
    unique_vals = set(df[col])

    retval = {}
    for v in unique_vals:
        df_sub = df[df[col] == v]
        retval[v] = df_sub
    return retval
    

def log_zinb_positive(x, mu, theta, pi, eps=1E-6):
    """
    Note: All inputs are torch Tensors

    log likelihood (scalar) of a minibatch according to a zinb model.

    Notes:
    We parametrize the bernoulli using the logits, hence the softplus functions appearing
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    theta = torch.clamp(theta, max=1e5)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    # first part with zero cases
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    # second part with non-zero cases
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    # total loss
    res = mul_case_zero + mul_case_non_zero
    return res




