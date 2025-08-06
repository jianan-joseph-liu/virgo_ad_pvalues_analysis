"""Module to compute the spectral density given the best surrogate posterior parameters and samples"""
import numpy as np
import tensorflow as tf
from typing import Tuple, List
#from scipy.stats import median_abs_deviation

def compute_psd(
        Xmat_delta:tf.Tensor,
        Xmat_theta:tf.Tensor,
        p_dim:int,
        vi_samples: List[tf.Tensor],
        quantiles=[0.05, 0.5, 0.95],
        psd_scaling=1.0,
        fs=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is used to compute the spectral density given the best surrogate posterior parameters
    :param vi_samples: the surrogate posterior parameters

    Computes:
        1  psd: one instance of the spectral density [1, n-freq, dim, dim]

    """
    delta2_all_s = tf.exp(
        tf.matmul(Xmat_delta, tf.transpose(vi_samples[0], [0, 2, 1]))
    )  # (500, #freq, p)


    D_all = tf.map_fn(
        lambda x: tf.linalg.diag(x), delta2_all_s
    ).numpy()  # (500, #freq, p, p)


    psd_all = D_all
    
    pointwise_ci = __get_pointwise_ci(psd_all, quantiles)
    #uniform_ci = __get_uniform_ci(psd_all, pointwise_ci)
    

    # changing freq from [0, 1/2] to [0, samp_freq/2] (and applying scaling)
    if fs:
        original_fmax = 0.5
        true_fmax = fs / 2
        new_scale = true_fmax / original_fmax
        psd_all = psd_all / new_scale
        pointwise_ci = pointwise_ci / new_scale

    return (psd_all * psd_scaling ** 2,
           pointwise_ci * psd_scaling ** 2)
        

def __get_pointwise_ci(psd_all, quantiles):
    _, num_freq, p_dim, _ = psd_all.shape
    psd_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

    diag_indices = np.diag_indices(p_dim)
    psd_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(
        np.real(psd_all[:, :, diag_indices[0], diag_indices[1]]),
        quantiles,
        axis=0,
    )

    return psd_q




















