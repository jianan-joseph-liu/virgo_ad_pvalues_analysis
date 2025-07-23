import numpy as np
import sys
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
import scipy.signal as sig
from spectrum import aryule, arma2psd
from src.sgvb_psd.psd_estimator import PSDEstimator
from statsmodels.stats.diagnostic import normal_ad
import pandas as pd


if len(sys.argv) < 2:
    print("Usage: python p_test_ad_virgo.py <segment_id>")
    sys.exit(1)

segment_id = int(sys.argv[1])
print(f"Processing segment {segment_id}")


srate = 2048
n_segments = 1000
gps_start = 1261872018  # UTC 0:0:0, January 1, 2020
gps_end = gps_start + 15 * 24 * 3600  

v1_flag = DataQualityFlag.fetch_open_data('V1_DATA', gps_start, gps_end)

gps_segments = np.array([[int(seg[0]), int(seg[1])]
                         for seg in v1_flag.active])

def pick_segments(seg_array, n_segments=1000, seg_len=4, min_gap=1000):
    picked = []
    last_end = -np.inf
    for start, stop in seg_array:
        t = max(start, last_end + min_gap)
        while t + seg_len <= stop:
            picked.append((t, t + seg_len))
            if len(picked) == n_segments:
                return np.array(picked)
            last_end = t + seg_len
            t = last_end + min_gap
    return np.array(picked)  

all_segments = pick_segments(gps_segments)


start, stop = all_segments[segment_id]
ts = TimeSeries.fetch_open_data('V1', start, stop, cache=False)
downsampled_noise = ts.value[::2]
window = sig.windows.tukey(len(downsampled_noise), alpha=0.8)
windowed_virgo_noise = downsampled_noise * window

# AR model PSD
ar, variance, coeff_reflection = aryule(windowed_virgo_noise, 20)
ar_PSD = arma2psd(ar, B=None, rho=variance, T=srate,
                  NFFT=len(windowed_virgo_noise), norm=False)[1:len(windowed_virgo_noise)//2]
ar_PSD_v1 = ar_PSD

# compute the whitened data based on ar_PSD_v1
V1_f = np.fft.rfft(windowed_virgo_noise)[1:-1] / np.sqrt(srate * len(windowed_virgo_noise))
wh_V1_f = V1_f / np.sqrt(ar_PSD_v1)

# transform to the time domain, apply the SGVB
wh_V1_td = np.fft.irfft(wh_V1_f * np.sqrt(srate * len(windowed_virgo_noise)),
                        n=len(windowed_virgo_noise))

import_cleaned_data = wh_V1_td[:, np.newaxis]

psd_estimator = PSDEstimator(
    x=import_cleaned_data,
    N_theta=50,
    nchunks=1,
    ntrain_map=10000,
    N_samples=500,
    fs=srate,
    max_hyperparm_eval=1,
    degree_fluctuate=50,
    n_elbo_maximisation_steps=600,
    fmax_for_analysis=srate/2
)

result = psd_estimator.run(lr=0.03)
freq = psd_estimator.freq
psd_matrices = psd_estimator.pointwise_ci

final_psd = ar_PSD_v1 * psd_matrices[1,:,0,0]

# whitened the virgo noise based on the AR + SGVB
wh_V1_f = V1_f / np.sqrt(final_psd)

# apply the Anderson-Darling normality test p-values
index_low = np.where(freq == 20)[0][0]
index_high = np.where(freq == 1020)[0][0]

wh_band = wh_V1_f[index_low:index_high + 1]
re_part = wh_band.real
im_part = wh_band.imag

pval_ad_re = normal_ad(re_part)[1]
pval_ad_im = normal_ad(im_part)[1]

pvals_df = pd.DataFrame({
    "AD_pval": [pval_ad_re, pval_ad_im]   
})

outname = f"virgo_ad_pvalues_segment_{segment_id}.csv"
pvals_df.to_csv(outname, index=False)
























