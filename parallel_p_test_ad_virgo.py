import numpy as np
import sys
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
#import scipy.signal as sig
from src.sgvb_psd.psd_estimator import PSDEstimator
from statsmodels.stats.diagnostic import normal_ad
import h5py
import scipy.special

if len(sys.argv) < 2:
    print("Usage: python p_test_ad_virgo.py <segment_id>")
    sys.exit(1)

segment_id = int(sys.argv[1])
print(f"Processing segment {segment_id}")

srate = 2048
n_segments = 1000
gps_start = 1261872018  # UTC 0:0:0, January 1, 2020
gps_end = gps_start + 20 * 24 * 3600  # UTC 0:0:0, January 20, 2020
seg_len=4
welch_blocks=32

v1_flag = DataQualityFlag.fetch_open_data('V1_DATA', gps_start, gps_end)

gps_segments = np.array([[int(seg[0]), int(seg[1])]
                         for seg in v1_flag.active])

def pick_segments(seg_array, n_segments=1000, seg_len=4, 
                  min_gap=1000, welch_blocks=32):
    picked = []
    last_end = -np.inf
    offset = welch_blocks * seg_len
    
    for start, stop in seg_array:
        t = max(start + offset, last_end + min_gap)
        
        while t + seg_len <= stop:
            picked.append((t, t + seg_len))
            if len(picked) == n_segments:
                return np.array(picked)
            last_end = t + seg_len
            t = last_end + min_gap
    return np.array(picked)  

analysis_segments = pick_segments(gps_segments)

offset = welch_blocks * seg_len
welch_segments = np.column_stack([analysis_segments[:, 0] - offset,
                                analysis_segments[:, 0]])


'''
# Prepare downsampled dataset for 128s
'''
start, stop = welch_segments[segment_id]

ts = TimeSeries.fetch_open_data('V1', start, stop, cache=False)
downsampled_ts = ts.value[::2]
length_seg = len(downsampled_ts)//welch_blocks   # length for each block, 8192

segments = downsampled_ts.reshape(welch_blocks, length_seg)
win = np.hanning(length_seg)                # shape (8192,)
windowed_data = segments * win        # shape (32, 8192) for SGVB


'''
# estimate PSD by SGVB
'''
imput_data = windowed_data.reshape(-1, 1)

# use SGVB to estimate the PSD for whitened data in time domain
psd_estimator = PSDEstimator(
    x=imput_data,
    N_theta=500,
    nchunks=32,
    ntrain_map=10000,
    N_samples=500,
    fs=srate,
    max_hyperparm_eval=1,
    degree_fluctuate=500,
    n_elbo_maximisation_steps=600,
    fmax_for_analysis=srate/2
)

result = psd_estimator.run(lr=0.03)
psd_matrices = psd_estimator.pointwise_ci
psd_matrices = psd_matrices[1,:,0,0]

final_psd = psd_matrices


'''
# find the whitened data based on final_psd
'''
# get the index for the analysis segment
analysis_start, analysis_stop = analysis_segments[segment_id]

# get the data for the analysis segment
analysis_data = TimeSeries.fetch_open_data('V1', analysis_start, analysis_stop, cache=False)

# downsample the dataset, fs = 2048
downsampled_noise = analysis_data.value[::2]

# apply the hanning window to the downsampled_noise
win = np.hanning(len(downsampled_noise))
win_analysis_data = win * downsampled_noise

# get the whitened data
analysis_fd = np.fft.rfft(win_analysis_data)[1:-1] / np.sqrt(srate * len(win_analysis_data))
wh_V1_fd = analysis_fd / np.sqrt(final_psd)


'''
# apply the Anderson-Darling normality test p-values
'''
freq = np.fft.rfftfreq(length_seg, d=1.0 / srate)[1:-1]
index_low = np.where(freq == 20)[0][0]
index_high = np.where(freq == 1020)[0][0]

wh_V1_fd_short = wh_V1_fd[index_low:index_high]

# chunk the wh_V1_f_short by every 8 Hz
bin_width_Hz=8
delta_f = freq[1] - freq[0]
bin_width = int(round(bin_width_Hz / delta_f)) # bin_width=8*4=32

n_bins = len(wh_V1_fd_short) // bin_width
pvalues = []

for i in range(n_bins):
    start_idx = i * bin_width
    end_idx = start_idx + bin_width
    
    chunk = wh_V1_fd_short[start_idx:end_idx]
    chunk_combined = np.concatenate([chunk.real, chunk.imag]) # 64 values/iter
    
    pval = normal_ad(chunk_combined)[1]
    pvalues.append(pval)


'''
# rayleigh_test
'''
wh_V1_td = np.fft.irfft(wh_V1_fd*np.sqrt(srate*(length_seg)), 
                        n=(length_seg))
wh_ts = TimeSeries(wh_V1_td, sample_rate=srate)
rayleigh = wh_ts.rayleigh_spectrum(fftlength=4/64, overlap=0)


'''
apply the Anderson-Darling N(0,1) test p-values
'''
# Anderson-Darling (AD) test
def empirical_cdf(data):
    """ Compute the empirical cumulative distribution function (ECDF). """
    sorted_data = np.sort(data)
    n = len(data)

    def ecdf(x):
        return np.searchsorted(sorted_data, x, side='right') / n

    return ecdf, sorted_data

def anderson_darling_statistic(data):
    """ Compute the Anderson-Darling test statistic for normality. """
    n = len(data)
    ecdf, sorted_data = empirical_cdf(data)

    standardized = sorted_data

    # Compute theoretical normal CDF values
    normal_cdf = 0.5 * (1 + scipy.special.erf(standardized / np.sqrt(2)))  # Standard normal CDF

    # Compute Anderson-Darling test statistic
    s = np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(normal_cdf) + np.log(1 - normal_cdf[::-1])))
    A2 = -n - s / n

    return A2

def anderson_p_value(data, freqs=None, fmin=0, fmax=np.inf):
    """ Approximate the p-value for the Anderson-Darling test for normality. """

    # If provided, cut the frequencies to a min/max value
    if freqs is not None:
        idxs = (freqs > fmin) & (freqs < fmax)
        data = data[idxs]

    # Concatenate the real and imaginary parts together
    data = np.concatenate([data.real, data.imag])

    if len(data) == 0:
        return np.nan

    A2 = anderson_darling_statistic(data)

    critical_values = [
        0.200, 0.300, 0.400, 0.500, 0.576, 0.656, 0.787, 0.918,
        1.092, 1.250, 1.500, 1.750, 2.000, 2.500, 3.000, 3.500,
        4.000, 4.500, 5.000, 6.000, 7.000, 8.000, 10.000
    ]

    significance_levels = [
        0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40,
        0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01, 0.005,
        0.0025, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001
    ]

    # Approximate p-value using interpolation
    if A2 < critical_values[0]:
        pval = significance_levels[0]
    elif A2 > critical_values[-1]:
        pval = significance_levels[-1]
    else:
        pval = np.interp(A2, critical_values, significance_levels)

    return float(pval)


pvalues_N01 = []

for i in range(n_bins):
    start_idx = i * bin_width
    end_idx = start_idx + bin_width
    
    chunk = wh_V1_fd_short[start_idx:end_idx]
    chunk_combined = np.concatenate([chunk.real, chunk.imag]) # 64 values/iter
    
    pval = anderson_p_value(chunk_combined)
    pvalues_N01.append(pval)








with h5py.File(f"virgo_ARSGVB_segment_{segment_id}_results.h5", "w") as hf:
    hf.create_dataset("ad_pvalues", data=pvalues)
    hf.create_dataset("ad_pvalues_N01", data=pvalues_N01)
    hf.create_dataset("rayleigh_test", data=rayleigh)
    hf.create_dataset("final_psd", data=final_psd)


















