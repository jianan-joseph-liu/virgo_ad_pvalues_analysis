import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import bilby
from bilby.core.utils import random

# -----------------------
# Configuration
# -----------------------

random.seed(123)

N_INJECTIONS = 1000
SNR_MIN = 10.0
SNR_MAX = 50.0

duration = 4.0
sampling_frequency = 1024.0

GEOCENT_TIME = 1126259642.413  # fixed reference GPS time

outdir = "snr_prior_study"
os.makedirs(outdir, exist_ok=True)

label = "bbh_snr_window"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# -----------------------
# Waveform + IFO setup
# -----------------------

waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# LVK design sensitivity network: H1, L1
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=GEOCENT_TIME - duration / 2,
)

# -----------------------
# Priors
# -----------------------

priors = bilby.gw.prior.BBHPriorDict()

# Broad distance prior — we enforce SNR window via rejection sampling
priors["luminosity_distance"] = bilby.core.prior.Uniform(
    minimum=100,
    maximum=5000,
    name="luminosity_distance",
    unit="Mpc",
)

# Fix geocenter time
priors["geocent_time"] = bilby.core.prior.DeltaFunction(
    peak=GEOCENT_TIME, name="geocent_time"
)

# -----------------------
# Helper: compute network SNR
# -----------------------

def compute_network_snr(parameters):
    """Compute network SNR using optimal_snr_squared."""
    h_pols = waveform_generator.frequency_domain_strain(parameters)
    snr2 = 0.0
    for ifo in ifos:
        h_det = ifo.get_detector_response(h_pols, parameters)
        snr2 += ifo.optimal_snr_squared(h_det)
    return np.sqrt(snr2)

# -----------------------
# Helper: rejection sampling
# -----------------------

def draw_injection_with_snr_window(
    priors, snr_min=SNR_MIN, snr_max=SNR_MAX, max_tries=400
):
    """Sample priors until SNR is between snr_min and snr_max."""
    for _ in range(max_tries):
        params = priors.sample()
        params["geocent_time"] = GEOCENT_TIME

        snr = compute_network_snr(params)

        if snr_min <= snr <= snr_max:
            return params, snr

    raise RuntimeError("Could not find injection in SNR window after many tries.")

# -----------------------
# Main injection loop
# -----------------------

injections = []

print(f"\nGenerating {N_INJECTIONS} injections with {SNR_MIN} < SNR < {SNR_MAX}\n")

for _ in tqdm(range(N_INJECTIONS), desc="Injecting BBHs"):
    params, snr = draw_injection_with_snr_window(priors)
    params = dict(params)
    params["network_snr"] = snr

    # Store per-detector SNRs
    h_pols = waveform_generator.frequency_domain_strain(params)
    for ifo in ifos:
        h_det = ifo.get_detector_response(h_pols, params)
        params[f"{ifo.name}_snr"] = np.sqrt(ifo.optimal_snr_squared(h_det))

    injections.append(params)

# -----------------------
# Save CSV
# -----------------------

df = pd.DataFrame(injections)
csv_path = os.path.join(outdir, "bbh_injections_snr_10_50.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {N_INJECTIONS} injections to {csv_path}")

# -----------------------
# Plot SNR vs distance
# -----------------------

plt.figure(figsize=(7, 5))
plt.scatter(df["luminosity_distance"], df["network_snr"], s=12, alpha=0.6)

plt.xlabel("Luminosity distance [Mpc]")
plt.ylabel("Network SNR")
plt.title("Injected BBHs: Network SNR vs Distance")
plt.grid(alpha=0.3)

figpath = os.path.join(outdir, "snr_vs_distance.png")
plt.savefig(figpath, dpi=150)
plt.close()

print(f"Saved SNR–distance plot to: {figpath}\n")