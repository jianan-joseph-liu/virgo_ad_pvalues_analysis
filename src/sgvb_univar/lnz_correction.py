"""
Post-process Bilby result to add PSD normalization correction to both
the signal and noise evidences.

We assume:
  - You already have Bilby `result` loaded (from a completed run).
  - You already have the interferometers (with PSDs) constructed.
  - The waveform_generator is accessible from the result.

The correction term is:
    Δ = - (2 / T) * sum_i sum_f log(Sn_i(f))
which is constant for a given PSD model and affects the evidence normalization.
"""

import numpy as np
import bilby
import os


def psd_ln_term(interferometers, duration):
    """Compute the PSD normalization term (−2/T * Σ log S_n(f))."""
    T = duration
    correction = 0.0
    for ifo in interferometers:
        psd = ifo.power_spectral_density_array
        psd = np.isfinite(psd)
        print("psd has shape:", psd.shape, "np.sum(np.log(psd)) is", np.sum(np.log(psd)))
        print("psd", psd[:10])
        correction -= (2.0 / T) * np.sum(np.log(psd))
    return correction


def apply_psd_correction(result, interferometers, duration=None):
    """
    Compute and apply the PSD normalization correction to:
      - log_evidence
      - log_noise_evidence

    Returns a new corrected result dictionary for safety.
    """
    if duration is None:
        waveform_generator = getattr(result, "waveform_generator", None)
        if waveform_generator is None:
            raise ValueError(
                "Duration must be provided when the result does not carry a waveform generator."
            )
        duration = waveform_generator.duration

    correction = psd_ln_term(interferometers, duration)
    
    # Copy to avoid mutating the original object
    corrected = dict(
        log_evidence=result.log_evidence + correction,
        log_noise_evidence=result.log_noise_evidence + correction,
        psd_norm_correction=correction,
    )
    return corrected



def apply_psd_corrections(results, interferometers, methods, outdir, duration=None):
    logz_corrected = {}
    print("Applying PSD normalization corrections...")
    for method in methods:
        corrected = apply_psd_correction(results[method], interferometers[method], duration=duration)
        print(
            f"{method.upper()} PSD normalization correction: "
            f"{corrected['psd_norm_correction']:.3f}"
        )
        logZ_corr = corrected["log_evidence"]
        print(f"{method.upper()} corrected logZ: {logZ_corr:.3f}")
        logz_corrected[method] = logZ_corr

    logBF_corrected = logz_corrected["sgvb"] - logz_corrected["welch"]
    print(f"Corrected logBF (sgvb - welch) = {logBF_corrected:.3f}, BF = {np.exp(logBF_corrected):.3e}")    

    # save corrected logZ values as a file
    with open(os.path.join(outdir, "corrected_logZ.txt"), "w") as f:
        for method in methods:
            f.write(f"{method} corrected logZ: {logz_corrected[method]:.6f}\n")
        logBF_corrected = logz_corrected["sgvb"] - logz_corrected["welch"]
        f.write(f"Corrected logBF (sgvb - welch) = {logBF_corrected:.6f}, BF = {np.exp(logBF_corrected):.6e}\n")


    return results
