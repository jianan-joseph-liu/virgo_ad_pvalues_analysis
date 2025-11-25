"""
Post-process Bilby result to add PSD normalization correction to both
the signal and noise evidences.

We assume:
  - You already have Bilby `result` loaded (from a completed run).
  - You already have the interferometers (with PSDs) constructed.
  - The waveform_generator is accessible from the result.

The LnL we used:
    lnL = -0.5 * sum_i sum_f [ |d_i(f) - h_i(f)|^2 / S_n_i(f) ]

The correct LnL:
    Lnl = -0.5 * sum_i sum_f [ |d_i(f) - h_i(f)|^2 / S_n_i(f) ] + (1/2) * sum_i sum_f log(S_n_i(f))

The correction term is:
    Δ = -  sum_i sum_f log(Sn_i(f))
which is constant for a given PSD model and affects the evidence normalization.
"""

import numpy as np
import bilby
import os

logger = bilby.core.utils.logger


def psd_ln_term(interferometers, duration):
    """Compute the PSD normalization term (?^'2/T * I? log S_n(f))."""
    
    correction = 0.0
    for ifo in interferometers:
        psd = np.asarray(ifo.power_spectral_density_array, dtype=float)
        mask = getattr(ifo, "frequency_mask", None)
        if mask is not None:
            psd = psd[mask]
        finite_mask = np.isfinite(psd)
        positive_mask = psd > 0.0
        valid_mask = finite_mask & positive_mask
        if not np.all(valid_mask):
            n_invalid = np.count_nonzero(~valid_mask)
            logger.warning(
                "PSD for %s contains %d/%d non-positive or non-finite bins; ignoring them for lnZ correction.",
                ifo.name,
                n_invalid,
                psd.size,
            )
        if not np.any(valid_mask):
            raise ValueError(
                f"PSD for {ifo.name} contains no positive finite samples; cannot compute lnZ correction."
            )
        safe_psd = psd[valid_mask]
        correction -=  np.sum(np.log(safe_psd))
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
    corrected_summary = {}
    print("Applying PSD normalization corrections...")
    for method in methods:
        corrected = apply_psd_correction(results[method], interferometers[method], duration=duration)
        print(
            f"{method.upper()} PSD normalization correction: "
            f"{corrected['psd_norm_correction']:.3f}"
        )
        logZ_corr = corrected["log_evidence"]
        logZ_noise_corr = corrected["log_noise_evidence"]
        logBF_signal_noise = logZ_corr - logZ_noise_corr
        print(f"{method.upper()} corrected logZ: {logZ_corr:.3f}")
        print(f"{method.upper()} corrected logZ_noise: {logZ_noise_corr:.3f}")
        print(f"{method.upper()} corrected logBF (signal - noise): {logBF_signal_noise:.3f}")
        corrected_summary[method] = {
            "log_evidence": logZ_corr,
            "log_noise_evidence": logZ_noise_corr,
            "log_bayes_factor": logBF_signal_noise,
        }

    # Save corrected logZ values as a file
    with open(os.path.join(outdir, "corrected_logZ.txt"), "w") as f:
        f.write("method,log_evidence,log_noise_evidence,log_bayes_factor\n")
        for method in methods:
            summary = corrected_summary[method]
            f.write(
                f"{method},{summary['log_evidence']:.6f},"
                f"{summary['log_noise_evidence']:.6f},{summary['log_bayes_factor']:.6f}\n"
            )

    return corrected_summary
