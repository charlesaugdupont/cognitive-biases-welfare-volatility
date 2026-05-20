import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from model import utility
from functools import partial
from multiprocessing import Pool
from scipy.signal import detrend
from experiment import unpack_and_dequantize

def process_file(f_name, directory, power_threshold_ratio=0.20, discard_steps=3000):
    """
    Load a file, compute utility, and return dominant frequencies
    using a robust method with windowing and power thresholding.
    """
    with open(os.path.join(directory, f_name), "rb") as f:
        res = pickle.load(f)

    grid_size = res["params"]["N"]
    w = unpack_and_dequantize(res["wealth"][:, discard_steps:], grid_size=grid_size)
    h = unpack_and_dequantize(res["health"][:, discard_steps:], grid_size=grid_size)
    u = utility(w, h, res["params"]["alpha"])

    dominant_frequencies = np.full(w.shape[0], np.nan)
    dominant_amplitudes = np.full(w.shape[0], np.nan)

    # Apply detrending
    u = detrend(u, axis=1)

    # Apply Hann window
    N = u.shape[1]
    window = np.hanning(N)
    u_windowed = u * window

    # FFT with proper normalization (divide by N and window RMS correction)
    window_rms = np.sqrt(np.mean(window ** 2))
    fft_vals_complex = np.fft.fft(u_windowed, axis=1) / (N * window_rms)
    fft_freqs = np.fft.fftfreq(N, 1)

    # Positive frequencies only
    pos_mask = fft_freqs > 0
    fft_freqs_pos = fft_freqs[pos_mask]

    # Power spectral density
    psd = np.abs(fft_vals_complex[:, pos_mask])**2

    # Denoise by thresholding
    peak_power_per_agent = np.max(psd, axis=1, keepdims=True)
    peak_power_per_agent[peak_power_per_agent == 0] = 1
    power_threshold = power_threshold_ratio * peak_power_per_agent
    psd[psd < power_threshold] = 0

    # Find dominant frequency and corresponding amplitude
    dominant_idx = np.argmax(psd, axis=1)
    dominant_freq_vals = fft_freqs_pos[dominant_idx]
    dominant_power_vals = psd[np.arange(psd.shape[0]), dominant_idx]

    # Handle zero-power cases
    zero_power_mask = np.max(psd, axis=1) == 0
    dominant_freq_vals[zero_power_mask] = np.nan
    dominant_power_vals[zero_power_mask] = np.nan

    # Store results
    dominant_frequencies = dominant_freq_vals
    dominant_amplitudes = 2 * np.sqrt(dominant_power_vals)

    return {
        "frequencies": dominant_frequencies, 
        "amplitudes": dominant_amplitudes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--cpt-weight-function", type=str)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    MAX_WORKERS = args.max_workers
    BETA = args.beta
    FUNC = args.cpt_weight_function
    MODEL = args.model

    # check that a valid model is passed
    if MODEL not in ["cpt", "eut", "pt", "lambda_bifurcation"]:
        raise Exception(f"Model name is invalid.")
    
    if MODEL == "cpt":
        if FUNC not in ["prelec", "kt", "ge"]:
            raise Exception(f"CPT weighting function is missing or invalid.")
        
    if MODEL in ["eut", "pt", "lambda_bifurcation"]:
        model_str = f"{MODEL}_{str(BETA).split(".")[1]}"
        model_dir = f"data/{MODEL}/{model_str}" 
    else:
        model_str = f"{MODEL}_{FUNC}_{str(BETA).split(".")[1]}"
        model_dir = f"data/{MODEL}/{model_str}"
    
    file_list = os.listdir(model_dir+"/raw")
    with Pool(MAX_WORKERS) as pool:
        process_func = partial(process_file, directory=model_dir+"/raw")
        results = list(tqdm(pool.imap(process_func, file_list), total=len(file_list)))

    with open(f"{model_dir}/dominant_frequencies_amplitudes.pickle", "wb") as f:
        pickle.dump(results, f)