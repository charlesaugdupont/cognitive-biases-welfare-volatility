from utils import generate_samples
import numpy as np
import argparse
import pickle

PARAMETER_RANGES = {
    "alpha":        [0.30, 0.70],
    "gamma":        [0.50, 0.80],
    "lambda":       [1.00, 2.50],
    "rate":         [1.00, 5.00],
    "A":            [0.05, 0.95]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--model", type=str, default="cpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MODEL = args.model
    if MODEL not in ["cpt", "pt", "eut"]:
        raise Exception(f"Invalid model name: {MODEL}")
    
    SEED = args.seed
    N_SAMPLES = args.n_samples
    samples = generate_samples(N_SAMPLES, len(PARAMETER_RANGES), SEED)

    scaled_samples = np.zeros_like(samples)
    for i, (param, (low, high)) in enumerate(PARAMETER_RANGES.items()):
        if MODEL == "eut" and param in ["gamma", "lambda"]:
             scaled_samples[:, i] = 1
        elif MODEL == "pt" and param == "gamma":
            scaled_samples[:, i] = 1
        else:
            scaled_samples[:, i] = samples[:, i] * (high - low) + low

    save_path = f"data/{MODEL}/{MODEL}_samples.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(scaled_samples, f)

    print(f"\nSuccesfully generated parameter sample with shape {scaled_samples.shape} for '{MODEL}' model")
    print(f"Results saved to: {save_path}")