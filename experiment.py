from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

# constants
THETA = 0.88
ETA = 0.88
BETA = 0.999
P_H_INCREASE = 0.95
P_H_DECREASE = 0.05
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def quantize_and_pack(data: np.ndarray, grid_size: int, dtype=np.uint16):
    """
    Quantizes continuous data from [1, grid_size] to the integer range of dtype.

    Args:
        data (np.ndarray): The floating-point array to convert.
        grid_size (int): The maximum value of the original range.
        dtype: The target integer type (e.g., np.uint8 or np.uint16).

    Returns:
        np.ndarray: The quantized data as the specified integer type.
    """
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        shifted_data = data - 1
        scale_factor = max_val / (grid_size - 1)
        scaled_data = np.round(shifted_data * scale_factor)
        return scaled_data.astype(dtype)
    else:
        raise ValueError("dtype must be an integer type.")

def unpack_and_dequantize(data: np.ndarray, grid_size: int, dtype=np.uint16):
    """
    De-quantizes integer data back to its approximate float value in [1, grid_size].
    This is for use in your analysis code, not the simulation script.
    """
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(dtype).max
        scale_factor = max_val / (grid_size - 1)
        # Convert back to float and reverse the scaling
        unscaled_data = data.astype(np.float32) / scale_factor
        # Shift back to the original [1, grid_size] range
        return unscaled_data + 1
    else:
        raise ValueError("data must be an integer array.")

def process_row(row, n_steps, model, grid_size, initial_states):
    # unpack parameter set
    alpha, gamma, lambduh, rate, A = row

    func = probability_weighting
    if model == "cpt_revised_prelec":
        func = probability_weighting_prelec
    elif model == "cpt_revised_ge":
        func = probability_weighting_goldstein_einhorn

    # compute policy
    policy, params = compute_optimal_policy(
        model=model,
        N=grid_size,
        alpha=alpha,
        gamma=gamma,
        lambduh=lambduh,
        eta=ETA if model in ["pt", "cpt"] else 1.0,
        P_H_increase=P_H_INCREASE,
        P_H_decrease=P_H_DECREASE,
        rate=rate,
        A=A,
        theta=THETA if model in ["pt", "cpt"] else 1.0,
        beta=BETA,
        weighting_function=func
    )

    # run agent simulations
    wealth, health = simulate(params, policy, n_steps, initial_states)

    storage_dtype = np.uint16
    result = {
        "params": params,
        "wealth": quantize_and_pack(wealth, grid_size, storage_dtype),
        "health": quantize_and_pack(health, grid_size, storage_dtype),
        "policy": policy.astype(np.uint8),
        "storage_dtype_info": str(storage_dtype)
    }
    output_file_name = os.path.join(model, f"{alpha}_{gamma}_{lambduh}_{rate}_{A}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model", type=str, default="cpt")
    parser.add_argument("--grid-size", type=int, default=200)
    args = parser.parse_args()

    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    MODEL = args.model
    GRID_SIZE = args.grid_size
    
    # check that a valid model is passed
    if MODEL not in ["cpt_revised_kt", "cpt_revised_prelec", "cpt_revised_ge", "eut_revised", "pt_revised"]:
        raise Exception(f"Model name is invalid.")
    
    # load initial agent states
    initial_states_path = "initial_states.pickle"
    if not os.path.exists(initial_states_path):
        raise Exception("Please generate initial agent states with: uv run generate_initial_states.py [--n-agents] [--grid-size] [--seed]")
    with open(initial_states_path, "rb") as f:
        initial_states = pickle.load(f)
    
    # load parameter samples
    samples_path = f"{MODEL.split("_")[0]}_samples_revised.pickle"
    if not os.path.exists(samples_path):
        raise Exception(f"Please a sample of parameter values with: uv run generate_parameter_sample.py --model {MODEL} [--n-samples]  [--seed]")
    with open(samples_path, "rb") as f:
        samples = pickle.load(f)

    # verify that we do not overwrite data
    if not os.path.exists(MODEL):
        os.makedirs(MODEL)
    else:
        raise Exception(f"Output directory for '{MODEL}' model already exists!")

    # construct set of samples and run simulations in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, MODEL, GRID_SIZE, initial_states) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()