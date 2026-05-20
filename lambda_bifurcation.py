from experiment import quantize_and_pack, THETA, ETA, P_H_DECREASE, P_H_INCREASE, SEED
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

np.random.seed(SEED)
random.seed(SEED)

def process_row(row, n_steps, model_dir, grid_size, initial_states, beta):
    alpha, rate, A, lambduh = row

    # compute optimal policy
    policy, params = value_iteration_pt_cpt(
        N=grid_size,
        alpha=alpha,
        gamma=1,
        lambduh=lambduh,
        eta=ETA,
        P_H_increase=P_H_INCREASE,
        P_H_decrease=P_H_DECREASE,
        rate=rate,
        A=A,
        theta=THETA,
        beta=beta,
        weighting_function=probability_weighting
    )

    # run agent simulation
    wealth, health = simulate(params, policy, n_steps, initial_states)

    storage_dtype = np.uint16
    result = {
        "params": params,
        "wealth": quantize_and_pack(wealth, grid_size, storage_dtype),
        "health": quantize_and_pack(health, grid_size, storage_dtype),
        "policy": policy.astype(np.uint8),
        "storage_dtype_info": str(storage_dtype)
    }

    output_file_name = os.path.join(model_dir, f"{alpha}_{rate}_{A}_{lambduh}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--beta", type=float, required=True)
    args = parser.parse_args()

    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    GRID_SIZE = args.grid_size
    BETA = args.beta
    MODEL = f"data/lambda_bifurcation/lambda_bifurcation_{str(BETA).split(".")[1]}/raw"

    if not os.path.exists(MODEL):
        os.makedirs(MODEL)
    else:
        raise Exception("lambda_bifurcation directory already exists!")

    with open("data/initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    # identify PT simulations with average amplitude > 10
    with open(f"data/pt/pt_{str(BETA).split(".")[1]}/dominant_frequencies_amplitudes.pickle", "rb") as f:
        pt_data = pickle.load(f)

    THRESHOLD = 10
    sims = np.array([np.mean(x["amplitudes"]) for x in pt_data])
    with open(f"data/pt/pt_{str(BETA).split(".")[1]}/params", "rb") as f:
        params = np.array(pickle.load(f))[sims>THRESHOLD][:,:-1]    

    # run a sweep of lambda values
    lambda_values = np.linspace(1, 2.5, 9)
    samples = []
    for p in params:
        for L in lambda_values:
            samples.append(
                (p[0], p[1], p[2], L)
            )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, MODEL, GRID_SIZE, initial_states, BETA) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()