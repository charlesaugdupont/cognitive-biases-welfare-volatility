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
    alpha, rate, A, lambduh, gamma = row

    # compute optimal policy
    policy, params, invest_val, save_val = value_iteration_pt_cpt(
        N=grid_size,
        alpha=alpha,
        gamma=gamma,
        lambduh=lambduh,
        eta=ETA,
        P_H_increase=P_H_INCREASE,
        P_H_decrease=P_H_DECREASE,
        rate=rate,
        A=A,
        theta=THETA,
        beta=beta,
        return_values=True,
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
        "storage_dtype_info": str(storage_dtype),
        "invest_val": invest_val,
        "save_val": save_val
    }

    output_file_name = os.path.join(model_dir, f"{alpha}_{rate}_{A}_{lambduh}_{gamma}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--sweep-gamma", type=str, default="true")
    args = parser.parse_args()

    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    GRID_SIZE = args.grid_size
    BETA = args.beta
    SWEEP_GAMMA = bool(args.sweep_gamma == "true")
    if SWEEP_GAMMA:
        MODEL = f"data/gamma_alpha/cpt_{str(BETA).split(".")[1]}/raw"
    else:
        MODEL = f"data/gamma_alpha/pt_{str(BETA).split(".")[1]}/raw"

    if not os.path.exists(MODEL):
        os.makedirs(MODEL)
    else:
        raise Exception("lambda_bifurcation directory already exists!")

    with open("data/initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    # constants
    A = 0.8
    LAMBDUH = 2.5
    RATE = 4.0

    # simulation samples
    samples = []
    if SWEEP_GAMMA:
        for alpha in np.linspace(0.3, 0.8, 24):
            for gamma in np.linspace(0.1, 0.8, 24):
                samples.append(
                    (alpha, RATE, A, LAMBDUH, gamma)
                )
    else:
        for alpha in np.linspace(0.3, 0.8, 24):
            samples.append(
                (alpha, RATE, A, LAMBDUH, 1.0)
            )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, MODEL, GRID_SIZE, initial_states, BETA) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()