from experiment import quantize_and_pack, THETA, ETA, SEED, P_H_DECREASE, P_H_INCREASE
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

np.random.seed(SEED)
random.seed(SEED)

PVALS = np.linspace(0.001,0.999,10000)


def process_row(row, n_steps, model_dir, grid_size, initial_states, beta, max_distort):
    alpha, rate, A, lambduh, gamma = row

    if max_distort:
        # maximally-distorted PH+ and PH- values
        p_h_decrease = PVALS[np.argmin(PVALS - probability_weighting(PVALS, gamma=gamma))]
        p_h_increase = PVALS[np.argmax(PVALS - probability_weighting(PVALS, gamma=gamma))]
    else:
        p_h_increase = P_H_INCREASE
        p_h_decrease = P_H_DECREASE

    # compute optimal policy
    policy, params, invest_val, save_val = value_iteration_pt_cpt(
        N=grid_size,
        alpha=alpha,
        gamma=gamma,
        lambduh=lambduh,
        eta=ETA,
        P_H_increase=p_h_increase,
        P_H_decrease=p_h_decrease,
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
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--model", type=str, default="cpt")
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--cpt-weight-function", type=str)
    parser.add_argument("--max-distort", type=str, default="true")
    args = parser.parse_args()

    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    GRID_SIZE = args.grid_size
    BETA = args.beta
    MODEL = args.model
    FUNC = args.cpt_weight_function
    MAX_DISTORT = bool(args.max_distort.lower() == "true")

    # check that a valid model is passed
    if MODEL not in ["cpt", "pt"]:
        raise Exception(f"Model name is invalid.")
    
    if MODEL == "cpt":
        if FUNC not in ["prelec", "kt", "ge"]:
            raise Exception(f"CPT weighting function is missing or invalid.")
    
    if MAX_DISTORT and MODEL == "cpt":
        model_dir = f"data/gamma_sweep/{MODEL}_max_distort/raw/"
    else:
        MAX_DISTORT = False
        model_dir = f"data/gamma_sweep/{MODEL}_no_distort/raw/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        raise Exception("gamma_sweep directory already exists!")

    with open("data/initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    # constants
    A = 0.8
    LAMBDUH = 2.5
    RATE = 4.0

    # sweeped values
    alpha_vals = np.linspace(0.3, 0.8, 20)
    gamma_vals = np.linspace(0.10, 0.50, 20)

    # simulation samples
    if MODEL == "cpt":
        samples = []
        for alpha in alpha_vals:
            for gamma in gamma_vals:
                samples.append(
                    (alpha, RATE, A, LAMBDUH, gamma)
                )
    else:
        samples = []
        for alpha in alpha_vals:
            samples.append(
                (alpha, RATE, A, LAMBDUH, 1.0)
            )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, model_dir, GRID_SIZE, initial_states, BETA, MAX_DISTORT) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()