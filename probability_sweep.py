from experiment import quantize_and_pack, THETA, ETA, SEED
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

np.random.seed(SEED)
random.seed(SEED)

REGIMES = {
    "high_welfare": {"alpha": 0.30, "gamma": 0.80},
    "transition": {"alpha": 0.65, "gamma": 0.60},
    "low_welfare": {"alpha": 0.70, "gamma": 0.20},
}

# Fixed parameters
A = 0.8
LAMBDUH = 2.5
RATE = 4.0

def process_row(row, n_steps, model_dir, grid_size, initial_states, beta):
    alpha, rate, A, lambduh, gamma, p_h_increase, p_h_decrease = row

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

    output_file_name = os.path.join(
        model_dir,
        f"{p_h_increase}_{p_h_decrease}.pickle"
    )
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.95)
    args = parser.parse_args()

    N_STEPS    = args.n_steps
    MAX_WORKERS = args.max_workers
    GRID_SIZE  = args.grid_size
    BETA       = args.beta

    with open("data/initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    p_values = np.linspace(0.05, 0.95, 15)

    for regime_name, regime_params in REGIMES.items():
        alpha = regime_params["alpha"]
        gamma = regime_params["gamma"]

        model_dir = f"data/probability_sweep/{regime_name}/raw"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            raise Exception(f"Directory {model_dir} already exists!")

        samples = []
        for p_h_increase in p_values:
            for p_h_decrease in p_values:
                samples.append(
                    (alpha, RATE, A, LAMBDUH, gamma, p_h_increase, p_h_decrease)
                )

        print(f"Running regime: {regime_name} (alpha={alpha}, gamma={gamma}) — {len(samples)} simulations")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_row, row, N_STEPS, model_dir, GRID_SIZE, initial_states, BETA)
                for row in samples
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=regime_name):
                future.result()
