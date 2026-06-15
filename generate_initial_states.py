from utils import generate_samples
import numpy as np
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N_AGENTS = args.n_agents
    GRID_SIZE = args.grid_size
    SEED = args.seed

    states = generate_samples(N_AGENTS, 2, SEED)
    scaled_states = np.zeros_like(states)
    for i in range(2):
        scaled_states[:, i] = states[:, i] * (GRID_SIZE - 1) + 1

    save_path = f"data/initial_states.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(scaled_states, f)

    print(f"\nSuccesfully generated {scaled_states.shape[0]} unique initial agent states.")
    print(f"Results saved to: {save_path}.")