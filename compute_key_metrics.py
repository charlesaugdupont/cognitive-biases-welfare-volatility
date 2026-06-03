import argparse
from tqdm.auto import tqdm
from utils import *
from experiment import *

def key_metrics(model_dir):
    mean = []
    gini = []
    sen = []
    P = []
    final_states = []
    data_dir = model_dir + "/raw"
    for f in tqdm(os.listdir(data_dir)):
        if ".DS_Store" in f:
            continue
        with open(os.path.join(data_dir, f), "rb") as file:
            res = pickle.load(file)
        wealth = unpack_and_dequantize(res["wealth"][:,-1], 200)
        health = unpack_and_dequantize(res["health"][:,-1], 200)
        util = utility(wealth, health, alpha=res['params']['alpha'])
        m = mean_util(util)
        g = gini_coeff(util)
        s = m * (1-g)
        mean.append(m)
        gini.append(g)
        sen.append(s)
        p = res["params"]
        if "pt_" in model_dir or "gamma_alpha" in model_dir or "gamma_sweep" in model_dir:
            params = ((p["alpha"], p["rate"], p["A"], p["lambda"], p["gamma"]))
        elif "lambda_bifurcation" in model_dir:
            params = ((p["alpha"], p["rate"], p["A"], p["lambda"]))
        else:
            params = ((p["alpha"], p["rate"], p["A"]))
        P.append(params)
        final_states.append((wealth, health))
    
    with open(f"{model_dir}/sen_welfare", "wb") as f:
        pickle.dump(sen, f)
    with open(f"{model_dir}/gini", "wb") as f:
        pickle.dump(gini, f)
    with open(f"{model_dir}/mean_util", "wb") as f:
        pickle.dump(mean, f)
    with open(f"{model_dir}/params", "wb") as f:
        pickle.dump(P, f)
    with open(f"{model_dir}/final_states", "wb") as f:
        pickle.dump(final_states, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--beta", type=float, required=False, default=0.0)
    parser.add_argument("--cpt-weight-function", type=str)
    parser.add_argument("--data-dir", type=str, required=False, default="")
    args = parser.parse_args()

    MODEL = args.model
    BETA = args.beta
    FUNC = args.cpt_weight_function
    DATA_DIR = args.data_dir

    if DATA_DIR:
        model_dir = DATA_DIR
    
    else:
        if BETA not in [0.95, 0.999]:
            raise Exception(f"Beta value is invalid.")
        
        if MODEL not in ["cpt", "eut", "pt", "lambda_bifurcation", "gamma_alpha"]:
            raise Exception(f"Model name is invalid.")
        
        if MODEL == "cpt":
            if FUNC not in ["prelec", "kt", "ge"]:
                raise Exception(f"CPT weighting function is missing or invalid.")

        if MODEL in ["eut", "pt", "lambda_bifurcation", "gamma_alpha"]:
            model_str = f"{MODEL}_{str(BETA).split(".")[1]}"
            model_dir = f"data/{MODEL}/{model_str}" 
        else:
            model_str = f"{MODEL}_{FUNC}_{str(BETA).split(".")[1]}"
            model_dir = f"data/{MODEL}/{model_str}"
    
    key_metrics(model_dir)