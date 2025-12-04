import time
import csv

gammas = [0.6, 0.9, 0.1]
modes = ["value", "policy"]

grid = [
    ["S", "_", "_", "X", "_"],
    ["_", "X", "_", "_", "_"],
    ["_", "_", "X", "_", "_"],
    ["X", "_", "_", "_", "G"],
    ["_", "_", "X", "_", "_"]
]

P0, states, pos2idx, _, _, start, _, _ = build_transition_model(grid)
results = []

for gamma in gammas:
    for mode in modes:
        print(f"calculating with gamma={gamma} using: {mode}")

        start_time = time.time()
        out = run_agent(grid, gamma=gamma, theta=1e-4, max_iters=10000, mode=mode)
        end_time = time.time()

        results.append({
            "gamma": gamma,
            "mode": mode,
            "iterations": out["iterations"],
            "eval_iters": out["eval_iters"],
            "runtime_sec": end_time - start_time,
            "path": extract_path(out["policy"], states, pos2idx, start, grid),
            "policy": out["policy"],
            "value_table": out["value_table"]
        })

with open("agent_experiments.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["gamma","mode","iterations","eval_iters","runtime_sec","path","policy","value_table"]
    )
    writer.writeheader()
    for r in res
