import copy
import time
from agent import run_agent, build_transition_model
from stochastic_agent import run_agent as run_stochastic_agent, build_transition_model as build_stochastic_transition_model

# --- Helper to extract path ---
def run_agent_extract_path(policy, grid, pos2idx, start):
    path = []
    pos = start
    visited = set()
    while True:
        if pos in visited: break
        visited.add(pos)
        path.append(pos)
        i, j = pos
        if grid[i][j] == "G": break
        idx = pos2idx[pos]
        action = policy[idx]
        di, dj = {"U":(-1,0),"R":(0,1),"D":(1,0),"L":(0,-1)}[action]
        pos = (i+di, j+dj)
    return path

# --- Grid definition ---
grid = [
    ["S", "_", "_", "X", "_"],
    ["_", "X", "_", "_", "_"],
    ["_", "_", "X", "_", "_"],
    ["X", "_", "_", "_", "G"],
    ["_", "_", "X", "_", "_"]
]

H, W = len(grid), len(grid[0])
results_visual = []

# --- Run deterministic agent ---
P0, states, pos2idx, _, _, start, _, _ = build_transition_model(grid)
gammas = [0.6, 0.1, 0.9]
modes = ["value", "policy"]

for gamma in gammas:
    for mode in modes:
        print(f"Running deterministic agent with gamma={gamma}, mode={mode}")
        t0 = time.time()
        result = run_agent(grid, gamma=gamma, mode=mode)
        t1 = time.time()
        result["runtime_sec"] = t1 - t0
        result["path"] = run_agent_extract_path(result["policy"], grid, pos2idx, start)
        result["gamma"] = gamma
        result["mode"] = mode
        result["mdp"] = "Deterministic"
        results_visual.append(result)

# --- Run stochastic agent ---
gamma = 0.6
for mode in modes:
    print(f"Running stochastic agent with gamma={gamma}, mode={mode}")
    t0 = time.time()
    result = run_stochastic_agent(grid, gamma=gamma, mode=mode)
    t1 = time.time()
    result["runtime_sec"] = t1 - t0
    result["path"] = run_agent_extract_path(result["policy"], grid, pos2idx, start)
    result["gamma"] = gamma
    result["mode"] = mode
    result["mdp"] = "Stochastic"
    results_visual.append(result)

# --- Create human-readable report ---
report_lines = []

for idx, res in enumerate(results_visual, 1):
    report_lines.append(f"==================== Run {idx} ====================")
    report_lines.append(f"MDP: {res['mdp']} Mode: {res['mode']} iteration, Gamma: {res['gamma']}")
    report_lines.append(f"Iterations: {res['iterations']}, Eval iterations: {res.get('eval_iters',0)}, Runtime: {res['runtime_sec']:.6f} s")
    
    # Policy Grid
    report_lines.append("Policy Grid:")
    for i in range(H):
        row = [res["policy"][pos2idx[(i,j)]] if (i,j) in pos2idx else "X" for j in range(W)]
        report_lines.append(" ".join(row))
    
    # Value Table Grid
    # Value Table Grid
    report_lines.append("Value Table Grid:")
    for i in range(H):
        row = []
        for j in range(W):
            if (i,j) in pos2idx:
                idx_state = pos2idx[(i,j)]
                val = res["value_table"][idx_state]
                row.append(f"{val:5.1f}")  # <- 1 decimal point
            else:
                row.append(" NaN")  # obstacle
        report_lines.append(" ".join(row))

    # Path
    report_lines.append(f"Path: {res['path']}\n")

# --- Save report ---
report_text = "\n".join(report_lines)
with open("agent_results_readable.txt", "w") as f:
    f.write(report_text)

print("Saved human-readable results to agent_results_readable.txt")
