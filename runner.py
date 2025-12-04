import time
import copy
import math
from stochastic_agent import run_agent, extract_path, build_transition_model

gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
modes = ["value", "policy"]
num_runs = 500

grid = [
    ["S", "_", "_", "X", "_"],
    ["_", "X", "_", "_", "_"],
    ["_", "_", "X", "_", "_"],
    ["X", "_", "_", "_", "G"],
    ["_", "_", "X", "_", "_"]
]

P0, states, pos2idx, _, _, start, _, _ = build_transition_model(grid)
results_visual = []

H, W = len(grid), len(grid[0])

for gamma in gammas:
    for mode in modes:
        print(f"Running {num_runs} times with gamma = {gamma}, mode = {mode}")

        iterations_list = []
        eval_iters_list = []
        runtime_list = []
        value_tables = []
        final_policy = None
        final_path = None

        for run in range(num_runs):
            start_time = time.time()
            out = run_agent(grid, gamma=gamma, theta=1e-8, max_iters=100000, mode=mode)
            end_time = time.time()

            iterations_list.append(out["iterations"])
            eval_iters_list.append(out["eval_iters"] if out["eval_iters"] else 0)
            runtime_list.append(end_time - start_time)
            value_tables.append(out["value_table"])

            if run == 0:
                final_policy = out["policy"]
                final_path = extract_path(out["policy"], states, pos2idx, start, grid)

        # Compute averages
        avg_iterations = sum(iterations_list) / num_runs
        avg_eval_iters = sum(eval_iters_list) / num_runs
        avg_runtime = sum(runtime_list) / num_runs
        avg_value_table = [sum(col)/num_runs for col in zip(*value_tables)]

        # Compute standard deviations
        std_iterations = math.sqrt(sum((x - avg_iterations) ** 2 for x in iterations_list) / num_runs)
        std_eval_iters = math.sqrt(sum((x - avg_eval_iters) ** 2 for x in eval_iters_list) / num_runs)
        std_runtime = math.sqrt(sum((x - avg_runtime) ** 2 for x in runtime_list) / num_runs)
        std_value_table = [math.sqrt(sum((x - avg) ** 2 for x in col)/num_runs) for col, avg in zip(zip(*value_tables), avg_value_table)]

        # Build visual grid
        vis_grid = copy.deepcopy(grid)
        if final_path:
            start_pos = final_path[0]
            goal_pos = final_path[-1]
            for i, j in final_path[1:-1]:
                vis_grid[i][j] = "*"
            vis_grid[start_pos[0]][start_pos[1]] = "S"
            vis_grid[goal_pos[0]][goal_pos[1]] = "G"

        results_visual.append({
            "gamma": gamma,
            "mode": mode,
            "iterations": avg_iterations,
            "iterations_std": std_iterations,
            "eval_iters": avg_eval_iters,
            "eval_iters_std": std_eval_iters,
            "runtime_sec": avg_runtime,
            "runtime_std": std_runtime,
            "path": final_path,
            "policy": final_policy,
            "value_table": avg_value_table,
            "value_table_std": std_value_table,
            "visual_grid": vis_grid
        })

# Generate human-readable report
report_lines = []

for idx, res in enumerate(results_visual, 1):
    report_lines.append("="*60)
    report_lines.append(f"Experiment {idx}: Gamma = {res['gamma']}, Mode = {res['mode']} iteration")
    report_lines.append("-"*60)
    report_lines.append(f"Average Iterations: {res['iterations']:.4f} ± {res['iterations_std']:.4f}, "
                        f"Average Evalioation Iterations: {res['eval_iters']:.4f} ± {res['eval_iters_std']:.4f}, "
                        f"Average Runtime: {res['runtime_sec']:.6f} ± {res['runtime_std']:.6f} s")

    # Policy grid
    report_lines.append("Policy Grid:")
    policy_grid = [[" "]*W for _ in range(H)]
    for s_idx, action in enumerate(res['policy']):
        i,j = divmod(s_idx, W)
        policy_grid[i][j] = action
    for row in policy_grid:
        report_lines.append(" ".join(row))

    # Value table grid with std
    report_lines.append("Value Table Grid (avg ± std):")
    value_grid = [[" "]*W for _ in range(H)]
    for s_idx, (value, std) in enumerate(zip(res['value_table'], res['value_table_std'])):
        i,j = divmod(s_idx, W)
        value_grid[i][j] = f"{value:7.4f}±{std:7.4f}"
    for row in value_grid:
        report_lines.append(" ".join(row))

    # Path and visual grid
    report_lines.append("Path: " + " -> ".join(f"({i},{j})" for i,j in res['path']))
    report_lines.append("Visual Grid:")
    for row in res['visual_grid']:
        report_lines.append(" ".join(row))

    report_lines.append("="*60)
    report_lines.append("\n")

report_text = "\n".join(report_lines)

with open("stochastic_agent_results_readable.txt", "w") as f:
    f.write(report_text)

print("Saved human-readable averaged results with std to stochastic_agent_results_readable.txt")