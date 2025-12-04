ACTIONS = ["U", "R", "D", "L"]
ACTION_DELTA = {"U":(-1,0),"R":(0,1),"D":(1,0),"L":(0,-1)}

def build_states(grid):
    states = []
    pos2idx = {}
    is_obstacle = {}
    is_goal = {}
    start = None
    H = len(grid)
    W = len(grid[0])
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == "X":
                is_obstacle[(i,j)] = True
                continue
            index = len(states)
            states.append((i,j))
            pos2idx[(i,j)] = index
            is_obstacle[(i,j)] = False
            is_goal[(i,j)] = (c == "G")
            if c == "S":
                start = (i,j)
    return states, pos2idx, is_obstacle, is_goal, start, H, W


def mdp_successor(pos, action, grid, pos2idx, is_obstacle, is_goal, H, W):
    i, j = pos
    di, dj = ACTION_DELTA[action]
    ni, nj = i + di, j + dj

    if not (0 <= ni < H and 0 <= nj < W): return pos, -5
    if is_obstacle[(ni, nj)]: return pos, -5
    if grid[ni][nj] == "G": return (ni, nj), 10
    return (ni, nj), -1


def build_transition_model(grid):
    states, pos2idx, is_obstacle, is_goal, start, H, W = build_states(grid)
    n_states = len(states)
    P = {a: [None]*n_states for a in ACTIONS}

    for s_idx, pos in enumerate(states):

        if is_goal[pos]:
            for a in ACTIONS:
                P[a][s_idx] = [(s_idx, 1.0, 0.0)]
            continue

        for a in ACTIONS:
            transitions = []
            for a2 in ACTIONS:
                prob = 0.7 if a2 == a else 0.1
                next_pos, reward = mdp_successor(pos, a2, grid, pos2idx, is_obstacle, is_goal, H, W)
                next_idx = pos2idx[next_pos]
                transitions.append((next_idx, prob, reward))
            P[a][s_idx] = transitions

    return P, states, pos2idx, is_obstacle, is_goal, start, H, W


def extract_path(policy, states, pos2idx, start, grid):
    path = []
    pos = start
    visited = set()

    while True:
        if pos in visited: break
        visited.add(pos)

        path.append(pos)
        i, j = pos
        if grid[i][j] == "G":
            break

        idx = pos2idx[pos]
        action = policy[idx]
        di, dj = ACTION_DELTA[action]
        pos = (i + di, j + dj)

    return path


def value_iteration(P, n_states, gamma, theta, max_iters):
    V = [0.0] * n_states

    for iteration in range(max_iters):
        delta = 0.0
        new_V = [0.0] * n_states

        for s in range(n_states):
            q_values = []

            for action in ACTIONS:
                q = 0.0
                for next_s, prob, reward in P[action][s]:
                    q += prob * (reward + gamma * V[next_s])
                q_values.append(q)

            best_q = max(q_values)
            new_V[s] = best_q
            delta = max(delta, abs(best_q - V[s]))

        V = new_V
        if delta < theta:
            break

    policy = []
    for s in range(n_states):
        best_action = None
        best_value = float("-inf")

        for action in ACTIONS:
            q = 0.0
            for next_s, prob, reward in P[action][s]:
                q += prob * (reward + gamma * V[next_s])
            if q > best_value:
                best_value = q
                best_action = action

        policy.append(best_action)

    return V, policy, iteration


def policy_evaluation(P, policy, n_states, gamma, theta, max_iters):
    V = [0.0] * n_states

    for iteration in range(max_iters):
        delta = 0.0
        new_V = [0.0] * n_states

        for s in range(n_states):
            a = policy[s]
            new_V[s] = sum(prob * (reward + gamma * V[next_s])
                           for next_s, prob, reward in P[a][s])
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            break

    return V, iteration


def policy_iteration(P, n_states, gamma, theta, max_iters):
    policy = ["R"] * n_states

    for iteration in range(max_iters):
        V, eval_iters = policy_evaluation(P, policy, n_states, gamma, theta, max_iters)

        policy_stable = True

        for s in range(n_states):
            old_action = policy[s]

            best_action = None
            best_value = float("-inf")

            for action in ACTIONS:
                q = sum(prob * (reward + gamma * V[next_s])
                        for next_s, prob, reward in P[action][s])
                if q > best_value:
                    best_value = q
                    best_action = action

            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, policy, iteration, eval_iters


def run_agent(grid, gamma=0.6, theta=1e-4, max_iters=10000, mode="value"):
    P, states, pos2idx, is_obstacle, is_goal, start, H, W = build_transition_model(grid)
    n_states = len(states)

    if mode == "value":
        V, policy, iterations = value_iteration(P, n_states, gamma, theta, max_iters)
        eval_iters = 0

    elif mode == "policy":
        V, policy, iterations, eval_iters = policy_iteration(P, n_states, gamma, theta, max_iters)

    else:
        raise ValueError("mode must be 'value' or 'policy'")

    return {"value_table": V, "policy": policy, "iterations": iterations, "eval_iters": eval_iters}