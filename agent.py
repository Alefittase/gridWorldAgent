GRID_MAP = [
    ["S","_","_","X","_"],
    ["_","X","_","_","_"],
    ["_","_","X","_","_"],
    ["X","_","_","_","G"],
    ["_","_","X","_","_"],
]

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
            c = grid [i][j]
            if c == "X":
                is_obstacle[(i,j)] = True
                continue
            index = len(states)
            states.append((i,j))
            pos2idx[(i,j)] = index
            is_obstacle[(i,j)] = false
            if c == "G":
                is_goal[(i,j)] = True
            if c == "S":
                start = (i,j)
    return states, pos2idx, is_obstacle, is_goal, start, H, W

def mdp_successor(pos, action, grid, pos2idx, is_obstacle, is_goal, H, W):
    i, j = pos
    di, dj = ACTION_DELTA[action]
    ni, nj = i+di, j+dj

    if not (0 <= ni < H and 0 <= nj < W):
        return pos, -5
    if grid[ni][nj] == "X":
        return pos, -5
    if grid[ni][nj] == "G":
        return (ni, nj), 10
    return (ni, nj), -1

def build_transition_model(grid):
    states, pos2idx, is_obstacle, is_goal, start, H, W = build_states(grid)
    n_states = len(states)
    P = {action: [None] * n_states for action in ACTIONS}
    for s_idx, pos in enumerate(states):
        if is_goal[pos]:
            for action in ACTIONS:
                P[action][s_idx] = (s_idx, 1.0, 0.0)
            continue

        for action in ACTIONS:
            next_pos, reward = mdp_successor(pos, action, grid, pos2idx, is_obstacle, is_goal, H, W )
            next_idx = pos2idx[next_pos]
            P[action][s_idx] = (next_idx, 1.0, reward)

    return P, states, pos2idx, is_obstacle, is_goal, start, H, W

