"""
MDP visualization for Rock-Paper-Scissors.

States: 'Start', 'R', 'P', 'S' (last opponent move / initial)
Actions: 'R', 'P', 'S' (player's move)
Opponent model: Markov (configurable)
Reward: +1 win, 0 tie, -1 loss
"""

import networkx as nx
import matplotlib.pyplot as plt
from math import isclose

# -------------------------
# Configurable parameters
# -------------------------

# Opponent model choices: "markov" or "uniform"
OPPONENT_MODEL = "markov"  # change to "uniform" to assume opponent plays uniformly

# If markov: probability opponent repeats previous move
REPEAT_PROB = 0.6  # probability opponent repeats their previous move
# remaining probability split equally between the two other moves

# States and actions
STATES = ["Start", "R", "P", "S"]
ACTIONS = ["R", "P", "S"]

# Payoff matrix: player_action vs opponent_move -> reward
# R beats S, S beats P, P beats R
PAYOFF = {
    ("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1,
    ("P", "R"): 1, ("P", "P"): 0, ("P", "S"): -1,
    ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0,
}

# -------------------------
# Build opponent transition probabilities P(next_opp_move | current_state)
# -------------------------
def build_opponent_model(model="markov", repeat_prob=0.6):
    moves = ["R", "P", "S"]
    model_dict = {}
    if model == "uniform":
        for s in STATES:
            probs = {m: 1/3 for m in moves}
            model_dict[s] = probs
        return model_dict

    # Markov model keyed by current_state (which is last opponent move or "Start")
    # For "Start" we assume uniform prior (could be changed)
    for s in STATES:
        if s == "Start":
            model_dict[s] = {m: 1/3 for m in moves}
            continue
        # s is one of "R","P","S"
        probs = {}
        for m in moves:
            if m == s:
                probs[m] = repeat_prob
            else:
                probs[m] = (1 - repeat_prob) / 2
        # fix floating rounding
        total = sum(probs.values())
        if not isclose(total, 1.0):
            for k in probs:
                probs[k] /= total
        model_dict[s] = probs
    return model_dict

OPP_MODEL = build_opponent_model(OPPONENT_MODEL, REPEAT_PROB)

# -------------------------
# MDP construction:
# For each (state, action) -> distribution over next_state (which equals opponent move),
# reward = PAYOFF[(action, opp_move)]
# -------------------------
# transitions: dict[(state, action)] -> list of (next_state, prob, reward)
transitions = {}
for s in STATES:
    for a in ACTIONS:
        outcomes = []
        for opp_move, p in OPP_MODEL[s].items():
            reward = PAYOFF[(a, opp_move)]
            outcomes.append((opp_move, p, reward))
        transitions[(s, a)] = outcomes

# -------------------------
# Compute expected rewards per action at each state (policy = best-response)
# -------------------------
expected_rewards = {}  # (state) -> {action: expected_reward}
policy = {}  # best action(s) for each state (ties handled)
for s in STATES:
    ers = {}
    for a in ACTIONS:
        er = sum(p * r for (_, p, r) in transitions[(s, a)])
        ers[a] = er
    expected_rewards[s] = ers
    # choose best (max expected reward); allow multiple in tie
    max_val = max(ers.values())
    best_actions = [act for act, val in ers.items() if isclose(val, max_val, abs_tol=1e-9)]
    # if multiple actions tie, keep all
    policy[s] = best_actions

# -------------------------
# Build NetworkX MultiDiGraph for visualization
# Nodes: states
# Edges: for each (state,action) and each possible opp_move -> edge to next_state
# -------------------------
G = nx.MultiDiGraph()
for s in STATES:
    # annotate node with best action(s) and their expected values
    best = policy[s]
    ers = expected_rewards[s]
    best_str = ", ".join(best)
    # label will show best action(s) and ERs for each action
    er_strs = [f"{a}:{ers[a]:+.2f}" for a in ACTIONS]
    node_label = f"{s}\nbest: {best_str}\n" + " ".join(er_strs)
    G.add_node(s, label=node_label)

# Add edges with attributes
for (s, a), outcomes in transitions.items():
    for next_state, p, r in outcomes:
        # label contains action | prob | reward
        edge_label = f"{a} | p={p:.2f} | r={r:+d}"
        G.add_edge(s, next_state, action=a, prob=p, reward=r, label=edge_label)

# -------------------------
# Visualization
# -------------------------
def visualize_mdp(G, title="RPS MDP Visualization"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=123)  # layout can be changed

    # Node drawing with node labels from attribute
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_nodes(G, pos, node_size=2800, node_color="#f0f0f0", edgecolors="k")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Edge styling: highlight edges that correspond to a best action for the source state
    edge_colors = []
    edge_widths = []
    for u, v, k, data in G.edges(keys=True, data=True):
        src_best = policy[u]  # list of best actions for source state
        if data["action"] in src_best:
            edge_colors.append("#2ca02c")  # green for best-response edges
            edge_widths.append(3.0)
        else:
            edge_colors.append("#444444")
            edge_widths.append(1.2)

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",  # small arc to separate parallel edges
    )

    # Build edge labels mapping (u,v,key)->label
    edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Run visualization
if __name__ == "__main__":
    # Print model summary to console
    print("Opponent model (P(next_move | current_state)):")
    for s, probs in OPP_MODEL.items():
        print(f"  {s} -> {probs}")
    print("\nExpected rewards (state -> action: ER):")
    for s, ers in expected_rewards.items():
        print(f"  {s} -> " + ", ".join(f"{a}:{ers[a]:+.3f}" for a in ACTIONS))
    print("\nPolicy (best action(s) per state):")
    for s, acts in policy.items():
        print(f"  {s} -> {acts}")

    visualize_mdp(G, title=f"RPS MDP (opponent_model={OPPONENT_MODEL}, repeat_prob={REPEAT_PROB})")
