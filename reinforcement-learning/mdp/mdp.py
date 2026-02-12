import networkx as nx
import matplotlib.pyplot as plt

# =========================
# 1. Define the MDP
# =========================

states = ["S0", "S1", "S2"]

actions = ["a", "b"]

# Transition model:
# (state, action) -> [(next_state, probability, reward)]
transitions = {
    ("S0", "a"): [("S1", 0.8, 5), ("S2", 0.2, 0)],
    ("S0", "b"): [("S2", 1.0, 1)],
    ("S1", "a"): [("S0", 1.0, -1)],
    ("S2", "a"): [("S2", 1.0, 0)],  # terminal self-loop
}

# Optional: deterministic policy π(s)
policy = {
    "S0": "a",
    "S1": "a",
    "S2": None,  # terminal
}

# =========================
# 2. Build the MDP Graph
# =========================

G = nx.MultiDiGraph()

# Add states
for s in states:
    G.add_node(s)

# Add transitions
for (s, a), outcomes in transitions.items():
    for s_next, p, r in outcomes:
        G.add_edge(
            s,
            s_next,
            action=a,
            prob=p,
            reward=r,
            label=f"{a} | p={p}, r={r}"
        )

# =========================
# 3. Visualization Settings
# =========================

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=2500,
    node_color="lightgray",
    edgecolors="black"
)

# Draw node labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=12,
    font_weight="bold"
)

# =========================
# 4. Draw Edges (Policy Highlighted)
# =========================

edge_colors = []
edge_widths = []

for u, v, k, data in G.edges(keys=True, data=True):
    if policy.get(u) == data["action"]:
        edge_colors.append("tab:red")     # policy edge
        edge_widths.append(3.0)
    else:
        edge_colors.append("black")
        edge_widths.append(1.2)

nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    width=edge_widths,
    arrows=True,
    arrowstyle="->",
    arrowsize=20
)

# =========================
# 5. Edge Labels
# =========================

edge_labels = {
    (u, v, k): data["label"]
    for u, v, k, data in G.edges(keys=True, data=True)
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=9
)

# =========================
# 6. Final Touches
# =========================

plt.title("MDP Visualization (Policy Highlighted)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
