import networkx as nx
import matplotlib.pyplot as plt

actions = {"A": 0.2, "B": 0.5, "C": 0.8}  # Bernoulli means

G = nx.MultiDiGraph()
G.add_node("s", label="s")

for a, p in actions.items():
    G.add_edge("s", "s", label=f"{a} | E[r]={p:.2f}")

pos = {"s": (0, 0)}
nx.draw(G, pos, with_labels=True, node_size=2500)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)}
)
plt.title("Multi-Armed Bandit MDP")
plt.axis("off")
plt.show()
