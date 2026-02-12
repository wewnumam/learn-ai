import networkx as nx
import matplotlib.pyplot as plt

states = [0,1,2]
actions = [0,1]  # order quantity
demand = 1       # deterministic for simplicity

G = nx.MultiDiGraph()
for s in states:
    G.add_node(str(s))

for s in states:
    for a in actions:
        ns = max(0, min(2, s + a - demand))
        reward = s + a - ns - 0.2*a  # sales − order cost
        G.add_edge(str(s),str(ns),label=f"order {a} | r={reward:.1f}")

pos = {"0":(0,0),"1":(1,0),"2":(2,0)}
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(
    G,pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)}
)
plt.title("Inventory Control MDP")
plt.axis("off")
plt.show()
