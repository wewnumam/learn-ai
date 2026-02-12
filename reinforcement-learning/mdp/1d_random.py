import networkx as nx
import matplotlib.pyplot as plt

states = [0,1,2]
actions = ["left","right"]

transitions = {
    (1,"left"):  [(0,1,0)],
    (1,"right"): [(2,1,1)],
    (0,"left"):  [(0,1,0)],
    (2,"right"): [(2,1,0)],
}

G = nx.MultiDiGraph()
for s in states:
    G.add_node(str(s))

for (s,a),outs in transitions.items():
    for s2,p,r in outs:
        G.add_edge(str(s),str(s2),label=f"{a} | r={r}")

pos = {"0":(0,0),"1":(1,0),"2":(2,0)}
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(
    G,pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)}
)
plt.title("1D Random Walk MDP")
plt.axis("off")
plt.show()
