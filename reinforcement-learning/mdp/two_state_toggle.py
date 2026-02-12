import networkx as nx
import matplotlib.pyplot as plt

states = ["On", "Off"]
actions = ["toggle", "stay"]

transitions = {
    ("On","toggle"): [("Off",1,0)],
    ("On","stay"):   [("On",1,1)],
    ("Off","toggle"):[("On",1,1)],
    ("Off","stay"):  [("Off",1,0)],
}

G = nx.MultiDiGraph()
for s in states:
    G.add_node(s)

for (s,a),outs in transitions.items():
    for s2,p,r in outs:
        G.add_edge(s,s2,label=f"{a} | r={r}")

pos = {"On":(0,1),"Off":(0,-1)}
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(
    G,pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)}
)
plt.title("Two-State Toggle MDP")
plt.axis("off")
plt.show()
