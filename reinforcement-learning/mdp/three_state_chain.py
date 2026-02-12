import networkx as nx
import matplotlib.pyplot as plt

states = ["S0","S1","S2"]
actions = ["forward","backward"]

transitions = {
    ("S0","forward"): [("S1",1,0)],
    ("S1","forward"): [("S2",1,1)],
    ("S1","backward"):[("S0",1,0)],
    ("S2","forward"): [("S2",1,0)],
}

G = nx.MultiDiGraph()
for s in states:
    G.add_node(s)

for (s,a),outs in transitions.items():
    for s2,p,r in outs:
        G.add_edge(s,s2,label=f"{a} | r={r}")

pos = {"S0":(0,0),"S1":(1,0),"S2":(2,0)}
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(
    G,pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)}
)
plt.title("3-State Chain MDP")
plt.axis("off")
plt.show()
