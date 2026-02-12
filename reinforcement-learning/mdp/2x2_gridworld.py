import networkx as nx
import matplotlib.pyplot as plt

states = [(0,0),(0,1),(1,0),(1,1)]
actions = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
goal = (1,1)

G = nx.MultiDiGraph()

for s in states:
    G.add_node(s)

for s in states:
    for a,(dx,dy) in actions.items():
        ns = (s[0]+dx, s[1]+dy)
        if ns not in states:
            ns = s
        r = 1 if ns==goal else 0
        G.add_edge(s,ns,label=f"{a} | r={r}")

pos = {s:(s[1],-s[0]) for s in states}
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(
    G,pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)},
    font_size=7
)
plt.title("2x2 Gridworld MDP")
plt.axis("off")
plt.show()
