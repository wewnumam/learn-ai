import networkx as nx
import matplotlib.pyplot as plt

states = [0,1,2]
p = 0.5  # win prob

G = nx.MultiDiGraph()
for s in states:
    G.add_node(str(s))

# Only state 1 has action
G.add_edge("1","2",label="bet1 | p=0.5 | r=1")
G.add_edge("1","0",label="bet1 | p=0.5 | r=0")

pos = {"0":(0,0),"1":(1,0),"2":(2,0)}
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(
    G,pos,
    edge_labels={(u,v,k):d["label"] for u,v,k,d in G.edges(keys=True,data=True)}
)
plt.title("Gambler's Problem MDP")
plt.axis("off")
plt.show()
