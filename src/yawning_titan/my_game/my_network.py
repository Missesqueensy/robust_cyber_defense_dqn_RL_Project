import networkx as nx
from yawning_titan.networks.network import Network
from yawning_titan.networks.node import Node

N = 10

def make_graph(name):
    if name == "chain":
        return nx.path_graph(N)
    if name == "ring":
        return nx.cycle_graph(N)
    if name == "star":
        return nx.star_graph(N - 1)          # hub = 0
    if name == "bridge":
        g = nx.Graph()
        g.add_nodes_from(range(N))
        g.add_edges_from(nx.complete_graph(range(0, 5)).edges)   
        g.add_edges_from(nx.complete_graph(range(6, 10)).edges)  
        g.add_edges_from([(4, 5), (5, 6)])                       #routeur
        return g
    raise ValueError(name)

def build_network(name):
    g = make_graph(name)
    net = Network(set_random_vulnerabilities=True,
              node_vulnerability_lower_bound=0.2,  
              node_vulnerability_upper_bound=0.8)                     
    nodes = [Node(name=str(i)) for i in range(N)]
    for i, node in enumerate(nodes):
       node._uuid = f"{i:02d}"                          
       net.add_node(node)
    nodes[0].entry_node = True                
    for u, v in g.edges:
        net.add_edge(nodes[u], nodes[v])
    net.set_node_positions()
    net.reset_random_vulnerabilities()

    return net



if __name__ == "__main__":
    for name in ("chain", "ring", "star", "bridge"):
        net = build_network(name)
        adj, pos = net.to_adj_matrix_and_positions()
        print([n.name for n in sorted(net.get_nodes())]) 
        print(name, adj.shape, "symétrique:", (adj == adj.T).all(), "somme:", int(adj.sum()))