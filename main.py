import numpy as np
import networkx as nx
import scipy.sparse as sp
from ppr import *
import dynamic_ppr

if __name__ == '__main__':
    alpha = 0.1
    G0 = nx.Graph()
    G0.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (1, 5), (2, 5)])
    nodes = list(G0.nodes)
    # delta_G0 = [[1, 3, 'i'], [3, 1, 'i']]
    delta_G0 = [[1, 5, 'i'], [2, 5, 'i']]


    # A = nx.adjacency_matrix(G0).tocsr().astype(np.float32)
    # ppr = calc_ppr_exact(A, alpha=alpha)
    # print(np.around(ppr, 3))

    # ps, rs = calc_ppr(G1, alpha=0.1, epsilon=1e-8, nodes=list(G0.nodes))
    # vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]
    # print(np.around(vals, 3))
    # print(np.linalg.norm(ppr - np.asarray(vals), axis=1))

    ps, rs =DaynamicPPE(G1, S=[0, 1, 2, 3], epsilon=1e-3, alpha=0.1)
    vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]
    print(np.around(vals, 3))

    ps_new, rs_new = DaynamicPPE(G0, delta_G0=delta_G0, S=[0, 1, 2, 3], epsilon=1e-3, alpha=0.1)
    js_new = [list(ps_new[i].keys()) for i in ps_new.keys()]
    vals_new = [[ps_new[i][j] for j in sorted(ps_new[i])] for i in ps_new.keys()]
    print(np.round(vals_new,3))

    print(np.linalg.norm(vals - np.asarray(vals_new), axis=1))

    # ps, rs = calc_ppr(G0, alpha=0.1, epsilon=1e-8, nodes=list(G0.nodes))
    # vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]
    # print(np.linalg.norm(vals - np.asarray(vals_new), axis=1))


