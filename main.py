import numpy as np
import networkx as nx
import scipy.sparse as sp
from ppr import *
import dynamic_ppr

if __name__ == '__main__':
    alpha = 0.1
    G0 = nx.Graph()
    G0.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5), (1, 5), (2, 5), (1, 6)])
    nodes = list(G0.nodes)
    delta_G0 = [[4, 6, 'i'], [1, 4, 'i'], [3, 6, 'i']]
    delta_G1 = [[4, 6, 'i'], [1, 4, 'i'], [2, 6, 'i'], [3, 7, 'i'], [1, 8, 'i'], [5, 6, 'i'], [6, 7, 'i']]
    # A = nx.adjacency_matrix(G0).tocsr().astype(np.float32)
    # ppr = calc_ppr_exact(A, alpha=alpha)
    # print(np.around(ppr, 2))

    ps, rs = calc_ppr(G0, alpha=0.1, epsilon=1e-8, nodes=list(G0.nodes))
    vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]
    print(np.around(vals, 3))
    # print(np.linalg.norm(ppr - np.asarray(vals), axis=1))

    ps_new, rs_new = DaynamicPPE(G0, delta_G0, S=list(G0.nodes), epsilon=1e-8, alpha=0.2, p=ps, r=rs)
    js_new = [list(ps_new[i].keys()) for i in ps_new.keys()]
    vals_new = [[ps_new[i][j] for j in sorted(ps_new[i])] for i in ps_new.keys()]
    print(np.round(vals_new,3))
    ps, rs = calc_ppr(G0, alpha=0.1, epsilon=1e-8, nodes=list(G0.nodes))
    vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]
    print(np.linalg.norm(vals - np.asarray(vals_new), axis=1))

