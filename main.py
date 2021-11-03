import numpy as np
import networkx as nx
import scipy.sparse as sp
from ppr import calc_ppr_exact
from ppr import calc_ppr

if __name__ == '__main__':
    alpha = 0.1
    G0 = nx.Graph()
    G0.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5), (1, 5), (2, 5), (1, 6)])
    nodes = list(G0.nodes)
    delta_G0 = [[4, 6, 'i'], [1, 4, 'i']]
    delta_G1 = [[4, 6, 'i'], [1, 4, 'i'], [2, 6, 'i'], [3, 7, 'i'], [1, 8, 'i'], [5, 6, 'i'], [6, 7, 'i']]
    A = nx.adjacency_matrix(G0).tocsr().astype(np.float32)
    ppr = calc_ppr_exact(A, alpha=alpha)
    print(np.around(ppr, 6))
    print(np.sum(ppr, axis=1))

    ps, rs = calc_ppr(G0, alpha=0.1, epsilon=1e-8, nodes=nodes)
    js = [list(ps[i].keys()) for i in ps.keys()]
    vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]
    print(np.around(vals, 6))
    print(np.sum(vals, axis=1))
    print(np.linalg.norm(ppr-np.asarray(vals),axis=1))
    exit()

    # a = construct_sparse(js, vals, (len(nodes)+1, len(nodes)+1))
    ps_new, rs_new = ppr.DynamicSNE(G0, delta_G=delta_G0, epsilon=0.001, alpha=0.9, S=list(G0.nodes), p_pre=ps,
                                    r_pre=rs)
    js_new = [list(ps_new[i].keys()) for i in ps_new.keys()]
    vals_new = [list(ps_new[i].values()) for i in ps_new.keys()]
