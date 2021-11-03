import numpy as np
import numba
import networkx as nx
import scipy.sparse as sp


def _calc_ppr_node(inode, G, alpha, epsilon, p_pre=None, r_pre=None):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    if p_pre is None:
        p = {inode: f32_0}
    else:
        p = p_pre
    if r_pre is None:
        r = {}
        r[inode] = alpha
    else:
        r = r_pre
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in list(G.adj[unode]):
            _val = (1 - alpha) * res / G.degree[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * G.degree[vnode]:
                if vnode not in q:
                    q.append(vnode)
    return p, r


def calc_ppr(G, alpha, epsilon, nodes, p_pre=None, r_pre=None):
    ps = {}
    rs = {}
    for i, node in enumerate(nodes):
        if p_pre is not None and node in p_pre:
            p, r = _calc_ppr_node(node, G, alpha, epsilon, p_pre[node], r_pre[node])
        else:
            p, r = _calc_ppr_node(node, G, alpha, epsilon)
        ps[node] = p
        rs[node] = r
        # js.append(list(p.keys()))
        # vals.append(list(p.values()))
    return ps, rs


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def DynamicSNE(G, delta_G, epsilon, alpha, S=[1, 2, 3], p_pre=None, r_pre=None):
    for changes in delta_G:
        u, v, op = changes[0], changes[1], changes[2]
        flag = v in list(G.nodes)
        for s in S:
            if op == 'i':
                G.add_edge(u, v)
                delta_p = p_pre[s][u] / (G.degree[u] - 1)
            if op == 'd':
                G.remove_edge(u, v)
                delta_p = -p_pre[s][u] / (G.degree[u] + 1)
            p_pre[s][u] += delta_p
            r_pre[s][u] = r_pre[s][u] - delta_p / alpha
        if flag:
            r_pre[u][v] = r_pre[u][v] + delta_p / alpha - delta_p
        p_pre, r_pre = calc_ppr(G, alpha, epsilon, list(G.nodes), p_pre, r_pre)
    return p_pre, r_pre


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    # A = adj_matrix + sp.eye(nnodes)
    A = adj_matrix
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1. / D_vec
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return  D_invsqrt_corr @ A


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]

    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())
