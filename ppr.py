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
        r[inode] = alpha
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


def ForwardPush(p, r, s, G, epsilon, alpha):
    enque = s
    while len(enque) > 0:
        snode = enque.pop()
        p[snode] += alpha*r[snode]
        res = (1 - alpha) * r[snode] / G.degree(snode)
        for vnode in list(G.adj[snode]):
            r[vnode] += res
            res_vnode = r[vnode]
            if abs(res_vnode) >= epsilon * G.degree(vnode):
                if vnode not in enque:
                    enque.append(vnode)
        r[snode] = 0.
    return p, r

def add_point(G, p, r, unode, epsilon, alpha):
    p[unode] = dict(zip(G.nodes, [0] * len(G.nodes)))
    r[unode] = dict(zip(G.nodes, [0] * len(G.nodes)))
    for i in list(G.nodes):
        p[i][unode] = 0
        r[i][unode] = 0
    r[unode][unode] = 1
    p[unode], r[unode] = ForwardPush(p=p[unode], r=r[unode], s=[unode], G=G, epsilon=epsilon, alpha=alpha)


def DynamicSNE(G, delta_G, epsilon, alpha, S, p_pre=None, r_pre=None):
    for change in delta_G:
        u, v, op = change[0], change[1], change[2]
        if op == 'i':
            G.add_edge(u, v)
        if op == 'd':
            G.remove_edge(u, v)
        if u not in r_pre[S[0]]:
            add_point(G, p_pre, r_pre, v, epsilon, alpha)
        if v not in r_pre[S[0]]:
            add_point(G, p_pre, r_pre, v, epsilon, alpha)
    for s in S:
        for change in delta_G:
            u, v, op = change[0], change[1], change[2]
            if op == 'i':
                if G.degree[u]==1:
                    continue
                delta_pu = p_pre[s][u] / (G.degree[u] - 1)
                if G.degree[v] == 1:
                    continue
                delta_pv = p_pre[s][v] / (G.degree[v] - 1)
            if op == 'd':
                delta_pu = -p_pre[s][u] / (G.degree[u] + 1)
                delta_pv = -p_pre[s][v] / (G.degree[v] + 1)
            p_pre[s][u] += delta_pu
            r_pre[s][u] -= delta_pu / alpha
            r_pre[s][v] += delta_pu / alpha - delta_pu

            p_pre[s][v] += delta_pv
            r_pre[s][v] -= delta_pv / alpha
            r_pre[s][u] += delta_pv / alpha - delta_pv

        p_pre[s], r_pre[s] = ForwardPush(p=p_pre[s], r=r_pre[s], s=[u], G=G, epsilon=epsilon, alpha=alpha)
    return p_pre, r_pre


def DaynamicPPE(G, S, epsilon, alpha, delta_G0=None):
    p = {}
    r = {}
    for s in G.nodes:
        nnodes = len(G.nodes)
        p[s] = dict(zip(G.nodes, [0]*nnodes))
        r[s] = dict(zip(G.nodes, [0]*nnodes))
        r[s][s] = 1.
        p[s], r[s] = ForwardPush(p[s], r[s], [s], G, epsilon=epsilon, alpha=0.1)
    if delta_G0 is not None:
        p, r = DynamicSNE(G, delta_G0, epsilon, alpha, S=S, p_pre=p, r_pre=r)
    return p, r


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    # nnodes = adj_matrix.shape[0]
    A = adj_matrix
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1. / D_vec
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


