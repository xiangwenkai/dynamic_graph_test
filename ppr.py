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


def ForwardPush(p, r, s, G, epsilon,alpha,beta=0.5):
    enque = [s]
    while len(enque)>0:
        snode = enque.pop()
        res = r[snode]
        p[snode] += alpha*res
        r[snode] = (1-alpha)*r[snode]*beta
        for vnode in list(G.adj[snode]):
            r[vnode] += (1-alpha)*r[snode]*(1-beta)/G.degree(snode)
            res_vnode = r[vnode]
            if abs(res_vnode) > epsilon * G.degree(vnode):
                if vnode not in enque:
                    enque.append(vnode)
    return p, r


def DynamicSNE(G, delta_G, epsilon, alpha, S=[1,2,3],p_pre=None, r_pre=None):
    for change in delta_G:
        u, v, op = change[0],change[1],change[2]
        flag = v in list(G.nodes)
        if op == 'i':
            G.add_edge(u, v)
        if op == 'd':
            G.remove_edge(u, v)
        for s in S:
            if op == 'i':
                delta_p = p_pre[s][u] / (G.degree[u] - 1)
            if op == 'd':
                delta_p = -p_pre[s][u] / (G.degree[u] + 1)
            p_pre[s][u] += delta_p
            r_pre[s][u] = r_pre[s][u] - delta_p / alpha
            if flag:
                r_pre[s][v] = r_pre[s][v] + delta_p / alpha - delta_p
            # p_pre[s], r_pre[s] = _calc_ppr_node(s, G, alpha=0.1, epsilon=epsilon, p_pre=p_pre[s], r_pre=r_pre[s])
            p_pre[s], r_pre[s] = ForwardPush(p=p_pre[s], r=r_pre[s], s=s, G=G, epsilon=epsilon, alpha=alpha)
    return p_pre, r_pre


def DaynamicPPE(G, delta_G0, S, epsilon, alpha, p, r):
    # p = {}
    # r = {}
    # for s in S:
    #     nnodes = len(G.nodes)
    #     p[s] = dict(zip(G.nodes, [0]*nnodes))
    #     r[s] = dict(zip(G.nodes, [0]*nnodes))
    #     r[s][s] = 1
    #     p[s],r[s] = ForwardPush(p[s], r[s], s, G, epsilon=epsilon, alpha=alpha)
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


