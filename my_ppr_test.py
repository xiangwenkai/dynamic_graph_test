import numpy as np
import torch
import my_ppr_topk
import time
import scipy.sparse as sp
import numba


# 批量更新
def test_batch():
    I = [0, 1, 0, 1, 2, 3, 0, 4]
    J = [1, 2, 3, 0, 1, 0, 4, 0]
    V = [1, 1, 1, 1, 1, 1, 1, 1]
    mat1 = sp.coo_matrix((V, (I, J)), shape=(10, 10))
    mat1 = mat1.tocsr()

    I2 = [0, 1, 0, 1, 2, 3, 0, 4, 3, 4, 1, 4]
    J2 = [1, 2, 3, 0, 1, 0, 4, 0, 4, 3, 4, 1]
    V2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    mat2 = sp.coo_matrix((V2, (I2, J2)), shape=(10, 10))
    mat2 = mat2.tocsr()

    # forwardpush初始化
    # n_adj = len(mat1.indptr)
    start = time.time()
    adj_degree = np.sum(mat1 > 0, axis=1).A1
    ps, rs = my_ppr_topk.DynamicPPE_init(mat1.indptr, mat1.indices, adj_degree=adj_degree,
                                         S=numba.typed.List([0, 1, 2, 3, 4]), epsilon=1e-5, alpha=0.2)
    print(f"Init time: {time.time() - start}")

    # forwardpush更新
    adj_degree2 = np.sum(mat2 > 0, axis=1).A1
    delta_edges = [(3,4),(4,3),(1,4),(4,1)]
    start = time.time()
    ps_new, rs_new = my_ppr_topk.DynamicPPE_update(mat2.indptr, mat2.indices, adj_degree=adj_degree2,
                                                       delta_edge=delta_edges, S=numba.typed.List([0,1,2,3,4]),
                                                       epsilon=1e-5, alpha=0.2, p_pre=ps, r_pre=rs)
    print(f"t1: {time.time() - start}")


    p1 = [[ps_new[i][j] for j in sorted(ps_new[i])] for i in ps_new.keys()]

    # True ppr
    ps_G1, rs_G1 = my_ppr_topk.calc_ppr(mat2, epsilon=1e-5, alpha=0.2, nodes=[0, 1, 2, 3, 4])
    p2 = [[ps_G1[i][j] for j in sorted(ps_G1[i])] for i in ps_G1.keys()]

    print(f"Difference between update ppr and true ppr {np.linalg.norm(p1 - np.asarray(p2), axis=1)}")


# 逐步更新
def test_step():
    I = [0, 1, 0, 1, 2, 3, 0, 4]
    J = [1, 2, 3, 0, 1, 0, 4, 0]
    V = [1, 1, 1, 1, 1, 1, 1, 1]
    mat1 = sp.coo_matrix((V, (I, J)), shape=(10, 10))
    mat1 = mat1.tocsr()

    mat2 = mat1.copy()

    # forwardpush初始化
    start = time.time()
    adj_degree = np.sum(mat1 > 0, axis=1).A1
    ps, rs = my_ppr_topk.DynamicPPE_init(mat1.indptr, mat1.indices, adj_degree=adj_degree,
                                         S=numba.typed.List([0, 1, 2, 3, 4]), epsilon=1e-5, alpha=0.2)
    print(f"Init time: {time.time() - start}")

    # 尝试更新
    delta_edges = [[(3, 4), (4, 3)], [(1, 4), (4, 1)]]
    for delta_edge in delta_edges:
        for x in delta_edge:
            mat2[x[0], x[1]] = 1
        adj_degree2 = np.sum(mat2 > 0, axis=1).A1
        start = time.time()
        ps_new, rs_new = my_ppr_topk.DynamicPPE_update(mat2.indptr, mat2.indices, adj_degree=adj_degree2,
                                                       delta_edge=delta_edge, S=numba.typed.List([0, 1, 2, 3, 4]),
                                                       epsilon=1e-5, alpha=0.2, p_pre=ps, r_pre=rs)
        print(f"Update time: {time.time() - start}")


    p1 = [[ps_new[i][j] for j in sorted(ps_new[i])] for i in ps_new.keys()]

    # True ppr
    ps_G1, rs_G1 = my_ppr_topk.calc_ppr(mat2, epsilon=1e-5, alpha=0.2, nodes=[0, 1, 2, 3, 4])

    p2 = [[ps_G1[i][j] for j in sorted(ps_G1[i])] for i in ps_G1.keys()]
    print(f"Difference between update ppr and true ppr {np.linalg.norm(p1 - np.asarray(p2), axis=1)}")


test_batch()
test_step()





