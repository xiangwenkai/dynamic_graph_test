import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from ppr import *
import time

data = pd.read_csv('./wikipedia.csv', usecols=['user_id', 'item_id', 'timestamp', 'state_label'])
data = data[['user_id', 'item_id']].drop_duplicates()
G0_edges = [(i, j) for i, j in zip(data['user_id'].iloc[:10000], data['item_id'].iloc[:10000])]
G0 = nx.Graph()
G0.add_edges_from(G0_edges)
G1_edges = [(i, j, 'i') for i, j in zip(data['user_id'].iloc[10000:11000], data['item_id'].iloc[10000:11000])]
nodes = list(G0.nodes)

t0 = time.time()
ps, rs =DaynamicPPE(G0, S=[0, 1, 2, 3, 4, 5], epsilon=1e-3, alpha=0.1)
vals = [[ps[i][j] for j in sorted(ps[i])] for i in ps.keys()]

t0 = time.time()
ps_new, rs_new = DaynamicPPE(G0, delta_G0=G1_edges, S=nodes, epsilon=1e-3, alpha=0.1)

# js_new = [list(ps_new[i].keys()) for i in ps_new.keys()]
# vals_new = [[ps_new[i][j] for j in sorted(ps_new[i])] for i in ps_new.keys()]
# print(np.round(vals_new,3))


