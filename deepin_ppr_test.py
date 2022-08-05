import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import scipy.sparse as sp
import yaml
import ast
import time
import logging
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import my_ppr


data_files = ['DPPIN-Babu', 'DPPIN-Krogan(LCMS)', 'DPPIN-Uetz', 'DPPIN-Yu',
 'DPPIN-Ho', 'DPPIN-Breitkreutz', 'DPPIN-Krogan(MALDI)', 'DPPIN-Ito',
 'DPPIN-Tarassov', 'DPPIN-Gavin', 'DPPIN-Hazbun', 'DPPIN-Lambert']

path = f'data/DPPIN-main/{data_files[0]}'

# 动态数据
dynamic = pd.read_csv(path+'/Dynamic_PPIN.txt', header=None)
dynamic.columns = ['node_u', 'node_v', 'timestamp', 'weight']
# dynamic_2t = dynamic[dynamic['timestamp'].isin([0,1])]

# 节点特征
node_features = pd.read_csv(path+'/Node_Features.txt', header=None)
node_features.columns = ['node_id', 'node_name']+[f'value_at_t_{i}' for i in range(1, 37)]
all_nodes = list(node_features['node_id'])


# 静态数据
static = pd.read_csv(path+'/Static_PPIN.txt', sep='\t', header=None)
static.columns = ['node_u', 'node_v', 'weight']
for i in ['u', 'v']:
    temp = node_features[['node_name', 'node_id']]
    temp.columns = [f'node_{i}', f'node_id{i}']
    static = pd.merge(static, temp, on=f'node_{i}', how='left')
    static = static.dropna()
    static[f'node_id{i}'] = static[f'node_id{i}'].astype(int)
n_max = max(static['node_idu'].max(), static['node_idv'].max())


# Y标签
y_label = pd.read_excel('data/DPPIN-main/Node_Labels.xlsx')
y_label = y_label.drop_duplicates()
dict(y_label['Label'].value_counts())
# y_map = {'ORF-V': 1, 'ORF-U': 2, 'ORF-DU': 3, 'LTR': 4, 'tRNA gene': 5, 'snoRNA gene':6,
#          'ORF-M':7, 'pseudogene':8, 'centromere':9, 'rRNA gene':10, 'BRF':11,
#          'snRNA gene':12, 'ARS':13, 'ORF-DE':14, 'ORF':15, 'IER':16, 'retrotransposon':17,
#          'telomerase RNA gene':18, 'ncRNA gene':19}
y_label = pd.merge(node_features[['node_id', 'node_name']], y_label,
                   left_on='node_name', right_on='Node', how='left')
y_label['Label'] = y_label['Label'].fillna('unknown')
y_count = dict(y_label['Label'].value_counts())
y_map = {}
for i, key in enumerate(y_count.keys()):
    y_map[key] = i
y_label['Label'] = y_label['Label'].map(y_map)
labels = y_label['Label'].values
# y_label['Label'] = y_label['Label'].fillna(-1).astype(int)

# 每个时间节点的数据
timestamp = 0
num_features = 1
data_stamp = dynamic[dynamic['timestamp'] == timestamp]
ulist = data_stamp['node_u'].tolist()
vlist = data_stamp['node_v'].tolist()
ulist_bi = ulist[:] + vlist[:]
vlist_bi = vlist[:] + ulist[:]

val = [1 for i in range(len(ulist))]
mat = sp.coo_matrix((val, (ulist, vlist)), shape=(n_max+1, n_max+1)).tocsr()

all_idx = list(set(ulist))

# forwardpush初始化
delta_edge = [(i,j,'i') for i, j in zip(ulist_bi,vlist_bi)]
t = time.time()
ps, rs = my_ppr.DynamicPPE_init(adj_matrix=mat,delta_edge=delta_edge,S=all_idx,
                                epsilon=1e-3,alpha=0.1)
print(f"Init time: {time.time() - t:.2f}s")

train_size = int(0.7*len(all_idx))
train_idx = np.array(all_idx[:train_size])
val_idx = np.array(all_idx[train_size:])
ppr_matrix = my_ppr.ppr_convex(adj_matrix=mat, ps=ps, idx=train_idx, normalization='row')


# 计算每个timestamp相对于上一次的新增边和删除边
def cal_delta_edge(edge_set1, edge_set2):
    new_edge = edge_set2 - edge_set1
    delete_edge = edge_set1 - edge_set2
    return [(i, j, 'i') for i, j in new_edge] + [(i, j, 'd') for i, j in delete_edge]


# =========dynamic time test==================
T = dynamic['timestamp'].max()
edges = []
for timestamp in range(T):
    edge_set = []
    data_stamp = dynamic[dynamic['timestamp'] == timestamp]
    ulist = data_stamp['node_u'].tolist()
    vlist = data_stamp['node_v'].tolist()
    for i, j in zip(ulist, vlist):
        if i <= j:
            edge_set.append((i, j))
        else:
            edge_set.append((j, i))
    edges.append(set(edge_set))

delta_edges = []
for i in range(T-1):
    delta_edges.append(cal_delta_edge(edges[i], edges[i+1]))



