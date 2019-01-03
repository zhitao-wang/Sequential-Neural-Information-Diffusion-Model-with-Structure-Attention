#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import Counter
import os
import numpy as np
import scipy.sparse as sp

def _read_nodes(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                line = line.split(':')[0]
            data.extend(line.replace('\n', '').split(','))
        return data

def read_graph(filename, node_to_id):
    N = len(node_to_id)
    A = np.zeros((N,N), dtype=np.float32)
    with open(filename, 'r') as f:
        for line in f:
            edge = line.strip().split()
            if edge[0] in node_to_id and edge[1] in node_to_id:
                source_id = node_to_id[edge[0]]
                target_id = node_to_id[edge[1]]
                # if len(edge) >= 3:
                #     A[source_id,target_id] = float(edge[2])
                # else:
                A[source_id,target_id] = 1.0
    return A

# def read_graph_sparse(filename, node_to_id):
#     N = len(node_to_id)
#     data = []
#     row = []
#     col = []
#     with open(filename, 'r') as f:
#         for line in f:
#             edge = line.strip().split()
#             if edge[0] in node_to_id and edge[1] in node_to_id:
#                 row.append(node_to_id[edge[0]])
#                 col.append(node_to_id[edge[1]])
#                 data.append(1.0)
#     row = np.array(row)
#     col = np.array(col)
#     data = np.array(data)
#     A = sp.csr_matrix((data, (row, col)), shape=(N, N))
#     return A

# def sparse_neighbors_generate(x_batch): # x_batch is a (batch_size, seq_length) matrix
#     return A[x_batch]

def _build_vocab(filename):
    data = _read_nodes(filename)

    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    nodes, _ = list(zip(*count_pairs))
    nodes = list(nodes)
    nodes.insert(0,'-1') # index for mask
    node_to_id = dict(zip(nodes, range(len(nodes))))

    return nodes, node_to_id

def _file_to_node_ids(filename, node_to_id):
    data = []
    len_list = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                line = line.split(':')[0]
            seq = line.strip().split(',')
            ix_seq = [node_to_id[x] for x in seq if x in node_to_id]
            if len(ix_seq)>=2:
                data.append(ix_seq)
                len_list.append(len(ix_seq)-1)
    size = len(data)
    total_num = np.sum(len_list)
    return (data, len_list, size, total_num)

def to_nodes(seq, nodes):
    return list(map(lambda x: nodes[x], seq))

def read_raw_data(data_path=None):
    train_path = data_path + '-train'
    valid_path = data_path +  '-val'
    test_path = data_path +  '-test'

    nodes, node_to_id = _build_vocab(train_path)
    train_data = _file_to_node_ids(train_path, node_to_id)
    valid_data = _file_to_node_ids(valid_path, node_to_id)
    test_data = _file_to_node_ids(test_path, node_to_id)
    print('Node Num:' + str(len(nodes)-1)) # Exclude the masking index 0
    print('train size:' + str(len(train_data[0])) + '; ' + 'test size:' + str(len(test_data[0])))
    return train_data, valid_data,  test_data,  nodes, node_to_id

def batch_generator(train_data, batch_size=50):
    x = []
    y = []
    xs = []
    ys = []
    ss = []
    train_seq = train_data[0]
    train_steps = train_data[1]
    batch_len = len(train_seq) // batch_size

    for i in range(batch_len):
        batch_steps = np.array(train_steps[i * batch_size : (i + 1) * batch_size])
        max_batch_steps = batch_steps.max()
        for j in range(batch_size):
            seq = train_seq[i * batch_size + j]
            padded_seq = np.pad(np.array(seq),(0, max_batch_steps-len(seq)+1),'constant') # padding with 0
            x.append(padded_seq[:-1])
            y.append(padded_seq[1:])
        x = np.array(x)
        y = np.array(y)
        xs.append(x)
        ys.append(y)
        ss.append(batch_steps)
        x = []
        y = []
    rest_len = len(train_steps[batch_len * batch_size : ])
    if rest_len != 0:
        batch_steps = np.array(train_steps[batch_len * batch_size : ])
        max_batch_steps = batch_steps.max()
        for j in range(rest_len):
            seq = train_seq[batch_len * batch_size + j]
            padded_seq = np.pad(np.array(seq),(0, max_batch_steps-len(seq)+1),'constant')
            x.append(padded_seq[:-1])
            y.append(padded_seq[1:])
        x = np.array(x)
        y = np.array(y)
        xs.append(x)
        ys.append(y)
        ss.append(batch_steps)
    # Enumerator over the batches.
    return xs, ys, ss


# def main():
#     train_data, valid_data,  test_data, nodes, node_to_id = \
#         read_raw_data('generated_data/random-exp-1024-cascades')

#     x_train, y_train, seq_length = batch_generator(train_data)


#     # print(x_train.shape)

#     print(to_nodes(x_train[0][1], nodes))

#     print(to_nodes(y_train[0][1], nodes))
#     print(seq_length)

# if __name__ == '__main__':
#     main()