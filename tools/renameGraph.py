import re
import numpy as np

def renumber(input_file, new_adjlist_file, node_idx_file):
    nodes = []
    edges = []
    with open(input_file, 'r') as f:
        for row in f:
            n1, n2 = re.split('[ \t]', row.strip())
            edges.append([n1, n2])
            if n1 not in nodes: nodes.append(n1)
            if n2 not in nodes: nodes.append(n2)

    node_idx = {}
    for i, node in enumerate(nodes):
        node_idx[node] = i

    with open(new_adjlist_file, 'w') as f:
        for edge in edges:
            n1, n2 = edge
            line = '{}\t{}\n'.format(node_idx[n1], node_idx[n2])
            f.write(line)

    with open(node_idx_file, 'w') as f:
        for node, idx in node_idx.items():
            line = '{}\t{}\n'.format(node, idx)
            f.write(line)
    # print(node_idx_file)


# ========================


# 处理属性网络的节点的名字，属性，类别标签形式的文件。
def make_cmu(input_file, node_idx_file, output_cmu_file, output_feature_file):
    node_feature_label = np.genfromtxt(input_file, dtype=np.dtype(str))
    nodes = node_feature_label[:, 0]
    features = node_feature_label[:, 1:-1]
    labels = node_feature_label[:, -1]

    node_in_feat = {}
    label_nodes = {}
    for label in set(labels):
        label_nodes[label] = []
    for i, node in enumerate(nodes):
        label_nodes[labels[i]].append(node)
        node_in_feat[node] = i

    node_idx = {}
    id_node = {}
    with open(node_idx_file, 'r') as f:
        for row in f:
            node, idx = re.split('[ \t]', row.strip())
            idx = int(idx)
            node_idx[node] = idx
            id_node[idx] = node

    with open(output_cmu_file, 'w') as f:
        for _, cmu_nodes in label_nodes.items():
            cmu_nodes_idx = [node_idx[n] for n in cmu_nodes]
            line = ' '.join([str(idx) for idx in cmu_nodes_idx]) #?
            f.write(line + '\n')
    # print(output_cmu_file)

    with open(output_feature_file, 'w') as f:
        # for i, node in enumerate(nodes):
        #     line = ' '.join(features[i])
        #     f.write('{} {}\n'.format(node_idx[node], line))
        for i in range(len(nodes)):
            node_name = id_node[i]
            feat_idx = node_in_feat[node_name]
            line = ' '.join(features[feat_idx])
            f.write('{} {}\n'.format(i, line))

    print(output_feature_file)





# io_paths = [['../dataset/cora/cora.cites',
#              '../dataset/cora/remap/cora.ugraph', '../dataset/cora/remap/cora.map']
#            ]
# 
# for io_path in io_paths:
#     input_file, new_adjlist_file, node_idx_file = io_path
#     renumber(input_file, new_adjlist_file, node_idx_file)
# 
# 
# io_paths = [['../dataset/cora/cora.content', '../dataset/cora/remap/cora.map',
#              '../dataset/cora/remap/cora.cmty', '../dataset/cora/remap/cora.feat']
#             ]
# 
# for io_path in io_paths:
#     input_file, node_idx_file, output_cmu_file, output_feature_file = io_path
#     make_cmu(input_file, node_idx_file, output_cmu_file, output_feature_file)


io_paths = [['../dataset/citeseer/citeseer.cites',
             '../dataset/citeseer/remap/citeseer.ungraph', '../dataset/citeseer/remap/citeseer.map']
           ]

for io_path in io_paths:
    input_file, new_adjlist_file, node_idx_file = io_path
    renumber(input_file, new_adjlist_file, node_idx_file)


io_paths = [['../dataset/citeseer/citeseer.content', '../dataset/citeseer/remap/citeseer.map',
             '../dataset/citeseer/remap/citeseer.cmty', '../dataset/citeseer/remap/citeseer.feat']
            ]

for io_path in io_paths:
    input_file, node_idx_file, output_cmu_file, output_feature_file = io_path
    make_cmu(input_file, node_idx_file, output_cmu_file, output_feature_file)
    