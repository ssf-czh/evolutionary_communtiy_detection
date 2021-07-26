import numpy as np

def save_embedding(emb_file, features, nodeId = None):
    # save node embedding into emb_file with word2vec format
    f_emb = open(emb_file, 'w')
    f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")

    if nodeId is None:
        nodeId = {i: j for i, j in enumerate(range(features.shape[0]))}

    for i in range(len(features)):
        s = str(nodeId[i]) + " " + " ".join(str(f) for f in features[i].tolist())
        f_emb.write(s + "\n")
    f_emb.close()


def read_embedding(emb_file, ret_map=False):
    data = np.loadtxt(emb_file, dtype=float, delimiter=' ', skiprows=1)
    nodeID = data[:, 0].astype(int)
    map_id2node = {i: j for i, j in enumerate(nodeID)}
    map_node2id = {j: i for i, j in enumerate(nodeID)}
    vectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID
    if ret_map is False:
        return vectors,map_id2node
    else:
        return vectors,map_id2node,map_node2id

def read_features(feat_file):
    data = np.loadtxt(feat_file, dtype=float, delimiter=' ', skiprows=0)
    nodeID = data[:, 0].astype(int)
    map_id2node = {i: j for i, j in enumerate(nodeID)}
    vectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID
    return vectors,map_id2node


def save_map(map_file,map_id2node):
    f_map = open(map_file, 'w')
    for k,v in map_id2node.items():
        s = str(k)+ ' ' + str(v)
        f_map.write(s + "\n")
    f_map.close()


def read_map(map_file,force_int=True):
    map_node2id = {}
    map_id2node = {}
    with open(map_file,'r') as f:
        for row in f.readlines():
            m = row.split(' ',-1)
            if len(m) != 2:
                raise Exception('Error mapfile format')
            # m0 is id
            # m1 is original node name
            m0 = int(m[0])
            if force_int:
                m1 = int(m[1])
            else:
                m1 = m[1]

            map_id2node[m0] = m1
            map_node2id[m1] = m0

    return map_id2node,map_node2id
