
from metric import modularity,onmi,nmi
from tools import file_io,utils
import networkx as nx
import clustering

def eval(emb_file,network_file,label_file,clu_map=None,times=1):
    G = nx.read_edgelist(network_file,nodetype=int)
    label = file_io.read_communities(label_file,True)
    emb = utils.read_embedding(emb_file)

    sum_onmi = 0
    max_onmi = 0
    sum_nmi = 0
    max_nmi = 0
    sum_eq = 0
    max_eq = 0

    for i in range(times):
        result = clustering.kmeans_from_vec(emb,len(label),clu_map,True,n_init=10)
        _,vONMI = onmi.onmi(result,label)
        vNMI = nmi.calc(result,label)
        EQ = modularity.cal_EQ(result,G)

        sum_nmi += vNMI
        sum_onmi += vONMI
        sum_eq += EQ

    # print('ONMI: {}, NMI: {}, EQ:{}'.format(vONMI,vNMI,EQ))
    print('avg ONMI: {:.4f}, avg NMI: {:.4f}, avg EQ:{:.4f}'.format(sum_onmi/times, sum_nmi/times, sum_eq/times))

