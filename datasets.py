import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import dataset

class MovementLib(dataset.Dataset):
    def __init__(self, path, num_nodes):
        self.data = pd.read_csv(path, header=None, iterator=True)
        self.num_nodes = num_nodes
    
    def __getitem__(self, index):
        data = self.data.get_chunk(1).values.astype('float').squeeze()
        feat = torch.from_numpy(data[0:90])
        label = torch.from_numpy(np.array(data[-1].astype('int')))
        return feat, label
    
    def __len__(self):
        return self.num_nodes   
        
        
class EdgeIndexCU(object): 
    """
    Get 'edge_index' when graph is complete undirected.
    :v_num: Num of nodes.
    """
    def __init__(self, v_num):
        self.v_num = v_num
        self.edge_index0 = []
        self.edge_index1 = []
        
    def get_edge_index(self):
        self._a(self.edge_index0)
        # self._b(self.edge_index0)
        self._b(self.edge_index1)
        # self._a(self.edge_index1)
        
        self.edge_index0 = torch.from_numpy(np.array(self.edge_index0))
        self.edge_index1 = torch.from_numpy(np.array(self.edge_index1))
        edge_index = torch.zeros(2, int(self.v_num*(self.v_num-1)/2))
        edge_index[0] = self.edge_index0
        edge_index[1] = self.edge_index1
        edge_index = edge_index.long()
        return edge_index

    def _a(self, edge_index):
        times = self.v_num
        for node_index in range(self.v_num):
            times -= 1
            for time in range(times):
                edge_index.append(node_index)

    def _b(self, edge_index):
        times = self.v_num+1
        for node_index in range(self.v_num):
            times -= 1
            for time in range(1, times):
                edge_index.append(time+node_index) 
                
                
class EdgeAttrCU(object): 
    """
    Get adjacency matrix 'edge_attr' according to similarity of each nodes,
    when graph is complete undirected.
    :v_feat: Features of nodes. Nv*Nf
    """
    def __init__(self, v_feat): 
        Nv = v_feat.size(0)
        edge_attr = torch.zeros(Nv, Nv)
        # Weighted Aggregation 
        edge_attr = torch.mm(v_feat, v_feat.T)
        for i in range(Nv):
            for j in range(Nv):
                if i==j:
                    edge_attr[i, j] = 0
            
        # Each row of A should be normalized by softmax function.
        self.edge_attr = F.softmax(edge_attr, dim=-1) 
 
    def get_edge_attr(self):    
        return self.edge_attr        
                
                
if __name__ == "__main__":                
    ei = EdgeIndexCU(6)
    edge_index = ei.get_edge_index()
    print(edge_index)
    print(edge_index.type())
    print(edge_index.shape)