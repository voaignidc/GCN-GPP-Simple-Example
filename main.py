import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from gcn import Net
from datasets import MovementLib, EdgeIndexCU, EdgeAttrCU
from config import args

# torch.set_printoptions(profile='full')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(args):
    dataset = MovementLib(args['dataset'], args['num_nodes'])
    sample_x = torch.zeros(1, args['num_feats'])
    sample_y = torch.zeros(1, ).long()

    x = torch.zeros(args['num_nodes'], args['num_feats'])
    y = torch.zeros(1, args['num_nodes']).long()

    for index, data in enumerate(dataset):
        if index==0:
            sample_x = data[0]
            sample_y = data[1]
        else:   
            x[index-1] = data[0]
            y[:, index-1] = data[1]
    y = y.squeeze()    

    ei = EdgeIndexCU(dataset.__len__())
    edge_index = ei.get_edge_index()
    ea = EdgeAttrCU(x)#, sample_feat=torch.randn(90,))
    edge_attr = ea.get_edge_attr()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, y, sample_x, sample_y, edge_attr
    
#--- get train graph --- 
# args['dataset'] = './input/ML7_1.csv'
data, y, sample_x, sample_y, edge_attr = get_data(args) 
data = data.to(device)
y = y.to(device)
edge_attr = edge_attr.to(device)
sample_x = sample_x.to(device)
sample_y = sample_y.to(device)

#--- train ---  
model = Net(args, args['num_feats'], args['num_ids'])
model = model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=1e-4) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    # out = model(data.x, edge_attr, None)
    loss = criterion(out, y)
    print(loss.item())
    loss.backward()
    optimizer.step()

#--- eval train --- 
model.eval()
_, pred = model(data).max(dim=1)
# _, pred = model(data.x, edge_attr, None).max(dim=1)
print(pred)
print(y)

#--------------------------------------------- 

#--- get another graph for testing ---  
args['dataset'] = './input/ML7_2.csv'
data, y, sample_x, sample_y, edge_attr = get_data(args) 
data = data.to(device)
y = y.to(device)
edge_attr = edge_attr.to(device)
sample_x = sample_x.to(device)
sample_y = sample_y.to(device)

#--- eval test --- 
model.eval()
_, pred = model(data).max(dim=1)
# _, pred = model(data.x, edge_attr, None).max(dim=1)
print(pred)
print(y)