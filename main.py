import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import Data, DataLoader

from gcn import Net
from datasets import MovementLib, EdgeAttrCU#, EdgeIndexCU
from config import args

torch.set_printoptions(profile='full')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(args):
    dataset = MovementLib(args['dataset'], args['num_nodes'])
    sample_x = torch.zeros(1, args['num_feats'])
    # sample_y = torch.zeros(1, )
    x = torch.zeros(args['num_nodes'], args['num_feats'])
    y = torch.zeros(1, args['num_nodes'])

    for index, data in enumerate(dataset):
        if index==0:
            sample_x[:] = data[0]
            # sample_y[:] = data[1]
        else:   
            x[index-1] = data[0]
            y[:, index-1] = data[1]
    y = y.squeeze()  
    x -= sample_x
    
    # ei = EdgeIndexCU(dataset.__len__())
    # edge_index = ei.get_edge_index()
    ea = EdgeAttrCU(x)
    edge_attr = ea.get_edge_attr()

    return x, y, edge_attr

def train(args, model, optimizer, criterion):  
    model.train()
    for e in range(80):
        optimizer.zero_grad()
        loss = 0.
        for dataset_i in range(80):  
            #--- get train graph --- 
            args['dataset'] = './input/ML7_'+str(dataset_i)+'.csv'
            x, y, edge_attr = get_data(args) 
            x = x.to(device)
            y = y.to(device)
            edge_attr = edge_attr.to(device)

            out = model(x, edge_attr)
            loss += criterion(out, y)

        loss /= 80.
        print(e, loss.item())    
        loss.backward()
        optimizer.step()
    
def test(args, model):
    #--- get another graph for testing ---  
    x, y, edge_attr = get_data(args) 
    x = x.to(device)
    y = y.to(device)
    edge_attr = edge_attr.to(device)

    #--- eval test --- 
    model.eval()
    pred = model(x, edge_attr).detach().cpu().numpy() > 0.5
    print('pred=', pred+0)
    print('   y=', y.long().detach().cpu().numpy())
    print(' ')
    
if __name__ == "__main__": 
    model = Net(args, args['num_feats'], 1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.BCELoss().cuda()
    
    train(args, model, optimizer, criterion)      
    print("-----------------------------------------")    
    
    for i in range(20):
        args['dataset'] = './input/ML7_'+str(80+i)+'.csv'
        test(args, model)  
    