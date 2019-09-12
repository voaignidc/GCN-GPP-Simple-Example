import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_scatter import scatter_mean
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

torch.set_printoptions(profile='full')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


dataset = Planetoid(root='E:/DataSet/cluster/Cora', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(data.edge_index.T[100:200])



# model.train()
# for epoch in range(200):
    # optimizer.zero_grad()
    # out = model(data)
    # print(out.shape)
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # loss.backward()
    # optimizer.step()
    
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / data.test_mask.sum().item()
# print('Accuracy: {:.4f}'.format(acc))    
    
    

    
