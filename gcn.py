import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, A, features):
        x = torch.mm(A, features)
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(A, features)
        cat_feats = torch.cat([agg_feats, features], dim=1)
        out = torch.einsum('nd,df->nf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out

class Net(nn.Module):
    def __init__(self, args, input_channels, output_channels):
        super(Net, self).__init__()
        # In this case, batchNorm will cause a really bad result, don't use.
        # self.bn0 = nn.BatchNorm1d(input_channels, affine=False)
        self.conv1 = GraphConv(input_channels, input_channels, MeanAggregator)
        self.conv2 = GraphConv(input_channels, input_channels, MeanAggregator)
        self.conv3 = GraphConv(input_channels, args['layers'], MeanAggregator)
        self.conv4 = GraphConv(args['layers'], args['layers'],MeanAggregator)
        
        self.classifier = nn.Sequential(
                            nn.Linear(args['layers'], args['layers']),
                            nn.PReLU(args['layers']),
                            nn.Linear(args['layers'], output_channels))
    
    def forward(self, x, A, train=True):
        # N,D = x.shape
        # x = x.view(-1, D)
        # x = self.bn0(x)
        # x = x.view(N,D)

        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A) 
        pred = torch.sigmoid(self.classifier(x).squeeze())
        
        return pred