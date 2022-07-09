import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)




class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass, bias=False)

    def forward(self, x, adj):
        x = self.W(x)
        return F.log_softmax(x, dim=1)




def model_construct(model, args, *parm):
    if model=='GCN':
        gcn = GCN(nfeat=parm[0].shape[1],
                  nhid=args['hidden'],
                  nclass=parm[1].max().item() + 1,
                  dropout=args['dropout'])
        return gcn


    elif model=='SGC':
        sgc = SGC(nfeat=parm[0].shape[1],
                  nclass=parm[1].max().item() + 1)
        return sgc


    else:
        raise NotImplementedError('model: {} is not implemented'.format(model))