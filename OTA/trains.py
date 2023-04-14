import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import accuracy


def train(model, adj, features, labels, idx_train, idx_val, args, dataset):
    if model._get_name() == 'SGC':
        sgc_train(model, adj, features, labels, idx_train, idx_val, args, dataset)
    else:
        optimizer = optim.Adam(model.parameters(),
                            lr=args['lr'], weight_decay=args['weight_decay'])
        for epoch in range(args['epochs']):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()


def sgc_train(model, adj, features, labels, idx_train, idx_val, args, dataset):
    optimizer = optim.Adam(model.parameters(), lr=args['lr'],
                           weight_decay = args['{}_weight_decay'.format(dataset)])
    for epoch in range(args['epochs']):
        model.train()
        optimizer.zero_grad()
        output = model(features[idx_train], adj)
        loss_train = F.nll_loss(output, labels[idx_train])
        loss_train.backward()
        optimizer.step()


def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    preds, correct= accuracy(output[idx_test], labels[idx_test])
    acc_test = correct.sum() / len(correct)
    print(acc_test)
    return preds, correct