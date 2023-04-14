from __future__ import division
from __future__ import print_function

import sys
import copy
import torch
import argparse
import numpy as np

import utils
import scipy.sparse as sp
from utils import load_data
from trains import train, test
from models import model_construct
from modelparser import model_parser
from sklearn.linear_model import LogisticRegression

sys.path.append('../')
# Attack settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42,help='Random seed.')
parser.add_argument('--sparse', action='store_true', default=True,help='Adj with sparse version or not.')
parser.add_argument('--target_nodes_num', type=int, default=100,help='perturbed edges nums in non-target')
parser.add_argument('--attack_object', type=str, default='white',choices=['white', 'gray'],help='white-box attack and gray-box attack')
parser.add_argument('--model', type=str, default='GCN',choices=['GCN', 'SGC'],help='Model used to attack')
parser.add_argument('--dataset', type=str, default='citeseer',choices=['cora', 'citeseer', 'pubmed'],help='Dataset used to attack')
parser.add_argument('--attack_algorithm', type=str, default='one-time',choices=['one-time', 'iterative'],help='Comparative Experiment')

args = parser.parse_args()

args.cuda = False
device = torch.device('cpu')

# random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, tn_adj, features, tn_features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args)
idx_test_full = range(min(idx_test), max(idx_test)+1)
isolated_nodes = np.array(list(set(idx_test_full).difference(set(idx_test))))

model_args = model_parser(args.model, args.dataset)
model_before_attack = model_construct(args.model, model_args, features, labels)
if model_before_attack._get_name() == 'SGC':
    for i in range(2):
        tn_features = torch.spmm(tn_adj, tn_features)

model_before_attack.to(device)
tn_adj = tn_adj.to(device)
tn_features = tn_features.to(device)
labels = labels.to(device)

# # Train model
# t_total = time.time()
# train(model_before_attack, tn_adj, tn_features, labels, idx_train, idx_val, model_args, args.dataset)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# # Save model
# torch.save(model_before_attack, '../modelpath/{}_{}_bias.pth'.format(args.model, args.dataset))

# Load model
model_before_attack = torch.load('../modelpath/{}_{}.pth'.format(args.model, args.dataset))
preds, correct = test(model_before_attack, tn_adj, tn_features, labels, idx_test)


if args.model == 'GCN':
    path = '../targetnodes/{}_{}_100.txt'.format(args.model, args.dataset)
elif args.model == 'SGC':
    path = '../targetnodes/{}_{}_100.txt'.format(args.model, args.dataset)
else:
    path = None
target_node = np.loadtxt(path, delimiter=',').astype(np.int64)
print(target_node)

h_v = model_before_attack(tn_features, tn_adj)
pred_label = torch.argsort(h_v, descending=True).numpy()
if args.attack_object == 'white':
    labels = labels.numpy()
    ground_truth = copy.deepcopy(labels[target_node])

elif args.attack_object == 'gray':
    ground_truth = copy.deepcopy(labels[target_node])
    labels = pred_label[:, 0]
    if args.dataset == 'citeseer':
        labels[isolated_nodes] = -1


# Using LR to predict the input Y of LP model
fea_train = features[idx_train]
label_train = labels[idx_train]
nor_adj = utils.normalize_adj(adj+sp.eye(adj.shape[0]))
A_hat = nor_adj.copy()
for i in range(1):
    A_hat = nor_adj.dot(A_hat)
LR_model = LogisticRegression(C=1, penalty='l2', solver='lbfgs',
                              fit_intercept=True, max_iter=100).fit(fea_train,label_train)
pred_test = LR_model.predict_proba(features)
Y = pred_test.copy()
output = A_hat.dot(Y)
pred = np.argsort(output)[:,-1]
acc = np.sum(pred[idx_test]==labels[idx_test])/len(idx_test)
print(acc)



if args.attack_algorithm=='one-time':
    from attack import Sp_Target_Attack

    attacked_label = pred_label[target_node][:, 1]
    classes = labels.max().item() + 1
    perturbs = [2, 4, 6, 8, 10]
    attack = Sp_Target_Attack(adj, features, labels, isolated_nodes, device, target_node,
                              attacked_label, Y)
    count_accs,  times \
        = attack.one_time(model_before_attack, perturbs,  2, args.dataset, ground_truth)
    print("count_accs: {}".format(np.array(count_accs)))
    print("count_edges: {}".format(np.array(perturbs)))
    print("times: {}".format(np.array(times)))


elif args.attack_algorithm=='iterative':
    from attack import Sp_Target_Attack

    attacked_label = pred_label[target_node][:, 1]
    classes = labels.max().item() + 1
    perturbs = [2, 4, 6, 8, 10]
    attack = Sp_Target_Attack(adj, features, labels, isolated_nodes, device, target_node,
                              attacked_label, Y)
    count_accs, times = attack.iterative(model_before_attack, perturbs, 2, ground_truth)
    print("count_accs: {}".format(np.array(count_accs)))
    print("count_edges: {}".format(np.array(perturbs)))
    print("times: {}".format(np.array(times)))

else:
    raise NotImplementedError('attack_algorithm:{} is not implemented'.format(args.attack_algorithm))