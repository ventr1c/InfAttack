import torch
import copy
import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class Sp_Commen_Attack(object):
    def __init__(self, A, X, labels, isolated_nodes, device):
        self.dim = A.shape[0]
        self.A_ori = A + sp.eye(self.dim)
        self.A_hat = self.A_ori.tolil()
        self.X = X
        self.labels = labels
        self.isolated_nodes = isolated_nodes
        self.device = device


    def _normalize(self, tensor):
        tensor = sp.coo_matrix(tensor)
        rowsum = np.array(tensor.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        tensor = tensor.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        tensor = tensor.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((tensor.row, tensor.col)).astype(np.int64))
        values = torch.from_numpy(tensor.data)
        shape = torch.Size(tensor.shape)
        return torch.sparse_coo_tensor(indices, values, shape, device=self.device)

    def _calculate(self, model, A, K):
        A = self._normalize(A)
        if model._get_name() == 'SGC':
            X = torch.FloatTensor(np.array(self.X.todense()))
            for i in range(K):
                X = torch.spmm(A, X)
        else:
            X = torch.FloatTensor(np.array(self.X.todense()))
        output = model(X, A)
        return output



class Sp_Target_Attack(Sp_Commen_Attack):
    def __init__(self, adj, features, labels, isolated_nodes, device, target, c, pred_Y):
        super(Sp_Target_Attack, self).__init__(adj, features, labels, isolated_nodes, device)
        self.target = target
        self.yv = self.labels[self.target]
        self.c = c
        self.epsilon = 1e-4
        self.d = np.squeeze(np.array(self.A_ori.sum(1)), 1)
        self.pred_Y = pred_Y
        assert target not in self.isolated_nodes, 'the target node is isolated'


    def candidate_nodes_set(self, i, K, dataset):
        CN_a = (self.labels == self.c[i]).nonzero()[0].tolist()
        v_neighbor = self.A_hat[self.target[i]].nonzero()[1].tolist()
        CN_a = list(set(CN_a).difference(set(v_neighbor)))
        if dataset == 'pubmed' and K>2:
            if self.c[i]==0:
                CN_a = np.random.choice(CN_a, 100, replace=False)
            else:
                CN_a = np.random.choice(CN_a, 200, replace=False)
        CN_r = []
        for n in v_neighbor:
            if n != self.target[i] and self.labels[n] == self.yv[i]:
                CN_r.append(n)
        return np.array(CN_a), np.array(CN_r, dtype=int)


    def original_label_influence(self, K, v, s, i):
        if K == 0:
            self.I_l_yv += s * self.pred_Y[v, self.yv[i]]
            self.I_l_c += s * self.pred_Y[v, self.c[i]]
            return
        neighbor = self.A_hat[v].nonzero()[1]
        for n in neighbor:
            w = 1 / self.d[v] ** 0.5 * 1 / self.d[n] ** 0.5
            self.original_label_influence(K - 1, n, s * w, i)

    def delta_label_influence(self, CN, op, i, K):
        delta_I_l = np.empty(CN.shape)
        d_target = self.d[self.target[i]]
        if op == 'add':
            self.d[self.target[i]] = d_target + 1
            for j, u in enumerate(CN):
                self.I_l_c, self.I_l_yv = 0, 0
                self.d[u] += 1
                self.A_hat[self.target[i], u] = 1
                self.A_hat[u, self.target[i]] = 1
                self.original_label_influence(K - 1, u, 1, i)
                self.A_hat[self.target[i], u] = 0
                self.A_hat[u, self.target[i]] = 0

                delta_I_l[j] = 1 / self.d[u] ** 0.5 * 1 / self.d[self.target[i]] ** 0.5 \
                               * (self.I_l_c - self.I_l_yv )
                if K==2:
                    delta_I_l[j] += 1 / self.d[u] ** 0.5 * 1 / self.d[self.target[i]] **1.5 \
                                    * (self.pred_Y[u, self.c[i]] - self.pred_Y[u, self.yv[i]])
                self.d[u] -= 1

        elif op == 'remove':
            self.d[self.target[i]] = d_target - 1
            for j, u in enumerate(CN):
                self.I_l_c, self.I_l_yv = 0, 0
                self.original_label_influence(K - 1, u, 1, i)
                delta_I_l[j] = 1 / self.d[u] ** 0.5 * 1 / self.d[self.target[i]] ** 0.5 \
                               * (self.I_l_yv - self.I_l_c )
                if K==2:
                    delta_I_l[j] += 1 / self.d[u] ** 0.5 * 1 / self.d[self.target[i]] **1.5 \
                                    * (self.pred_Y[u, self.yv[i]] - self.pred_Y[u, self.c[i]])
        self.d[self.target[i]] = d_target
        return delta_I_l

    def label_influence(self, op, i, K):
        self.I_l_c, self.I_l_yv = 0, 0
        d_target = self.d[self.target[i]]
        if op == 'add':
            self.d[self.target[i]] = d_target + 1
        elif op == 'remove':
            self.d[self.target[i]] = d_target - 1
        self.original_label_influence(K, self.target[i], 1, i)
        C_0 = self.I_l_c - self.I_l_yv
        self.d[self.target[i]] = d_target
        return C_0

    def iter_label_influence(self, op, CN, i, K):
        I_l = np.empty(CN.shape)
        d_target = self.d[self.target[i]]
        if op == 'add':
            self.d[self.target[i]] = d_target + 1
            for j, u in enumerate(CN):
                self.I_l_c, self.I_l_yv = 0, 0
                self.d[u] += 1
                self.A_hat[self.target[i], u] = 1
                self.A_hat[u, self.target[i]] = 1
                self.original_label_influence(K, self.target[i], 1, i)
                I_l[j] = self.I_l_c - self.I_l_yv
                self.A_hat[self.target[i], u] = 0
                self.A_hat[u, self.target[i]] = 0
                self.d[u] -= 1
        elif op == 'remove':
            self.d[self.target[i]] = d_target - 1
            for j, u in enumerate(CN):
                self.I_l_c, self.I_l_yv = 0, 0
                self.d[u] -= 1
                self.A_hat[self.target[i], u] = 0
                self.A_hat[u, self.target[i]] = 0
                self.original_label_influence(K, self.target[i], 1, i)
                I_l[j] = self.I_l_c - self.I_l_yv
                self.A_hat[self.target[i], u] = 1
                self.A_hat[u, self.target[i]] = 1
                self.d[u] += 1
        self.d[self.target[i]] = d_target
        return I_l

    def iterative(self, model, perturbs, K, ground_truth):
        count_accs, times = [], []
        for perturb in perturbs:
            t = time.time()
            print("##### ... Number of adversarial samples: {} ... #####:".format(perturb))
            count_acc = np.array([False] * len(self.target))
            for i in tqdm(range(self.target.shape[0])):
                degree = copy.deepcopy(self.d)
                print('\n target node: {}, y_v label : {}, c lable : {}'
                      .format(self.target[i], self.yv[i], self.c[i]))
                for _ in range(perturb):
                    CN_a, CN_r = self.candidate_nodes_set(i, K, '')
                    I_l_a = self.iter_label_influence('add', CN_a, i, K)
                    I_l_r = self.iter_label_influence('remove', CN_r, i, K)
                    I_l = np.concatenate((I_l_a, I_l_r), axis=0)
                    index = np.argsort(I_l)[::-1]
                    U = np.concatenate((CN_a, CN_r), axis=0)[index]
                    if U[0] in CN_a:
                        self.A_hat[self.target[i], U[0]] = 1
                        self.A_hat[U[0], self.target[i]] = 1
                        self.d[self.target[i]] += 1
                    elif U[0] in CN_r:
                        self.A_hat[self.target[i], U[0]] = 0
                        self.A_hat[U[0], self.target[i]] = 0
                        self.d[self.target[i]] -= 1
                    else:
                        continue
                    print('candidate node : {}, y_v label : {}, label influence : {}'
                          .format(U[0], self.labels[U[0]], I_l[index][0]))
                h_v = self._calculate(model, self.A_hat, K)[self.target[i]]
                if h_v[self.c[i]].item() > h_v[ground_truth[i]].item():
                    count_acc[i] = True
                    print('attack succeed !')
                else:
                    print('attack failed !')
                print('target node {} neighbor nodes before attack: {}'
                      .format(self.target[i], self.A_ori[self.target[i]].nonzero()[1]))
                print('target node {} neighbor nodes label before attack: {}'
                      .format(self.target[i], self.labels[self.A_ori[self.target[i]].nonzero()[1]]))
                print('target node {} neighbor nodes after attack: {}'
                      .format(self.target[i], self.A_hat[self.target[i]].nonzero()[1]))
                print('target node {} neighbor nodes label after attack: {}'
                      .format(self.target[i], self.labels[self.A_hat[self.target[i]].nonzero()[1]]))
                self.A_hat = copy.deepcopy(self.A_ori).tolil()
                self.d = copy.deepcopy(degree)
            count_accs.append(sum(count_acc) / len(count_acc))
            times.append(time.time() - t)
        return count_accs, times

    def one_time(self, model, perturbs, K, dataset, ground_truth):
        count_accs, times = [], []
        for perturb in perturbs:
            t = time.time()
            print("##### ... Number of adversarial samples: {} ... #####:".format(perturb))
            count_acc = np.array([False] * len(self.target))
            for i in tqdm(range(self.target.shape[0])):
                degree = copy.deepcopy(self.d)
                CN_a, CN_r = self.candidate_nodes_set(i, K, dataset)
                delta_I_l_a = self.delta_label_influence(CN_a, 'add', i, K)
                delta_I_l_r = self.delta_label_influence(CN_r, 'remove', i, K) \
                    if self.d[self.target[i]] > 1.0 else np.array([], dtype=int)
                print('\n target node: {}, y_v label : {}, c lable : {}'
                      .format(self.target[i], self.yv[i], self.c[i]))
                for _ in range(perturb):
                    C_0_a = self.label_influence('add', i, K)
                    if (delta_I_l_r==-1e9).all():
                        C_0_r, delta_I_l_r = np.array([], dtype=int), np.array([], dtype=int)
                    else:
                        C_0_r = self.label_influence('remove', i, K)
                    I_l = np.concatenate((C_0_a + delta_I_l_a, C_0_r + delta_I_l_r), axis=0)
                    index = np.argsort(I_l)[::-1]
                    U = np.concatenate((CN_a, CN_r), axis=0)[index]
                    if index[0] >= delta_I_l_a.shape[0]:
                        delta_I_l_r[index[0] - delta_I_l_a.shape[0]] = -1e9
                    else:
                        delta_I_l_a[index[0]] = -1e9
                    if U[0] in CN_a:
                        self.A_hat[self.target[i], U[0]] = 1
                        self.A_hat[U[0], self.target[i]] = 1
                        self.d[self.target[i]] += 1
                    elif U[0] in CN_r:
                        self.A_hat[self.target[i], U[0]] = 0
                        self.A_hat[U[0], self.target[i]] = 0
                        self.d[self.target[i]] -= 1
                    else:
                        continue
                    print('candidate node : {}, y_v label : {}, label influence : {}'
                          .format(U[0], self.labels[U[0]], I_l[index][0]))
                h_v = self._calculate(model, self.A_hat, K)[self.target[i]]
                if h_v[self.c[i]].item() > h_v[ground_truth[i]].item():
                    count_acc[i] = True
                    print('attack succeed !')
                else:
                    print('attack failed !')
                print('target node {} neighbor nodes before attack: {}'
                      .format(self.target[i], self.A_ori[self.target[i]].nonzero()[1]))
                print('target node {} neighbor nodes label before attack: {}'
                      .format(self.target[i], self.labels[self.A_ori[self.target[i]].nonzero()[1]]))
                print('target node {} neighbor nodes after attack: {}'
                      .format(self.target[i], self.A_hat[self.target[i]].nonzero()[1]))
                print('target node {} neighbor nodes label after attack: {}'
                      .format(self.target[i], self.labels[self.A_hat[self.target[i]].nonzero()[1]]))
                self.A_hat = copy.deepcopy(self.A_ori).tolil()
                self.d = copy.deepcopy(degree)
            count_accs.append(sum(count_acc) / len(count_acc))
            times.append(time.time() - t)

        return count_accs, times