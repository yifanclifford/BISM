import numpy as np
import pytrec_eval
import torch


# def sample_negative(pos, N, n):
#     num_pos = len(pos)
#     neg = np.setdiff1d(np.arange(N), pos)
#     return np.random.choice(neg, n * num_pos, True)


def read_neg(file):
    C = []
    with open(file, 'r') as file:
        for line in file:
            items = line.split('\t')
            cand = [int(i) for i in items[1:]]
            item = items[0].split(',')[1]
            C.append([int(item[:-1])] + cand)
    # print(C)
    return torch.tensor(C, dtype=torch.long)


def mmread(R, type='float32'):
    row = R.row.astype(int)
    col = R.col.astype(int)
    val = torch.from_numpy(R.data.astype(type))
    index = torch.from_numpy(np.row_stack((row, col)))
    m, n = R.shape
    return torch.sparse.FloatTensor(index, val, torch.Size([m, n]))


def dist(X):
    v = torch.sum(X * X, 1, keepdim=True)
    Y = v + torch.t(v) - 2 * torch.matmul(X, X.t())
    # Y[Y < 0] = 0
    return Y


def sort2query_batch(run, user):
    n = run.shape[1]
    return {str(u): {str(int(run[u, j])): float(1.0 / (j + 1)) for j in range(n)} for u in user}


def sort2query(run):
    m, n = run.shape
    return {str(i): {str(int(run[i, j])): float(1.0 / (j + 1)) for j in range(n)} for i in range(m)}


def cand2query(score, candidate):
    return {str(i): {str(int(c)): float(s) for s, c in zip(sco, cand)} for i, (sco, cand) in
            enumerate(zip(score, candidate))}


def csr2test(test):
    return {str(r): {str(test.indices[ind]): int(1)
                     for ind in range(test.indptr[r], test.indptr[r + 1])}
            for r in range(test.shape[0]) if test.indptr[r] != test.indptr[r + 1]}


def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


# haddling the case when we have too much positive items
def sample_positive(pos, n):
    return np.random.choice(pos, n, True)


def sample_negative(pos, N, n):
    num_pos = len(pos)
    neg = np.setdiff1d(np.arange(N), pos)
    return np.random.choice(neg, n * num_pos, True)


def form_index(C):
    m, n = C.shape
    x = torch.arange(m).unsqueeze(-1)
    x = x.expand(-1, n)
    x = torch.flatten(x)
    y = torch.flatten(C)
    return x, y


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)

# class Dataset:
#     def __init__(self, params):
#         super(Dataset, self).__init__()
#         train_file = '{}/{}/{}.train.rating'.format(params['dataset_dir'], params['dataset'], params['dataset'])
#         test_file = '{}/{}/{}.test.rating'.format(params['dataset_dir'], params['dataset'], params['dataset'])
#         neg_file = '{}/{}/{}.test.negative'.format(params['dataset_dir'], params['dataset'], params['dataset'])
#         self.user_item_map, self.cand_items, self.num_user, self.num_item = self.read(train_file,
#                                                                                       test_file,
#                                                                                       neg_file)
#
#     def read(self, train_file, test_file, neg_file):
#         cols = ['user', 'item', 'rating', 'timestamp']
#         train_data = pd.read_csv(train_file, sep='\t', names=cols, engine='python')
#
#         user_item_map = {user: items['item'].values for user, items in train_data.groupby('user')}
#         # user_rating_map = {user: items['rating'].values.tolist() for user, items in train_data.groupby('user')}
#
#         test_data = pd.read_csv(test_file, sep='\t', names=cols, engine='python')
#         # test_items = test_data['item'].values
#
#         num_user = max(max(train_data['user']), max(test_data['user'])) + 1
#         num_item = max(max(train_data['item']), max(test_data['item'])) + 1
#
#         return user_item_map, self.read_neg(neg_file), num_user, num_item
#
#
#
#     def get_train_user(self, user):
#         return self.user_item_map[user]
#
#     def get_test_users(self, user):
#         return self.user_item_map[user], self.cand_items[user]
