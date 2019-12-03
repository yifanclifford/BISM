import argparse
import os
import shutil
import numpy as np
import torch
import yaml
from scipy import io
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm

from lib.model import BISM
from lib.utils import Evaluator, csr2test, sort2query, cand2query, read_neg


def evaluate_cand():
    prediction = dict()
    for user in tqdm(loader):
        R = torch.from_numpy(train_data[user].todense().astype('float32')).to(args.device)
        score = model.predict(R)
        score = torch.gather(score, 1, C[user])
        score, pred = torch.topk(score, args.N, dim=1)
        score = score.cpu()
        pred = pred.cpu()
        for idx, u in enumerate(user):
            prediction[str(u.item())] = {str(int(C[u, pred[idx, i]])): float(score[idx, i]) for i in range(args.N)}
    evaluator.evaluate(prediction, test_data)
    result = evaluator.show(['recall_10', 'recip_rank_cut_10', 'ndcg_cut_10'])
    return result


def evaluate():
    prediction = dict()
    for user in tqdm(loader):
        R = torch.from_numpy(train_data[user].todense().astype('float32')).to(args.device)
        score = model.predict(R)
        rated = train_data[user].tocoo()
        score[rated.row, rated.col] = 0
        score, pred = torch.topk(score, args.N, dim=1)
        score = score.cpu()
        pred = pred.cpu()
        for idx, u in enumerate(user):
            prediction[str(u.item())] = {str(int(pred[idx, i])): float(score[idx, i]) for i in range(args.N)}
    evaluator.evaluate(prediction, test_data)
    result = evaluator.show(['recall_5', 'recall_10', 'recall_15', 'recall_20',
                             'recip_rank_cut_5', 'recip_rank_cut_10', 'recip_rank_cut_15', 'recip_rank_cut_20',
                             'ndcg_cut_10'])
    return result


def train(epoch):
    if args.algo == 'BISM' and epoch % args.initer == 0:
        model.update_F()
    model.update_S()
    return model.object()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='running on the GPU', action='store_true')
    parser.add_argument('--save', help='whether to save each epoch', action='store_true')
    parser.add_argument('--decouple', help='whether to decouple the learning', action='store_true')
    parser.add_argument('--task', type=int, help='get the id of the task', default=0)
    parser.add_argument('--cand', help='recommend from a candidate set of items', action='store_true')
    parser.add_argument('--record', help='whether to record the results for each epoch', action='store_true')
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=.1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=.1)
    parser.add_argument('-l', '--lamb', help='parameter lambda', type=float, default=.1)
    parser.add_argument('-c', help='parameter c', type=int, default=1)
    parser.add_argument('-m', '--maxiter', help='max number of iteration', type=int, default=5)
    parser.add_argument('-N', help='number of recommended items', type=int, default=20)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    with open('config', 'r') as file:
        params = yaml.load(file, Loader=yaml.Loader)

    args.initer = params['inside_iteration']
    path = '{}/{}/{}.train.mtx'.format(params['dataset_dir'], params['dataset'], params['dataset'])
    train_data = io.mmread(path)
    args.m, args.n = train_data.shape
    evaluator = Evaluator({'recall', 'recip_rank_cut', 'ndcg_cut'})

    path = '{}/{}/{}.test.mtx'.format(params['dataset_dir'], params['dataset'], params['dataset'])
    test_data = io.mmread(path)
    test_data = csr2test(test_data.tocsr())

    if args.cand:
        path = '{}/{}/{}.test.negative'.format(params['dataset_dir'], params['dataset'], params['dataset'])
        C = read_neg(path).to(args.device)

    model = BISM(train_data, args).to(args.device)
    users = np.flatnonzero(np.sum(train_data, 1))
    train_data = train_data.tocsr()
    loader = DataLoader(users, batch_size=params['batch_size'], shuffle=False)

    path = '{}/{}/BISM{}'.format(params['save_dir'], params['dataset'], args.task)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    hr = []
    result_all = []

    result = evaluate_cand() if args.cand else evaluate()
    print('initial HR@10={:.4}, ARHR@10={:.4}, NDCG@10={:.4}'.format(result['recall_10'], result['recip_rank_cut_10'],
                                                                     result['ndcg_cut_10']))
    for epoch in range(args.maxiter):
        print('Training =======>')
        train_time = time()
        loss = train(epoch)
        train_time = time() - train_time
        print('Evaluating =======>')
        result = evaluate_cand() if args.cand else evaluate()
        print('epoch={}, loss={:.4f}, HR@10={:.4}, '
              'ARHR@10={:.4}, NDCG@10={:.4}, Train={:.4}'.format(epoch,
                                                                 loss,
                                                                 result['recall_10'],
                                                                 result['recip_rank_cut_10'],
                                                                 result['ndcg_cut_10'], train_time))
        hr.append(result['recall_10'])
        result_all.append(result)
        if args.save:
            path = '{}/{}/BISM{}/BISM{}'.format(params['save_dir'], params['dataset'], args.task, epoch)
            model.cpu()
            torch.save(model.state_dict(), path)
            model.to(args.device)

    hr = np.array(hr)
    best_epoch = np.argmax(hr)
    best_result = result_all[best_epoch]
    print('best epoch={}, HR@10={:.4f}, ARHR@10={:.4f}'.format(best_epoch, hr[best_epoch],
                                                               best_result['recip_rank_cut_10'],
                                                               best_result['ndcg_cut_10']))

    with open('{}/{}'.format(params['result_dir'], params['dataset']), 'a') as file:
        file.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(args.algo, args.task, best_epoch,
                                                                  best_result['recall_5'],
                                                                  best_result['recall_10'],
                                                                  best_result['recall_15'],
                                                                  best_result['recall_20'],
                                                                  best_result['recip_rank_cut_5'],
                                                                  best_result['recip_rank_cut_10'],
                                                                  best_result['recip_rank_cut_15'],
                                                                  best_result['recip_rank_cut_20'],
                                                                  best_result['ndcg_cut_10']))

    if args.record:
        hr = np.array(hr)
        path = '{}/result_{}_BISM'.format(params['result_dir'], params['dataset'])
        np.savetxt(path, hr)
