#!/usr/bin/python3
# -*-coding:UTF-8-*-
import time
import argparse
import pickle
from GNF.model import *
from GNF.utils import *
from torch.utils.data import DataLoader
import logging
import torch
import os
import multiprocessing
from tqdm import tqdm
import datetime
import numpy as np


cores = multiprocessing.cpu_count()


# 设置随机数种子
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/yoochoose')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--neighbor_num', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--local_dropout_p', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--global_dropout_p', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--similar_num', type=int, default=12)

opt = parser.parse_args()


def forward(model, data):
    mask, targets, inputs, graph_adj_matrix, alias_graph_item, current_graph_mat, current_graph_mask, path_num = data
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    path_num = trans_to_cuda(path_num).float()
    graph_adj_matrix = trans_to_cuda(graph_adj_matrix).long()
    alias_graph_item = trans_to_cuda(alias_graph_item).long()
    current_graph_mat = trans_to_cuda(current_graph_mat).long()
    hidden = model(inputs, mask, graph_adj_matrix, alias_graph_item, current_graph_mat, path_num)
    return targets, model.get_score(hidden)


def test(model, test_data):
    model.eval()
    test_loader = DataLoader(test_data, num_workers=4, batch_size=model.batch_size, shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    hit_10, mrr_10 = [], []
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores_10 = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        targets = targets.numpy()
        for score, score_10, target, mask in zip(sub_scores, sub_scores_10, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            hit_10.append(np.isin(target - 1, score_10))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
            if len(np.where(score_10 == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score_10 == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    result.append(np.mean(hit_10) * 100)
    result.append(np.mean(mrr_10) * 100)
    return result


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = DataLoader(train_data, num_workers=4, batch_size=model.batch_size, shuffle=True, pin_memory=True)

    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        logging.info('loss:' + str(loss.item()))
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss.item())
    logging.info('total loss:' + str(total_loss.item()))
    model.scheduler.step()
    logging.info('start predicting:')
    print('start predicting: ', datetime.datetime.now())
    result = test(model, test_data)
    return result


def main():
    if not os.path.exists('log'):
        os.mkdir('log')
    logging.basicConfig(format='%(filename)s [%(asctime)s]  %(message)s', filename='log/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')) + '_log.txt', filemode='a', level=logging.DEBUG)
    logging.info(opt)
    init_seed(2020)
    if opt.dataset == 'diginetica':
        num_node = 43098
    elif opt.dataset == 'yoochoose1_64'or opt.dataset == 'yoochoose1_4':
        num_node = 37484
    else:
        return
    train_data = pickle.load(open('dataset/' + opt.dataset + '/train.txt', 'rb'))
    train_global_graph = pickle.load(open('dataset/' + opt.dataset + '/train_global_graph_' + str(opt.similar_num) + '.pkl', 'rb'))
    test_global_graph = pickle.load(open('dataset/' + opt.dataset + '/test_global_graph_' + str(opt.similar_num) + '.pkl', 'rb'))
    if opt.validation:
        logging.info('split validation')
        train_data, valid_data, train_global_graph, val_global_graph = split_validation(train_data, opt.valid_portion, train_global_graph)
        test_data = pickle.load(open('dataset/' + opt.dataset + '/test.txt', 'rb'))
        test_data = (test_data[0], test_data[1], test_global_graph, test_data[3])
        train_data = (train_data[0], train_data[1], train_global_graph, train_data[2])
    else:
        test_data = pickle.load(open('dataset/' + opt.dataset + '/test.txt', 'rb'))
        train_data = (train_data[0], train_data[1], train_global_graph, train_data[3])
        test_data = (test_data[0], test_data[1], test_global_graph, test_data[3])
    train_data = Data(train_data)
    test_data = Data(test_data)
    model = trans_to_cuda(SIOGNN(num_node, opt))
    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_result_10 = [0, 0]
    best_epoch = [0, 0]
    best_epoch_10 = [0, 0]
    bad_counter = 0

    logging.info('train...')
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        logging.info('epoch:' + str(epoch))
        hit, mrr, hit_10, mrr_10 = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            torch.save(model.state_dict(), 'best_recall_20.mdl')
            torch.save(model.state_dict(), 'mrr_20.mdl')
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            torch.save(model.state_dict(), 'best_mrr_20.mdl')
            torch.save(model.state_dict(), 'recall_20.mdl')
        if hit_10 >= best_result_10[0]:
            best_result_10[0] = hit_10
            best_epoch_10[0] = epoch
            torch.save(model.state_dict(), 'best_recall_10.mdl')
            torch.save(model.state_dict(), 'mrr_10.mdl')
        if mrr_10 >= best_result_10[1]:
            best_result_10[1] = mrr_10
            best_epoch_10[1] = epoch
            torch.save(model.state_dict(), 'best_mrr_10.mdl')
            torch.save(model.state_dict(), 'recall_10.mdl')
        logging.info('Current Result:\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        logging.info('Best Result:\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        logging.info('Current Result:\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit_10, mrr_10))
        print('Current Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit_10, mrr_10))
        logging.info('Best Result:\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_10[0], best_result_10[1], best_epoch_10[0], best_epoch_10[1]))
        print('Best Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_10[0], best_result_10[1], best_epoch_10[0], best_epoch_10[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()

