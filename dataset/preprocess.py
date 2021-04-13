#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np
import torch
import logging
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose/sample')
parser.add_argument('--top_k', type=int, default=32, help='每个session保留top_k个item')
parser.add_argument('--path_len', type=int, default=3)
opt = parser.parse_args()
print(opt)
if not os.path.exists('log'):
    os.mkdir('log')
logging.basicConfig(format='%(filename)s [%(asctime)s]  %(message)s', filename='log/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')) + '_log.txt', filemode='a', level=logging.DEBUG)
logging.info(opt)

dataset = 'train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    if not os.path.exists('yoochoose-clicks-withHeader.dat'):
        with open('yoochoose-clicks.dat', 'r') as f, open('yoochoose-clicks-withHeader.dat', 'w') as fn:
            fn.write('sessionId,timestamp,itemId,category' + '\n')
            for line in f:
                fn.write(line)
    dataset = 'yoochoose-clicks-withHeader.dat' if os.path.exists('yoochoose-clicks-withHeader.dat') else 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['itemId']
        else:
            item = data['itemId'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
print(len(tra_sess))
print(len(tes_sess))
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def get_topk_item(seqs):
    seq_length = len(seqs)
    node = np.unique(seqs)
    node_num = len(node)

    adj_matrix = np.zeros((node_num, node_num))
    indices = []
    seqs = np.array(seqs)
    for i in np.arange(seq_length - 1):
        u = np.where(node == seqs[i])[0][0]
        indices.append(u)
        v = np.where(node == seqs[i + 1])[0][0]
        adj_matrix[u][v] += 1
    if seq_length > 1:
        u = np.where(node == seqs[-1])[0][0]
        indices.append(u)
    adj_matrix = torch.from_numpy(adj_matrix)
    adj_k = [adj_matrix]
    for k in range(0, opt.path_len - 1):
        adj_k.append(torch.matmul(adj_k[k], adj_matrix))
    adj_matrix = sum(adj_k)
    path_nums = torch.sum(adj_matrix, 0)
    if seq_length == 1:
        path = torch.tensor([0])
    else:
        path = torch.tensor([path_nums[j] for j in indices])
    if node_num > opt.top_k:
        top_k_indexes = torch.sort(path.topk(opt.top_k)[1], -1)[0].numpy()
        n = 1
        while len(np.unique(seqs[top_k_indexes])) < opt.top_k:
            top_k_indexes = torch.sort(path.topk(opt.top_k + n)[1], -1)[0].numpy()
            n += 1
        seqs = seqs[top_k_indexes]
        path = path[top_k_indexes]
    return list(seqs), list(path.numpy())


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    all_len = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            all_len.append(len(np.unique(seq[:-i])))
            out_dates += [date]
            ids += [id]
    print(str(max(all_len)))
    assert opt.top_k <= max(all_len)
    assert opt.top_k > 1
    cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(cores) as pool:
        result = pool.map(get_topk_item, out_seqs)
    out_seqs, path = [], []
    for r in result:
        out_seqs.append(r[0])
        path.append(r[1])
    return out_seqs, out_dates, labs, ids, path


tr_seqs, tr_dates, tr_labs, tr_ids, tr_path = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids, te_path = process_seqs(tes_seqs, tes_dates)


item = []
index = 0
for i in tr_seqs:
    item += i
    index += 1
    if index % 1000 == 0:
        item = list(np.unique(item))
for i in tr_labs:
    item.append(i)
    index += 1
    if index % 1000 == 0:
        item = list(np.unique(item))
item = list(np.unique(item))
print(str(len(item)))

data = [(te_seqs[i], te_labs[i], i) for i in range(len(te_seqs))]


def filter_node(data_):
    indice = -1
    seq = []
    for j in data_[0]:
        if j in item:
            seq.append(j)
    if len(seq) > 0 and data_[1] in item:
        indice = data_[2]
    return seq, indice


cores = multiprocessing.cpu_count()
with multiprocessing.Pool(cores) as pool:
    seq_indices = pool.map(filter_node, data)
indices = [s[1] for s in seq_indices]
te_ids_ = []
te_labs_ = []
te_dates_ = []
te_seqs_ = []
te_path_ = []
for i in indices:
    if i == -1:
        continue
    te_seqs_.append(seq_indices[i][0])
    te_ids_.append(te_ids[i])
    te_labs_.append(te_labs[i])
    te_dates_.append(te_dates[i])
    te_path_.append(te_path[i])
te_ids = te_ids_
te_labs = te_labs_
te_dates = te_dates_
te_seqs = te_seqs_
te_path = te_path_
tra = (tr_seqs, tr_labs, tr_ids, tr_path)
tes = (te_seqs, te_labs, te_ids, te_path)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    print('item:' + str(len(item)))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    # split64 = int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    # tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    # seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]
    tra64 = (tr_seqs[-split64:], tr_labs[-split64:], tr_ids[-split64:], tr_path[-split64:])
    tra4 = (tr_seqs[-split4:], tr_labs[-split4:], tr_ids[-split4:], tr_path[-split4:])
    # seq64 = tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    tes = (te_seqs, te_labs, te_ids, te_path)
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))
    print('item:' + str(len(item)))
print('Done.')
