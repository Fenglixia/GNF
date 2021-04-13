#!/usr/bin/python3
# -*-coding:UTF-8-*-
import pickle
import argparse
import numpy as np
import time
from tqdm import tqdm
import os
import collections
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying')
parser.add_argument('--similar_num', type=int, default=12, help='相似session的个数')
parser.add_argument('--threshold', default=0.5, help='session之间相似度阈值')
opt = parser.parse_args()

dataset = opt.dataset
cores = multiprocessing.cpu_count()

train_data = pickle.load(open('dataset/' + dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('dataset/' + dataset + '/test.txt', 'rb'))

train_seqs = train_data[0]
test_seq = test_data[0]

train_label = train_data[1]
test_label = test_data[1]

train_id = train_data[2]
test_id = list(np.asanyarray(test_data[2]) + train_id[-1] + 1)
all_data = train_seqs + test_seq
all_id = train_id + test_id
assert len(all_data) == len(all_id)


def collect(lst):
    return dict(collections.Counter(lst))


def get_all_session_edge(all_session):
    all_edge = []
    target_edge = []
    edge_to_session = dict()
    for i in tqdm(range(len(all_session))):
        s_edge = []
        t_edge = []
        for j in range(len(all_session[i])):
            if j >= 1:
                s_edge.append((all_session[i][j - 1], all_session[i][j]))
                if (all_session[i][j - 1], all_session[i][j]) in edge_to_session.keys():
                    edge_to_session[(all_session[i][j - 1], all_session[i][j])] += [i]
                else:
                    edge_to_session[(all_session[i][j - 1], all_session[i][j])] = [i]
            for k in range(j + 1, len(all_session[i])):
                t_edge.append((all_session[i][j], all_session[i][k]))
                t_edge.append((all_session[i][k], all_session[i][j]))
        t_edge = list(set(t_edge))
        target_edge.append(t_edge)
        all_edge.append(s_edge)
    return all_edge, target_edge, edge_to_session


all_session_edge, all_target_edge, edge_to_sessions = get_all_session_edge(all_data)


def calc_similarity(target_session, all_session, sess_index, ind, threshold=0.5):
    target_len = len(target_session)
    if target_len < 2:
        return []
    target_edge = all_target_edge[ind]
    s = []
    for e in target_edge:
        if e in edge_to_sessions.keys():
            s += edge_to_sessions[e]
    s = list(np.unique(s))

    neighbors = []
    for index in range(len(s)):
        if all_id[s[index]] >= sess_index:
            break
        if len(all_session[s[index]]) < 2:
            continue
        s_edge = all_session_edge[s[index]]
        count = len(set(target_edge).intersection(set(s_edge)))
        similarity = count/len(all_session[s[index]])
        if similarity >= threshold:
            neighbors.append([s[index], similarity])
    return neighbors


def get_sess_neighbor(session_and_index):
    possible_neighbors = calc_similarity(session_and_index[0], all_data, all_id[session_and_index[1]], session_and_index[1], opt.threshold)
    possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
    if len(possible_neighbors) > 0:
        possible_neighbors = list(np.asarray(possible_neighbors, dtype=np.int32)[:, 0])

    if len(possible_neighbors) > opt.similar_num:
        return possible_neighbors[:opt.similar_num]
    elif len(possible_neighbors) > 0:
        return possible_neighbors
    else:
        return 0


def get_neigh_sess(all_session):
    session_and_index = [(s, i) for i, s in enumerate(all_session)]
    with multiprocessing.Pool(cores) as pool:
        all_sess_neigh = pool.map(get_sess_neighbor, session_and_index)
    return all_sess_neigh


def start_build_graph(data):
    edge = data[0]
    neighbor = data[1]
    current_g = data[2]
    if neighbor != 0:
        for neigh_index in neighbor:
            edge += all_session_edge[neigh_index]
            current_g += all_data[neigh_index]

    items_g = list(np.unique(current_g))
    edge += [(item, item) for item in items_g]
    g = collect(edge)
    return g


def build_global_graph(all_session):
    if os.path.exists('dataset/' + dataset + '/all_session_neighbor_' + str(opt.similar_num) + '.txt'):
        all_session_neighbor = pickle.load(open('dataset/' + dataset + '/all_session_neighbor_' + str(opt.similar_num) + '.txt', 'rb'))
    else:
        all_session_neighbor = get_neigh_sess(all_session)
        pickle.dump(all_session_neighbor, open('dataset/' + dataset + '/all_session_neighbor_' + str(opt.similar_num) + '.txt', 'wb'))
    assert len(all_session) == len(all_session_neighbor)

    edge_and_neighbor = [(all_session_edge[i], all_session_neighbor[i], all_session[i]) for i in range(len(all_session))]
    with multiprocessing.Pool(cores) as pool:
        all_graph = pool.map(start_build_graph, edge_and_neighbor)
    return all_graph


start = time.time()
global_graph = build_global_graph(all_data)
pickle.dump(global_graph[:len(train_seqs)], open('dataset/' + dataset + '/train_global_graph_' + str(opt.similar_num) + '.pkl', 'wb'))
pickle.dump(global_graph[len(train_seqs):], open('dataset/' + dataset + '/test_global_graph_' + str(opt.similar_num) + '.pkl', 'wb'))
end = time.time()
print(end-start)
assert len(all_data) == len(global_graph)
assert len(test_seq) == len(global_graph[len(train_seqs):])

