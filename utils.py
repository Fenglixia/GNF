import numpy as np
import torch
from torch.utils.data import Dataset


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def split_validation(train_set, proportion, train_global_graph):
    train_set_data, train_set_label, tr_ids, train_set_path = train_set
    num = len(train_set_data)
    index = np.arange(num, dtype='int32')
    np.random.shuffle(index)
    train_len = int(np.round(num * (1. - proportion)))
    val_set_data = [train_set_data[s] for s in index[train_len:]]
    val_set_label = [train_set_label[s] for s in index[train_len:]]
    val_set_path = [train_set_path[s] for s in index[train_len:]]
    val_graph = [train_global_graph[s] for s in index[train_len:]]
    train_set_data = [train_set_data[s] for s in index[:train_len]]
    train_set_label = [train_set_label[s] for s in index[:train_len]]
    train_set_path = [train_set_path[s] for s in index[:train_len]]
    train_graph = [train_global_graph[s] for s in index[:train_len]]
    return (train_set_data, train_set_label, train_set_path), (val_set_data, val_set_label, val_set_path), train_graph, val_graph


class Data(Dataset):
    def __init__(self, data, train_len=None):
        self.train_len = train_len
        inputs, mask, max_len, max_graph_len, graph_item, global_graph_mat, graph_mask, path = self.handle_data(data[0], data[2], data[3])   # data[0]为所有的划分后的session
        self.inputs = np.asarray(inputs)
        self.path = path
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.global_graph = data[2]
        self.length = len(data[0])
        self.max_len = max_len
        self.max_graph_len = max_graph_len
        self.graph_item = graph_item
        self.global_graph_mat = global_graph_mat
        self.graph_mask = graph_mask

    def __getitem__(self, index):
        u_input, mask, target, graph, graph_item, current_graph_mat, current_graph_mask, path_num = self.inputs[index], self.mask[index], self.targets[index], self.global_graph[index], self.graph_item[index], self.global_graph_mat[index], self.graph_mask[index], self.path[index]
        max_num_node = self.max_graph_len
        node = np.unique(current_graph_mat)
        graph_adj_matrix = np.zeros((max_num_node, max_num_node))
        for edge in graph.keys():
            index1 = np.where(node == edge[0])[0][0]
            index2 = np.where(node == edge[1])[0][0]
            if not graph_adj_matrix[index1][index2]:
                graph_adj_matrix[index1][index2] = graph[edge]
                graph_adj_matrix[index2][index1] = graph[edge]
            else:
                graph_adj_matrix[index1][index2] += graph[edge]
                graph_adj_matrix[index2][index1] += graph[edge]
        alias_graph_item = [np.where(node == i)[0][0] for i in current_graph_mat]
        return [torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), torch.tensor(graph_adj_matrix), torch.tensor(alias_graph_item), torch.tensor(current_graph_mat), torch.tensor(current_graph_mask), torch.tensor(path_num)]

    def __len__(self):
        return self.length

    def handle_data(self, input_data, graph, path):
        data_len = [len(sub_data) for sub_data in input_data]
        graph_len = []
        graph_item = []
        for g in graph:
            item = list(g.keys())
            item = list(np.unique(item))
            graph_len.append(len(item))
            graph_item.append(item)
        max_graph_len = max(graph_len)
        max_len = max(data_len) if self.train_len is None else self.train_len
        data = [list(reversed(sub_data)) + [0] * (max_len - length) if length < max_len else list(reversed(sub_data[-max_len:])) for sub_data, length in zip(input_data, data_len)]
        path_num = [list(reversed(sub_path)) + [-9e15] * (max_len - length) if length < max_len else list(reversed(sub_path[-max_len:])) for sub_path, length in zip(path, data_len)]
        mask = [[1] * length + [0] * (max_len - length) if length < max_len else [1] * max_len for length in data_len]
        global_graph = [sub_graph + [0] * (max_graph_len - length) for sub_graph, length in zip(graph_item, graph_len)]
        graph_mask = [[1] * length + [0] * (max_graph_len - length) for length in graph_len]
        return data, mask, max_len, max_graph_len, graph_item, global_graph, graph_mask, path_num
