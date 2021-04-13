#!/usr/bin/python3
# -*-coding:UTF-8-*-
import math
from torch import nn
from GNF.intent import LocalIntent, GlobalIntent
import torch.nn.functional as F
from GNF.utils import *


class SIOGNN(nn.Module):
    def __init__(self, num_node, opt):
        super(SIOGNN, self).__init__()
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.hidden_dim = opt.hidden_dim
        self.alpha = opt.alpha
        self.layer_num = opt.layer_num
        self.global_dropout_p = opt.global_dropout_p
        self.local_dropout_p = opt.local_dropout_p
        self.neighbor_num = opt.neighbor_num
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.step_size = opt.lr_step
        self.gamma = opt.lr_dc

        self.w_3 = nn.Parameter(torch.Tensor(2 * self.hidden_dim, self.hidden_dim))
        self.position_embedding = nn.Embedding(200, self.hidden_dim)
        self.embedding = nn.Embedding(self.num_node, self.hidden_dim)
        self.b_1 = nn.Parameter(torch.Tensor(self.hidden_dim))
        self.q = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.w_g1 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.w_g2 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_2 = nn.Parameter(torch.Tensor(self.hidden_dim))
        self.w_h = nn.Parameter(torch.Tensor(2 * self.hidden_dim, self.hidden_dim))
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.local_intent = LocalIntent(self.hidden_dim, self.alpha)
        self.global_intent = []
        for i in range(self.layer_num):
            agg = GlobalIntent(self.hidden_dim, self.alpha, self.global_dropout_p)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_intent.append(agg)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self.init_parameters()

    def forward(self, data, input_mask, graph_adj_matrix, alias_graph_item, current_graph_mat, path):
        batch_size = data.shape[0]
        seqs_len = data.shape[1]
        hidden = self.embedding(data)
        s_l = self.local_intent(hidden, input_mask, path)
        neighbors = [data]
        neighbors_weight = []
        neighbors_mask = []
        support_size = seqs_len
        for layer in range(self.layer_num):
            item_sample_i, weight_sample_i, neighbors_mask_i = self.get_neighbors(neighbors[-1], graph_adj_matrix, alias_graph_item, current_graph_mat)
            support_size *= self.neighbor_num
            neighbors.append(item_sample_i.view(batch_size, support_size))
            neighbors_weight.append(weight_sample_i.view(batch_size, support_size))
            neighbors_mask.append(neighbors_mask_i.view(batch_size, support_size))
        entity_vectors = [self.embedding(i.to(torch.int64)) for i in neighbors]
        weight_vectors = neighbors_weight
        mask_vectors = neighbors_mask
        for layer in range(self.layer_num):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.neighbor_num, self.hidden_dim]
            for hop in range(self.layer_num - layer):
                aggregator = self.global_intent[layer]
                vector = aggregator(targets=entity_vectors[hop],
                                    neighbors=entity_vectors[hop+1].view(shape),
                                    neighbors_weight=weight_vectors[hop].view(batch_size, -1, self.neighbor_num),
                                    neighbors_mask=mask_vectors[hop].view(batch_size, -1, self.neighbor_num),
                                    )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        h_global = entity_vectors[0].view(batch_size, seqs_len, self.hidden_dim)
        pos_emb = self.position_embedding.weight[:hidden.shape[1]]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        s = self.leaky_relu(torch.matmul(torch.cat([pos_emb, h_global], -1), self.w_3) + self.b_1)
        beta = torch.matmul(s, self.w_g1) + torch.matmul(s_l.unsqueeze(1).repeat(1, s.shape[1], 1), self.w_g2) + self.b_2
        beta = torch.matmul(self.leaky_relu(beta), self.q).squeeze(-1)
        beta = torch.softmax(beta, dim=-1)
        s_g = torch.sum(s * beta.unsqueeze(-1), dim=-2)
        s_l = F.dropout(s_l, self.local_dropout_p, training=self.training)
        s_g = F.dropout(s_g, self.global_dropout_p, training=self.training)
        output = torch.matmul(torch.cat([s_l, s_g], dim=-1), self.w_h)
        output = self.leaky_relu(output)
        return output

    def get_score(self, session_intent):
        b = self.embedding.weight[1:]
        scores = torch.matmul(session_intent, b.transpose(1, 0))
        return scores

    def init_parameters(self):
        stand = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stand, stand)

    def get_neighbors(self, data, graph_adj_matrix, alias_graph_item, current_graph_mat):
        batch_size = data.shape[0]
        seqs_len = data.shape[1]
        neighbor = trans_to_cuda(torch.zeros((batch_size, seqs_len, self.neighbor_num)))
        neighbor_weight = trans_to_cuda(-9e15*torch.ones((batch_size, seqs_len, self.neighbor_num)))

        for i in range(batch_size):
            for j in range(seqs_len):
                if data[i][j] == 0:
                    break
                ind = torch.where(current_graph_mat[i] == data[i][j])[0][0]
                item_neighbor = graph_adj_matrix[i][alias_graph_item[i][ind]]
                i_mask = item_neighbor > 0
                items = torch.tensor([current_graph_mat[i][torch.where(alias_graph_item[i] == index)[0]] for index in torch.where(i_mask)[0]])
                items = trans_to_cuda(items)
                weights = item_neighbor[i_mask]

                if items.size(0) <= self.neighbor_num:
                    neighbor[i][j][:items.size(0)] = items
                    neighbor_weight[i][j][:weights.size(0)] = weights
                else:
                    top = weights.topk(self.neighbor_num)
                    neighbor[i][j] = items[top[1]]
                    neighbor_weight[i][j] = top[0]
        neighbors_mask = neighbor > 0
        return neighbor, neighbor_weight, neighbors_mask
