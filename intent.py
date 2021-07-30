#!/usr/bin/python3
# -*-coding:UTF-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalIntent(nn.Module):
    def __init__(self, dim, alpha, dropout_p=0.):
        super(LocalIntent, self).__init__()
        self.hidden_dim = dim
        self.alpha = alpha
        self.dropout_p = dropout_p

        self.position_embedding = nn.Embedding(200, self.hidden_dim)
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_dim * 2, self.hidden_dim))
        # self.w2 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        # self.w3 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        # self.w4 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.q = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_dim))
        # self.bias2 = nn.Parameter(torch.Tensor(self.hidden_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, hidden, input_mask, path):
        h = hidden
        batch_size = h.shape[0]
        pos_emb = self.position_embedding.weight[:hidden.shape[1]]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        # s_m = torch.sum(torch.softmax(path, -1).unsqueeze(-1).repeat(1, 1, self.hidden_dim) * hidden, 1)
        # s_m = torch.matmul(s_m.unsqueeze(1).repeat(1, hidden.shape[1], 1), self.w2)  #
        t = self.leaky_relu(torch.matmul(torch.cat([pos_emb, hidden], -1), self.w1) + self.bias)
        # h_n = t[:, 0]
        # h_n = h_n.unsqueeze(-2).repeat(1, hidden.shape[1], 1)
        # t = self.leaky_relu(torch.matmul(t, self.w3) + torch.matmul(h_n, self.w4) + s_m + self.bias2)
        t = F.dropout(t, self.dropout_p, training=self.training)
        t = torch.matmul(t, self.q).squeeze(-1)
        mask = -9e15 * torch.ones_like(t)
        alpha = torch.where(input_mask.eq(1), t, mask)
        alpha = torch.softmax(alpha, -1)
        session_intent = torch.sum(alpha.unsqueeze(-1).repeat(1, 1, self.hidden_dim) * hidden, 1)
        return session_intent


class GlobalIntent(nn.Module):
    def __init__(self, dim, alpha, dropout_p=0.):
        super(GlobalIntent, self).__init__()
        self.hidden_dim = dim
        self.alpha = alpha
        self.dropout_p = dropout_p
        self.w_s = nn.Parameter(torch.Tensor(2 * self.hidden_dim, self.hidden_dim))
        self.w_c = nn.Parameter(torch.Tensor(2 * self.hidden_dim, self.hidden_dim))
        self.w = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.q = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, targets, neighbors, neighbors_weight, neighbors_mask):
        mask_neighbor = neighbors * neighbors_mask.unsqueeze(-1)
        h_i = torch.pow(torch.sum(mask_neighbor, -2), 2) - torch.pow(mask_neighbor, 2).sum(dim=-2)
        neighbors_mask = 2 * neighbors_mask.sum(-1)
        neighbors_mask[neighbors_mask <= 0] = 1
        h_i = h_i / neighbors_mask.unsqueeze(-1)
        h_s = torch.sum(neighbors * torch.softmax(neighbors_weight, dim=-1).unsqueeze(dim=-1), dim=-2)
        h_n = torch.matmul(torch.cat([h_i, h_s], -1), self.w_s)
        h_n = F.dropout(h_n, self.dropout_p, training=self.training)
        targets = torch.matmul(torch.cat([targets, h_n], dim=-1), self.w_c)
        targets = self.leaky_relu(targets)
        return targets


