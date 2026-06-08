import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Conv1D(nn.Module):
    def __init__(self, cnn_method: str, in_channels: int, cnn_kernel_num: int, cnn_window_size: int):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=cnn_window_size, padding=(cnn_window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=5, padding=2)
        else:
            assert cnn_kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=2, padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=4, padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=5, padding=2)
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return F.relu(self.conv(feature)) # [batch_size, cnn_kernel_num, length]
        elif self.cnn_method == 'group3':
            return F.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return F.relu(torch.cat([self.conv1(feature), \
                                     self.conv2(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv3(feature), \
                                     self.conv4(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv5(feature)], dim=1))


class Conv2D_Pool(nn.Module):
    def __init__(self, cnn_method: str, in_channels: int, cnn_kernel_num: int, cnn_window_size: int, last_channel_num: int):
        super(Conv2D_Pool, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group4']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        self.last_channel_num = last_channel_num
        self.cnn_window_size = cnn_window_size
        if self.cnn_method == 'naive':
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=[cnn_window_size, last_channel_num], padding=[(cnn_window_size - 1) // 2, 0])
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[1, last_channel_num], padding=[0, 0])
            self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[2, last_channel_num], padding=[0, 0])
            self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[3, last_channel_num], padding=[1, 0])
        else:
            assert cnn_kernel_num % 4 == 0
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[1, last_channel_num], padding=[0, 0])
            self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[2, last_channel_num], padding=[0, 0])
            self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[3, last_channel_num], padding=[1, 0])
            self.conv4 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[4, last_channel_num], padding=[1, 0])
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num]
    def forward(self, feature):
        length = feature.size(2)
        if self.cnn_method == 'naive':
            conv_relu = F.relu(self.conv(feature), inplace=True)                                                     # [batch_size, cnn_kernel_num, length]
            conv_relu_pool, _ = torch.max(conv_relu[:, :, :length - self.cnn_window_size + 1], dim=2, keepdim=False) # [batch_size, cnn_kernel_num]
            return conv_relu_pool
        elif self.cnn_method == 'group3':
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1, self.last_channel_num], device=self.device)
            conv1_relu = F.relu(self.conv1(feature), inplace=True)
            conv1_relu_pool, _ = torch.max(conv1_relu, dim=2, keepdim=False)
            conv2_relu = F.relu(self.conv2(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv2_relu_pool, _ = torch.max(conv2_relu[:, :, :length - 1], dim=2, keepdim=False)
            conv3_relu = F.relu(self.conv3(feature), inplace=True)
            conv3_relu_pool, _ = torch.max(conv3_relu[:, :, :length - 2], dim=2, keepdim=False)
            return torch.cat([conv1_relu_pool, conv2_relu_pool, conv3_relu_pool], dim=1)
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1, self.last_channel_num], device=self.device)
            conv1_relu = F.relu(self.conv1(feature), inplace=True)
            conv1_relu_pool, _ = torch.max(conv1_relu, dim=2, keepdim=False)
            conv2_relu = F.relu(self.conv2(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv2_relu_pool, _ = torch.max(conv2_relu[:, :, :length - 1], dim=2, keepdim=False)
            conv3_relu = F.relu(self.conv3(feature), inplace=True)
            conv3_relu_pool, _ = torch.max(conv3_relu[:, :, :length - 2], dim=2, keepdim=False)
            conv4_relu = F.relu(self.conv4(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv4_relu_pool, _ = torch.max(conv4_relu[:, :, :length - 3], dim=2, keepdim=False)
            return torch.cat([conv1_relu_pool, conv2_relu_pool, conv3_relu_pool, conv4_relu_pool], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_Q = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_K = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_V = nn.Linear(d_model, self.h*self.d_v, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, h * d_v]
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])                                           # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])                                           # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])                                           # [batch_size, len_k, h, d_v]
        Q = Q.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_q, self.d_k])                   # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_k])                   # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_v])                   # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar                                  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.h]).view([batch_size * self.h, 1, self.len_k]).repeat([1, self.len_q, 1]) # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(_mask == 0, -1e9), dim=2)                                              # [batch_size * h, len_q, len_k]
        else:
            alpha = F.softmax(A, dim=2)                                                                            # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])                                 # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view([batch_size, self.len_q, self.out_dim])                  # [batch_size, len_q, h * d_v]
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out


class ScaledDotProduct_CandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(ScaledDotProduct_CandidateAttention, self).__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.K(feature), self.Q(query).unsqueeze(dim=2)).squeeze(dim=2) / self.attention_scalar # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                          # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                       # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                       # [batch_size, feature_dim]
        return out


class CandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(CandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(feature_dim, attention_dim, bias=False)
        self.query_affine = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_affine = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = self.attention_affine(torch.tanh(self.feature_affine(feature) + self.query_affine(query).unsqueeze(dim=1))).squeeze(dim=2) # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                                                   # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                                                # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                                                # [batch_size, feature_dim]
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False, layer_norm=False):
        super(GCNLayer, self).__init__()
        self.residual = residual
        self.layer_norm = layer_norm
        if self.residual and in_dim != out_dim:
            raise Exception('To facilitate residual connection, in_dim must equal to out_dim')
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        if self.layer_norm:
            self.layer_normalization = nn.LayerNorm(normalized_shape=[out_dim])

    def initialize(self):
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W.bias)

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = self.W(torch.bmm(graph, feature)) # [batch_size, node_num, feature_num]
        if self.layer_norm:
            out = self.layer_normalization(out) # [batch_size, node_num, feature_num]
        out = F.relu(out)                       # [batch_size, node_num, feature_num]
        if self.residual:
            out = out + feature                 # [batch_size, node_num, feature_num]
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, num_layers=1, dropout=0.1, residual=False, layer_norm=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = []
        if self.num_layers == 1:
            self.gcn_layers.append(GCNLayer(in_dim, out_dim, residual=residual, layer_norm=layer_norm))
        else:
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.gcn_layers.append(GCNLayer(in_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            for i in range(1, self.num_layers - 1):
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            self.gcn_layers.append(GCNLayer(hidden_dim, out_dim, residual=residual, layer_norm=layer_norm))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

    def initialize(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gcn_layers[i](out, graph))
        out = self.gcn_layers[self.num_layers - 1](out, graph)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, dropout=0.2):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, head_dim * num_heads)
        self.W_K = nn.Linear(d_model, head_dim * num_heads)
        self.W_V = nn.Linear(d_model, head_dim * num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape

        # Linear projection
        q = self.W_Q(Q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(K).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(V).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = torch.exp(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            scores = scores * mask

        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, v)  # [B, num_heads, seq_len, head_dim]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim)

        return self.dropout(context)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PENR_AdditiveAttention(nn.Module):
    """
    Additive attention for PENR (used for both word-level and view-level attention)
    Computes: α_i = q^T tanh(V x_i + v)
    Then applies softmax and weighted sum

    Used for:
    - Word-level attention in Title encoder (Equations 8-9): pools words → r^t ∈ R^300
    - Word-level attention in Abstract encoder (Equations 8-9): pools words → r^a ∈ R^300
    - View-level attention (Section 1.4): pools 3 views → r ∈ R^300
    """
    def __init__(self, feature_dim: int, query_dim: int):
        super(PENR_AdditiveAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query_dim = query_dim

        # V: projection matrix [query_dim, feature_dim]
        self.V = nn.Linear(feature_dim, query_dim, bias=True)
        # q: attention query vector [query_dim]
        self.q = nn.Parameter(torch.zeros(query_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.V.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.V.bias)
        nn.init.xavier_uniform_(self.q.unsqueeze(0))

    # Input
    # feature : [batch_size, seq_len, feature_dim]
    # mask    : [batch_size, seq_len] (optional)
    # Output
    # out     : [batch_size, feature_dim] (weighted sum of features)
    def forward(self, feature, mask=None):
        # Compute attention scores: α_i = q^T tanh(V x_i + v)
        projected = torch.tanh(self.V(feature))  # [batch_size, seq_len, query_dim]
        scores = torch.matmul(projected, self.q)  # [batch_size, seq_len]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        alpha = F.softmax(scores, dim=1)  # [batch_size, seq_len]

        # Weighted sum
        out = torch.bmm(alpha.unsqueeze(1), feature).squeeze(1)  # [batch_size, feature_dim]
        return out