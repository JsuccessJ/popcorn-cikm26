import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from layers import MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention, GCN
try:
    from torch_geometric.nn import GraphSAGE
except ImportError:
    pass
from newsEncoders import NewsEncoder, HDC
from torch_scatter import scatter_sum, scatter_softmax # need to be installed by following `https://pytorch-scatter.readthedocs.io/en/latest`


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.device = torch.device('cuda')
        self.auxiliary_loss = None

    # Input
    # user_title_text               : [batch_size, max_history_num, max_title_length]
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_title_entity             : [batch_size, max_history_num, max_title_length]
    # user_content_text             : [batch_size, max_history_num, max_content_length]
    # user_content_mask             : [batch_size, max_history_num, max_content_length]
    # user_content_entity           : [batch_size, max_history_num, max_content_length]
    # user_category                 : [batch_size, max_history_num]
    # user_subCategory              : [batch_size, max_history_num]
    # user_history_mask             : [batch_size, max_history_num]
    # user_history_graph            : [batch_size, max_history_num, max_history_num]
    # user_history_category_mask    : [batch_size, category_num]
    # user_history_category_indices : [batch_size, max_history_num]
    # user_embedding                : [batch_size, user_embedding]
    # candidate_news_representation : [batch_size, news_num, news_embedding_dim]
    # Output
    # user_representation           : [batch_size, news_embedding_dim]
    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation, candidate_category=None, candidate_subCategory=None):
        raise Exception('Function forward must be implemented at sub-class')


class SUE(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(SUE, self).__init__(news_encoder, config)
        self.attention_dim = max(config.attention_dim, self.news_embedding_dim // 4)
        self.proxy_node_embedding = nn.Parameter(torch.zeros([config.category_num, self.news_embedding_dim]))
        self.gcn = GCN(in_dim=self.news_embedding_dim, out_dim=self.news_embedding_dim, hidden_dim=self.news_embedding_dim, num_layers=config.gcn_layer_num, dropout=config.dropout_rate / 2, residual=not config.no_gcn_residual, layer_norm=config.gcn_layer_norm)
        self.intraCluster_K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=False)
        self.intraCluster_Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.clusterFeatureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.interClusterAttention = ScaledDotProduct_CandidateAttention(self.news_embedding_dim, self.news_embedding_dim, self.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.category_num = config.category_num + 1 # extra one category index for padding news
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))

    def initialize(self):
        self.gcn.initialize()
        nn.init.zeros_(self.proxy_node_embedding)
        nn.init.xavier_uniform_(self.intraCluster_K.weight)
        nn.init.xavier_uniform_(self.intraCluster_Q.weight)
        nn.init.zeros_(self.intraCluster_Q.bias)
        nn.init.xavier_uniform_(self.clusterFeatureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.clusterFeatureAffine.bias)
        self.interClusterAttention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        user_history_category_mask[:, -1] = 1
        user_history_category_mask = user_history_category_mask.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous()                                  # [batch_size, news_num, category_num]
        user_history_category_indices = user_history_category_indices.unsqueeze(dim=1).expand(-1, news_num, -1)                                         # [batch_size, news_num, max_history_num]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                          # [batch_size, max_history_num, news_embedding_dim]
        # 1. GCN
        history_embedding = torch.cat([history_embedding, self.dropout_(self.proxy_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1) # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = self.gcn(history_embedding, user_history_graph) + history_embedding                                                               # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = gcn_feature[:, :self.max_history_num, :] # [64, 67, 900] -> [64, 50, 900] remove proxy nodes, keep only news nodes                                                                                 # [batch_size, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1, -1) # [64, 50, 900] -> [64, 1, 50, 900] -> [64, 5, 50, 900] use the same GCN features for each candidate news                                                                         # [batch_size, news_num, max_history_num, news_embedding_dim]
        # 2. Intra-cluster attention
        K = self.intraCluster_K(gcn_feature).view([batch_news_num, self.max_history_num, self.attention_dim])                                           # [batch_size * news_num, max_history_num, attention_dim]
        Q = self.intraCluster_Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])                                            # [batch_size * news_num, attention_dim, 1]
        a = torch.bmm(K, Q).view([batch_size, news_num, self.max_history_num]) / self.attention_scalar                                                  # [batch_size, news_num, max_history_num]
        alpha_intra = scatter_softmax(a, user_history_category_indices, 2).unsqueeze(dim=3) # a shape: [batch_size, news_num, max_history_num]; apply softmax only within the same category among user history news                                                             # [batch_size, news_num, max_history_num, 1]
        intra_cluster_feature = scatter_sum(alpha_intra * gcn_feature, user_history_category_indices, dim=2, dim_size=self.category_num)                # [batch_size, news_num, category_num, news_embedding_dim]
        # perform nonlinear transformation on intra-cluster features
        intra_cluster_feature = self.dropout(F.relu(self.clusterFeatureAffine(intra_cluster_feature), inplace=True) + intra_cluster_feature)            # [batch_size, news_num, category_num, news_embedding_dim]
        # 3. Inter-cluster attention
        inter_cluster_feature = self.interClusterAttention(
            intra_cluster_feature.view([batch_news_num, self.category_num, self.news_embedding_dim]), # history features aggregated per category, used as K and V
            candidate_news_representation.view([batch_news_num, self.news_embedding_dim]), # candidate news as Query
            mask=user_history_category_mask.view([batch_news_num, self.category_num]) # indicates whether the category exists
        ).view([batch_size, news_num, self.news_embedding_dim])                                                                                         # [batch_size, news_num, news_embedding_dim]
        return inter_cluster_feature


class LSTUR(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(LSTUR, self).__init__(news_encoder, config)
        self.masking_probability = 1.0 - config.long_term_masking_probability
        self.gru = nn.GRU(self.news_embedding_dim, self.news_embedding_dim, batch_first=True)

    def initialize(self):
        for parameter in self.gru.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        user_history_num = user_history_mask.sum(dim=1, keepdim=False).long()                                                                           # [batch_size]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                          # [batch_size, max_history_num, news_embedding_dim]
        sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)                                                         # [batch_size]
        _, desorted_indices = torch.sort(sorted_indices, descending=False)                                                                              # [batch_size]
        nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)
        if nonzero_indices.size(0) == 0:
            user_representation = user_embedding.unsqueeze(dim=1).expand(-1, news_num, -1)                                                              # [batch_size, news_num, news_embedding_dim]  
            return user_representation
        index = nonzero_indices[-1]
        if index + 1 == batch_size:
            sorted_user_embedding = user_embedding.index_select(0, sorted_indices)                                                                      # [batch_size, user_embedding_dim]
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(torch.empty([batch_size, 1], device=self.device).fill_(self.masking_probability))              # [batch_size, user_embedding_dim]
            sorted_history_embedding = history_embedding.index_select(0, sorted_indices)                                                                # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num.cpu(), batch_first=True)           # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))                                                    # [1, batch_size, news_embedding_dim]
            user_representation = h.squeeze(dim=0).index_select(0, desorted_indices)                                                                    # [batch_size, news_embedding_dim]
        else:
            non_empty_indices = sorted_indices[:index+1]
            empty_indices = sorted_indices[index+1:]
            sorted_user_embedding = user_embedding.index_select(0, non_empty_indices)                                                                   # [batch_size, user_embedding_dim]
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(torch.empty([index + 1, 1], device=self.device).fill_(self.masking_probability))               # [batch_size, user_embedding_dim]
            sorted_history_embedding = history_embedding.index_select(0, non_empty_indices)                                                             # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num[:index+1].cpu(), batch_first=True) # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))                                                    # [1, batch_size, news_embedding_dim]
            user_representation = torch.cat([h.squeeze(dim=0), user_embedding.index_select(0, empty_indices)], dim=0).index_select(0, desorted_indices) # [batch_size, news_embedding_dim]
        user_representation = user_representation.unsqueeze(dim=1).expand(-1, news_num, -1)                                                             # [batch_size, news_num, news_embedding_dim]
        return user_representation


class MHSA(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(MHSA, self).__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.affine = nn.Linear(config.head_num*config.head_dim, self.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                  # [batch_size, max_history_num, news_embedding_dim]
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask) # [batch_size, max_history_num, head_num * head_dim]
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)               # [batch_size, max_history_num, news_embedding_dim]
        user_representation = self.attention(h).unsqueeze(dim=1).repeat(1, news_num, 1)                         # [batch_size, news_num, news_embedding_dim]
        return user_representation


class ATT(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(ATT, self).__init__(news_encoder, config)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)  # number of candidate news, e.g. candidate_news_representation.shape = [64, 5, 400] -> news_num = 5
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)            # [batch_size, max_history_num, news_embedding_dim]
        user_representation = self.attention(history_embedding).unsqueeze(dim=1).expand(-1, news_num, -1) # [batch_size, news_num, news_embedding_dim]
        return user_representation


class CATT(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(CATT, self).__init__(news_encoder, config)
        self.affine1 = nn.Linear(self.news_embedding_dim * 2, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=True)
        self.max_history_num = config.max_history_num

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                  # [batch_size, max_history_num, news_embedding_dim]
        user_history_mask = user_history_mask.unsqueeze(dim=1).expand(-1, news_num, -1)                                         # [batch_size, news_num, max_history_num]
        candidate_news_representation = candidate_news_representation.unsqueeze(dim=2).expand(-1, -1, self.max_history_num, -1) # [batch_size, news_num, max_history_num, news_embedding_dim]
        history_embedding = history_embedding.unsqueeze(dim=1).expand(-1, news_num, -1, -1)                                     # [batch_size, news_num, max_history_num, news_embedding_dim]
        concat_embeddings = torch.cat([candidate_news_representation, history_embedding], dim=3)                                # [batch_size, news_num, max_history_num, news_embedding_dim * 2]
        hidden = F.relu(self.affine1(concat_embeddings), inplace=True)                                                          # [batch_size, news_num, max_history_num, attention_dim]
        a = self.affine2(hidden).squeeze(dim=3)                                                                                 # [batch_size, news_num, max_history_num]
        alpha = F.softmax(a.masked_fill(user_history_mask == 0, -1e9), dim=2)                                                   # [batch_size, news_num, max_history_num]
        user_representation = (alpha.unsqueeze(dim=3) * history_embedding).sum(dim=2, keepdim=False)                            # [batch_size, news_num, news_embedding_dim]
        return user_representation


class FIM(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(FIM, self).__init__(news_encoder, config)
        assert type(self.news_encoder) == HDC, 'For FIM, the news encoder must be HDC'
        self.HDC_sequence_length = news_encoder.HDC_sequence_length
        self.max_history_num = config.max_history_num
        self.scalar = math.sqrt(float(config.HDC_filter_num))
        self.conv_3D_a = nn.Conv3d(in_channels=4, out_channels=config.conv3D_filter_num_first, kernel_size=config.conv3D_kernel_size_first)
        self.conv_3D_b = nn.Conv3d(in_channels=config.conv3D_filter_num_first, out_channels=config.conv3D_filter_num_second, kernel_size=config.conv3D_kernel_size_second)
        self.maxpool_3D = torch.nn.MaxPool3d(kernel_size=config.maxpooling3D_size, stride=config.maxpooling3D_stride)

    def initialize(self):
        pass

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        candidate_news_d0, candidate_news_dL = candidate_news_representation
        history_embedding_d0, history_embedding_dL = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                                                       user_content_text, user_content_mask, user_content_entity, \
                                                                       user_category, user_subCategory, user_embedding)
        batch_size = candidate_news_d0.size(0)
        news_num = candidate_news_d0.size(1)
        batch_news_num = batch_size * news_num
        # 1. compute 3D matching images
        candidate_news_d0 = candidate_news_d0.unsqueeze(dim=2).permute(0, 1, 2, 4 ,3)                                                       # [batch_size, news_num, 1, HDC_sequence_length, HDC_filter_num]
        candidate_news_dL = candidate_news_dL.unsqueeze(dim=2).permute(0, 1, 2, 3 ,5, 4)                                                    # [batch_size, news_num, 1, 3, HDC_sequence_length, HDC_filter_num]
        history_embedding_d0 = history_embedding_d0.unsqueeze(dim=1)                                                                        # [batch_size, 1, max_history_num, HDC_filter_num, HDC_sequence_length]
        history_embedding_dL = history_embedding_dL.unsqueeze(dim=1)                                                                        # [batch_size, 1, max_history_num, 3, HDC_filter_num, HDC_sequence_length]
        matching_images_d0 = torch.matmul(candidate_news_d0, history_embedding_d0) / self.scalar                                            # [batch_size, news_num, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images_dL = torch.matmul(candidate_news_dL, history_embedding_dL) / self.scalar                                            # [batch_size, news_num, max_history_num, 3, HDC_sequence_length, HDC_sequence_length]
        matching_images = torch.cat([matching_images_d0.unsqueeze(dim=3), matching_images_dL], dim=3).permute(0, 1, 3, 2, 4, 5)             # [batch_size, news_num, 4, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images = matching_images.view(batch_news_num, 4, self.max_history_num, self.HDC_sequence_length, self.HDC_sequence_length) # [batch_size * news_num, 4, max_history_num, HDC_sequence_length, HDC_sequence_length]
        # 2. 3D convolution layers
        Q1 = F.elu(self.conv_3D_a(matching_images), inplace=True)                                                                           # [batch_size * news_num, conv3D_filter_num_first, max_history_num, HDC_sequence_length, HDC_sequence_length]
        Q1 = self.maxpool_3D(Q1)                                                                                                            # [batch_size * news_num, conv3D_filter_num_first, max_history_num_conv1_size, HDC_sequence_length_conv1_size, HDC_sequence_length_conv1_size]
        Q2 = F.elu(self.conv_3D_b(Q1), inplace=True)                                                                                        # [batch_size * news_num, conv3D_filter_num_second, max_history_num_pool1_size, HDC_sequence_length_pool1_size, HDC_sequence_length_pool1_size]
        Q2 = self.maxpool_3D(Q2)                                                                                                            # [batch_size * news_num, conv3D_filter_num_second, max_history_num_conv2_size, HDC_sequence_length_conv2_size, HDC_sequence_length_conv2_size]
        salient_signals = Q2.view([batch_size, news_num, -1])                                                                               # [batch_size * news_num, feature_size]
        return salient_signals


class PUE(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(PUE, self).__init__(news_encoder, config)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(self.news_embedding_dim, config.personalized_embedding_dim, config.attention_dim)

    def initialize(self):
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                # [batch_size, max_history_num, news_embedding_dim]
        q_d = F.relu(self.dense(user_embedding), inplace=True)                                                                                # [batch_size, personalized_embedding_dim]
        user_representation = self.personalizedAttention(history_embedding, q_d, user_history_mask).unsqueeze(dim=1).expand(-1, news_num, -1) # [batch_size, news_num, news_embedding_dim]
        return user_representation


class GRU(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(GRU, self).__init__(news_encoder, config)
        self.gru = nn.GRU(self.news_embedding_dim, config.hidden_dim, batch_first=True)
        self.dec = nn.Linear(config.hidden_dim, self.news_embedding_dim, bias=True)

    def initialize(self):
        for parameter in self.gru.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.dec.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.dec.bias)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        user_history_num = user_history_mask.sum(dim=1, keepdim=False).long()                                                                           # [batch_size]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                          # [batch_size, max_history_num, news_embedding_dim]
        sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)                                                         # [batch_size]
        _, desorted_indices = torch.sort(sorted_indices, descending=False)                                                                              # [batch_size]
        nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)
        if nonzero_indices.size(0) == 0:
            user_representation = torch.zeros([batch_size, news_num, self.news_embedding_dim], device=self.device)                                      # [batch_size, news_num, news_embedding_dim]
            return user_representation
        index = nonzero_indices[-1]
        if index + 1 == batch_size:
            sorted_history_embedding = history_embedding.index_select(0, sorted_indices)                                                                # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num.cpu(), batch_first=True)           # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding)                                                                                            # [1, batch_size, news_embedding_dim]
            h = torch.tanh(self.dec(h.squeeze(dim=0)))                                                                                                  # [batch_size, news_embedding_dim]
            user_representation = h.index_select(0, desorted_indices)                                                                                   # [batch_size, news_embedding_dim]
        else:
            non_empty_indices = sorted_indices[:index+1]
            sorted_history_embedding = history_embedding.index_select(0, non_empty_indices)                                                             # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num[:index+1].cpu(), batch_first=True) # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding)                                                                                            # [1, batch_size, news_embedding_dim]
            h = torch.tanh(self.dec(h.squeeze(dim=0)))                                                                                                  # [batch_size, news_embedding_dim]
            user_representation = torch.cat([h, torch.zeros([batch_size - 1 - index, self.news_embedding_dim], device=self.device)], \
                                            dim=0).index_select(0, desorted_indices)                                                                    # [batch_size, news_embedding_dim]
        user_representation = user_representation.unsqueeze(dim=1).expand(-1, news_num, -1)                                                             # [batch_size, news_num, news_embedding_dim]
        return user_representation


class OMAP(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(OMAP, self).__init__(news_encoder, config)
        self.max_history_num = config.max_history_num
        self.OMAP_head_num = config.OMAP_head_num
        self.HiFi_Ark_regularizer_coefficient = config.HiFi_Ark_regularizer_coefficient
        self.scalar = math.sqrt(float(self.news_embedding_dim))
        self.W = nn.parameter.Parameter(torch.zeros([self.news_embedding_dim, self.OMAP_head_num]))
        self.J_k = torch.ones([self.OMAP_head_num, self.OMAP_head_num])
        self.I_k = torch.eye(self.OMAP_head_num)

    def initialize(self):
        nn.init.orthogonal_(self.W.data)
        self.J_k.cuda()
        self.I_k.cuda()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)        # [batch_size, max_history_num, news_embedding_dim]
        # 1. self-attention
        a = torch.bmm(history_embedding, history_embedding.permute(0, 2, 1)) / self.scalar            # [batch_size, max_history_num, max_history_num]
        mask = user_history_mask.unsqueeze(dim=1).expand(-1, self.max_history_num, -1)                # [batch_size, max_history_num, max_history_num]
        alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=2)                                      # [batch_size, max_history_num, max_history_num]
        history_embedding = history_embedding + torch.bmm(alpha, history_embedding)                   # [batch_size, max_history_num, news_embedding_dim]
        # 2. compute archives of OMAP
        b = torch.matmul(history_embedding, self.W) / self.scalar                                     # [batch_size, max_history_num, OMAP_head_num]
        mask = user_history_mask.unsqueeze(dim=2).expand(-1, -1, self.OMAP_head_num)                  # [batch_size, max_history_num, OMAP_head_num]
        beta = F.softmax(b.masked_fill(mask == 0, -1e9), dim=2)                                       # [batch_size, max_history_num, OMAP_head_num]
        archives = torch.bmm(beta.permute(0, 2, 1), history_embedding)                                # [batch_size, OMAP_head_num, news_embedding_dim]
        # 3. aggregate archives into user representation
        betatheta = torch.bmm(candidate_news_representation, archives.permute(0, 2, 1)) / self.scalar # [batch_size, news_num, OMAP_head_num]
        archive_weights = F.softmax(betatheta, dim=2)                                                 # [batch_size, news_num, OMAP_head_num]
        user_representation = torch.bmm(archive_weights, archives)                                    # [batch_size, news_num, news_embedding_dim]
        # 4. auxiliary loss to regularize the pooling heads \Lambda
        # To minimize the term \Omega = ||\Lambda^{T}\Lambda \odot (J_{k}-I_{k})||_{F} in Hi-Fi Ark
        if self.training:
            Omega = torch.norm(torch.mm(self.W.transpose(1, 0), self.W) * (self.J_k - self.I_k), p='fro')
            self.auxiliary_loss = self.HiFi_Ark_regularizer_coefficient * Omega
        return user_representation


class MINER(UserEncoder):
    """
    Core components:
    1. Poly Attention: extract K interest vectors using K context codes
    2. Disagreement Regularization: encourage diversity among interest vectors
    3. Category-Aware Attention: leverage category information (optional)
    4. Score Aggregation: max/mean/weighted aggregation
    """

    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(MINER, self).__init__(news_encoder, config)

        # Poly attention parameters
        self.K = config.num_interest_vectors  # Number of interest vectors (default: 32)
        self.context_dim = config.context_code_dim  # Context code dimension (default: 200)
        self.aggregation = config.miner_aggregation  # 'max', 'mean', 'weighted'
        self.disagreement_beta = config.disagreement_beta  # Default: 0.8

        # K learnable context codes
        # Shape: [K, context_dim]
        self.context_codes = nn.Parameter(torch.zeros(self.K, self.context_dim))

        # Projection layer for additive attention
        # project h_j to context_dim
        self.W_h = nn.Linear(
            self.news_embedding_dim,
            self.context_dim,
            bias=False
        )

        # Target-aware attention for weighted aggregation
        if self.aggregation == 'weighted':
            self.W_e = nn.Linear(
                self.news_embedding_dim,
                self.news_embedding_dim,
                bias=True
            )

        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.attention_scalar = math.sqrt(float(self.context_dim))
        
        # Category-aware attention parameter
        # In the paper, lambda appears to be a hyperparameter, but here we make it learnable
        self.category_aware_lambda = nn.Parameter(torch.tensor(config.category_aware_lambda, dtype=torch.float32))
        self.category_embedding = news_encoder.category_embedding  # store reference

    def initialize(self):
        """Parameter initialization"""
        # Context codes: orthogonal initialization
        nn.init.orthogonal_(self.context_codes.data)

        # Projection layer: Xavier uniform
        nn.init.xavier_uniform_(self.W_h.weight)

        # Weighted aggregation layer
        if self.aggregation == 'weighted':
            nn.init.xavier_uniform_(self.W_e.weight)
            nn.init.zeros_(self.W_e.bias)

    def poly_attention(self, history_embeddings, history_mask, user_category=None, candidate_category=None):
        """
        Poly Attention with Category-aware Weighting (MINER Equation 5)

        Args:
            history_embeddings: [batch_size, max_history_num, news_embedding_dim]
            history_mask: [batch_size, max_history_num]
            user_category: [batch_size, max_history_num] - history news categories
            candidate_category: [batch_size, news_num] - candidate news categories

        Returns:
            interest_vectors: [batch_size, news_num, K, news_embedding_dim] if category-aware
                           or [batch_size, K, news_embedding_dim] if not
        """
        batch_size = history_embeddings.size(0)
        max_history_num = history_embeddings.size(1)

        # Project history embeddings
        h_proj = torch.tanh(self.W_h(history_embeddings))

        # Compute attention for all K context codes
        logits = torch.matmul(h_proj, self.context_codes.T) / self.attention_scalar  # [B, M, K]

        # Category-aware attention weighting (Equation 5)
        if user_category is not None and candidate_category is not None:
            # Get category embeddings
            hist_cat_emb = self.category_embedding(user_category)  # [B, M, cat_dim]
            cand_cat_emb = self.category_embedding(candidate_category)  # [B, N, cat_dim]

            # Normalize for cosine similarity
            hist_cat_norm = F.normalize(hist_cat_emb, p=2, dim=2)  # [B, M, cat_dim]
            cand_cat_norm = F.normalize(cand_cat_emb, p=2, dim=2)  # [B, N, cat_dim]

            # Cosine similarity: [B, M, cat_dim] @ [B, cat_dim, N] = [B, M, N]
            category_sim = torch.bmm(hist_cat_norm, cand_cat_norm.transpose(1, 2))

            # Expand logits: [B, M, K] → [B, M, N, K]
            news_num = candidate_category.size(1)
            logits_expanded = logits.unsqueeze(2).expand(-1, -1, news_num, -1)

            # Add category bias: λ * cos(b_j, b_c)
            category_bias = self.category_aware_lambda * category_sim.unsqueeze(3)  # [B, M, N, 1]
            logits = logits_expanded + category_bias  # [B, M, N, K]

            # Mask
            mask_expanded = history_mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, news_num, self.K)
            logits = logits.masked_fill(mask_expanded == 0, -1e9)

            # Softmax over history dimension
            attn_weights = F.softmax(logits, dim=1)  # [B, M, N, K]

            # Weighted sum: [B, N, K, M] @ [B, M, D] = [B, N, K, D]
            interest_vectors = torch.einsum('bmnk,bmd->bnkd', attn_weights, history_embeddings)
        else:
            # Original category-agnostic poly attention
            mask_expanded = history_mask.unsqueeze(2).expand(-1, -1, self.K)
            logits = logits.masked_fill(mask_expanded == 0, -1e9)
            attn_weights = F.softmax(logits, dim=1)
            interest_vectors = torch.bmm(attn_weights.transpose(1, 2), history_embeddings)  # [B, K, D]

        return interest_vectors

    def compute_disagreement_loss(self, interest_vectors):
        """
        Disagreement Regularization (Equation 3)

        Encourage diversity by minimizing the average cosine similarity between interest vectors

        Args:
            interest_vectors: [batch_size, K, news_embedding_dim]
                or [batch_size, news_num, K, news_embedding_dim]

        Returns:
            loss: scalar
        """
        # Reshape if needed
        if interest_vectors.dim() == 4:
            batch_size, news_num, K, D = interest_vectors.size()
            interest_vectors = interest_vectors.view(batch_size * news_num, K, D)

        # Normalize interest vectors
        # [B, K, D]
        normalized = F.normalize(interest_vectors, p=2, dim=2)

        # Pairwise cosine similarity
        # [B, K, D] @ [B, D, K] = [B, K, K]
        similarity_matrix = torch.bmm(
            normalized,
            normalized.transpose(1, 2)
        )

        # Average over all pairs
        K = interest_vectors.size(1)
        loss = similarity_matrix.sum(dim=(1, 2)) / (K * K)

        return loss.mean()

    def forward(
        self,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_category,
        user_subCategory,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        user_embedding,
        candidate_news_representation,
        candidate_category=None
    ):
        """
        MINER Forward Pass

        Args:
            user_category: [batch_size, max_history_num]
            user_history_mask: [batch_size, max_history_num]
            candidate_news_representation: [batch_size, news_num, news_embedding_dim]
            candidate_category: [batch_size, news_num] - candidate news categories

        Returns:
            user_representation: [batch_size, news_num, news_embedding_dim]
        """
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)

        # Encode history news
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_embedding
        )

        # Poly attention: extract K interest vectors
        # Category-aware if candidate_category provided
        interest_vectors = self.poly_attention(
            history_embedding,
            user_history_mask,
            user_category,
            candidate_category
        )

        # Expand for each candidate news (if not already expanded by category-aware attention)
        if candidate_category is not None:
            # Already [B, N, K, D] from category-aware poly_attention
            interest_vectors_exp = interest_vectors
        else:
            # [B, K, D] → [B, 1, K, D] → [B, N, K, D]
            interest_vectors_exp = interest_vectors.unsqueeze(1).expand(-1, news_num, -1, -1)

        # Compute disagreement loss (auxiliary loss)
        if self.training:
            self.auxiliary_loss = self.disagreement_beta * self.compute_disagreement_loss(interest_vectors_exp)

        # Aggregate interest vectors for each candidate news
        if self.aggregation == 'weighted':
            # Target-aware weighted sum
            # [B, N, D]
            W_e_h_c = F.gelu(self.W_e(candidate_news_representation))

            # Attention logits
            # [B, N, 1, D] @ [B, N, D, K] = [B, N, 1, K] → [B, N, K]
            logits = torch.matmul(
                W_e_h_c.unsqueeze(2),  # [B, N, 1, D]
                interest_vectors_exp.transpose(2, 3)  # [B, N, D, K]
            ).squeeze(2)  # [B, N, K]

            # Attention weights
            alpha = F.softmax(logits, dim=2)  # [B, N, K]

            # Weighted sum of interest vectors
            user_representation = (alpha.unsqueeze(3) * interest_vectors_exp).sum(dim=2)  # [B, N, D]

        elif self.aggregation == 'mean':
            # Average pooling
            user_representation = interest_vectors_exp.mean(dim=2)  # [B, N, D]

        elif self.aggregation == 'max':
            # Max pooling
            user_representation = interest_vectors_exp.max(dim=2)[0]  # [B, N, D]

        return user_representation


class PENR(UserEncoder):
    """
    PENR User Encoder (CIKM'21)
    Multi-View Interest Representation with 5 independent attention networks
    """
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(PENR, self).__init__(news_encoder, config)
        self.num_views = getattr(config, 'penr_num_interest_views', 5)  # N_a = 5
        self.query_dim = getattr(config, 'penr_attention_query_dim', 200)
        self.news_embedding_dim = news_encoder.news_embedding_dim  # 300

        # 5 Independent Attention Networks (Equations 11-13)
        # Each view has separate parameters: (q_i, V_i, v_i)
        self.view_projections = nn.ModuleList([
            nn.Linear(self.news_embedding_dim, self.query_dim, bias=True)
            for _ in range(self.num_views)
        ])
        self.view_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(self.query_dim))
            for _ in range(self.num_views)
        ])

        # Interest Discriminator (Equation 22, training only)
        self.discriminator = nn.Linear(self.news_embedding_dim, self.num_views, bias=True)

    def initialize(self):
        # Initialize attention networks
        for i in range(self.num_views):
            nn.init.xavier_uniform_(self.view_projections[i].weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.view_projections[i].bias)
            nn.init.xavier_uniform_(self.view_queries[i].unsqueeze(0))

        # Initialize discriminator
        nn.init.xavier_uniform_(self.discriminator.weight)
        nn.init.zeros_(self.discriminator.bias)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory,
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)

        # Encode browsing history: {r_1, r_2, ..., r_N_h}
        history_embedding = self.news_encoder(
            user_title_text, user_title_mask, user_title_entity,
            user_content_text, user_content_mask, user_content_entity,
            user_category, user_subCategory, user_embedding
        )  # [batch_size, max_history_num, news_embedding_dim]

        # Expand history for each candidate news
        history_embedding_exp = history_embedding.unsqueeze(1).expand(-1, news_num, -1, -1)  # [B, N, max_history_num, D]
        user_history_mask_exp = user_history_mask.unsqueeze(1).expand(-1, news_num, -1)  # [B, N, max_history_num]

        # Compute multi-view interest representations (Equations 11-13)
        view_interests = []
        for i in range(self.num_views):
            # Equation 11: α_i,k = q_i^T tanh(V_i r_k + v_i)
            projected = torch.tanh(self.view_projections[i](history_embedding_exp))  # [B, N, max_history_num, query_dim]
            scores = torch.matmul(projected, self.view_queries[i])  # [B, N, max_history_num]

            # Apply mask and softmax
            scores = scores.masked_fill(user_history_mask_exp == 0, -1e9)
            alpha = F.softmax(scores, dim=2)  # [B, N, max_history_num]

            # Equation 12: u_i = Σ_{k=1}^{N_h} α_i,k r_k
            u_i = torch.bmm(
                alpha.reshape(batch_size * news_num, 1, -1),  # [B*N, 1, max_history_num]
                history_embedding_exp.reshape(batch_size * news_num, -1, self.news_embedding_dim)  # [B*N, max_history_num, D]
            ).reshape(batch_size, news_num, self.news_embedding_dim)  # [B, N, D]

            view_interests.append(u_i)

        # Equation 13: u = Stack(u_1, ..., u_N_a)
        u = torch.stack(view_interests, dim=2)  # [B, N, N_a, D] where N_a=5

        # Compute Interest Discriminator loss (Equation 22, training only)
        if self.training:
            # Compute discriminator predictions for each view
            # d̂_i = softmax(W_sub u_i + b_sub)
            discriminator_logits = []
            for i in range(self.num_views):
                logits_i = self.discriminator(view_interests[i])  # [B, N, N_a]
                discriminator_logits.append(logits_i)

            discriminator_logits = torch.stack(discriminator_logits, dim=2)  # [B, N, N_a, N_a]

            # Target: each view i should belong to subspace i
            # Create one-hot targets: view 0 → [1,0,0,0,0], view 1 → [0,1,0,0,0], etc.
            targets = torch.eye(self.num_views, device=u.device).unsqueeze(0).unsqueeze(0).expand(batch_size, news_num, -1, -1)  # [B, N, N_a, N_a]

            # Compute cross-entropy loss (Equation 23)
            # ℓ_aux = -(1/(N·N_a)) Σ Σ [d_i,j log(d̂_i,j) + (1-d_i,j)log(1-d̂_i,j)]
            discriminator_probs = F.softmax(discriminator_logits, dim=3)  # [B, N, N_a, N_a]
            aux_loss = -(targets * torch.log(discriminator_probs + 1e-8) + (1 - targets) * torch.log(1 - discriminator_probs + 1e-8))
            aux_loss = aux_loss.mean()

            # Store auxiliary loss for use in trainer
            self.auxiliary_loss = aux_loss
        else:
            self.auxiliary_loss = None

        # Return stacked multi-view representation
        # u: [B, N, N_a, D] where N_a=5, D=300
        return u

class POPCORN(UserEncoder):
    """
    POPCORN User Encoder with Candidate-guided User Modeling (I2)

    Structure:
        1. Encode user history with the News Encoder -> [f_j ; p_j]
        2. Candidate-guided Top-k Attention (Module 1) — two modes:
            [topic-aware] Q=t_c (candidate topic), K=t_j (history topic), V=f_j/p_j
                |- compute alpha once (shared topic for f/p)
                |- Top-K Selection & Reweighting
                +- Gated Residual (for f_j and p_j separately)
            [candidate-aware] Q=f_c/p_c, K=f_j/p_j, V=f_j/p_j
                |- alpha_f = attn(f_c, f_j), alpha_p = attn(p_c, p_j) computed independently
                |- Top-K Selection & Reweighting
                +- Gated Residual (for f_j and p_j separately)
        3. User Encoder (Module 2) - Plug-in design:
            |- f^a_j -> Base User Encoder -> f_u (pure content user embedding)
            +- p^a_j -> Base User Encoder -> p_u (popularity-pattern user embedding)

    Attention Mode (config.popcorn_attention_mode):
        - 'topic-aware': both Q and K at the topic level (same semantic space)
        - 'candidate-aware': both Q and K at the feature level (same semantic space)

    Input:
        - candidate_news_representation: (batch_size, N, base_dim) - [f_c ; p_c]
        - candidate_category: (batch_size, N)
        - candidate_subCategory: (batch_size, N)
        - user history raw data

    Output:
        - user_representation: (batch_size, N, base_dim) - [f_u ; p_u] concatenated
    """

    def __init__(self, news_encoder, config):
        super(POPCORN, self).__init__(news_encoder, config)
        self.config = config

        # news_encoder is POPCORN necessary.
        assert news_encoder.__class__.__name__ == 'POPCORN', 'POPCORN user encoder requires POPCORN news encoder'

        base_dim = news_encoder.news_embedding_dim
        self.news_embedding_dim = base_dim

        # I1/I2 flag check
        self.use_I1 = getattr(config, 'use_I1', False)
        self.use_I2 = getattr(config, 'use_I2', False)

        assert self.use_I1 or self.use_I2, "POPCORN requires at least I1 or I2. For base models, use the base encoder directly (e.g., --user_encoder=MHSA)"

        # Dimension setup
        if self.use_I1:
            # I1=True: base_dim = 2 * original_dim
            d_half = base_dim // 2
            self.d_half = d_half
            self.original_dim = d_half
        else:
            # I1=False: base_dim = original_dim
            self.d_half = None
            self.original_dim = base_dim

        # Initialize Top-K Attention parameters only when I2=True
        if self.use_I2:
            # Candidate-guided News Selection parameters
            self.K = config.popcorn_top_k
            self.epsilon = config.popcorn_epsilon

            # Whether to use the gate
            self.use_gate = getattr(config, 'popcorn_use_gate', True)

            # (1) Candidate-guided Top-k Attention parameters
            self.attention_mode = getattr(config, 'popcorn_attention_mode', 'topic-aware')
            self.num_heads = getattr(config, 'popcorn_attention_heads', getattr(config, 'head_num', 20))

            # Topic embedding dimension
            d_topic = config.category_embedding_dim + config.subCategory_embedding_dim  # e.g., 100
            self.d_topic = d_topic

            if self.use_I1:
                # I1=True, I2=True: dual-branch
                d_half = self.d_half

                if self.attention_mode == 'topic-aware':
                    # Q=t_c (topic), K=t_j (topic) -> same level
                    self.d_k = d_topic // self.num_heads
                    assert self.d_k > 0, f"d_topic ({d_topic}) must be >= num_heads ({self.num_heads})"
                    self.W_Q_topic = nn.Linear(d_topic, self.num_heads * self.d_k, bias=False)
                    self.W_K_topic = nn.Linear(d_topic, self.num_heads * self.d_k, bias=False)  # single (shared by f/p)
                elif self.attention_mode == 'candidate-aware':
                    # Q=f_c/p_c, K=f_j/p_j -> same level
                    self.d_k = d_half // self.num_heads
                    assert self.d_k > 0, f"d_half ({d_half}) must be >= num_heads ({self.num_heads})"
                    self.W_Q_f = nn.Linear(d_half, self.num_heads * self.d_k, bias=False)
                    self.W_K_f = nn.Linear(d_half, self.num_heads * self.d_k, bias=False)
                    self.W_Q_p = nn.Linear(d_half, self.num_heads * self.d_k, bias=False)
                    self.W_K_p = nn.Linear(d_half, self.num_heads * self.d_k, bias=False)
                else:
                    raise ValueError(f'Unknown attention mode: {self.attention_mode}')

                # (2) Gated Residual Connection (dual-branch)
                if self.use_gate:
                    self.W_g_f = nn.Linear(d_half, 1, bias=False)  # for f_j
                    self.W_g_p = nn.Linear(d_half, 1, bias=False)  # for p_j
                else:
                    self.W_g_f = None
                    self.W_g_p = None

            else:
                # I1=False, I2=True: single-branch
                original_dim = self.original_dim

                if self.attention_mode == 'topic-aware':
                    self.d_k = d_topic // self.num_heads
                    assert self.d_k > 0, f"d_topic ({d_topic}) must be >= num_heads ({self.num_heads})"
                    self.W_Q_topic = nn.Linear(d_topic, self.num_heads * self.d_k, bias=False)
                    self.W_K_topic = nn.Linear(d_topic, self.num_heads * self.d_k, bias=False)
                elif self.attention_mode == 'candidate-aware':
                    self.d_k = original_dim // self.num_heads
                    assert self.d_k > 0, f"original_dim ({original_dim}) must be >= num_heads ({self.num_heads})"
                    self.W_Q_h = nn.Linear(original_dim, self.num_heads * self.d_k, bias=False)
                    self.W_K_h = nn.Linear(original_dim, self.num_heads * self.d_k, bias=False)
                else:
                    raise ValueError(f'Unknown attention mode: {self.attention_mode}')

                # (2) Gated Residual Connection (single-branch)
                if self.use_gate:
                    self.W_g_single = nn.Linear(original_dim, 1, bias=False)
                else:
                    self.W_g_single = None

        # (4) Select Base User Encoder (specified in config, shared for f_u and p_u)
        base_type = getattr(config, 'popcorn_base_user_encoder', 'ATT')

        # (3) Identity News Encoder - returns r_j as-is
        class IdentityNewsEncoder:
            def __init__(self, dim, category_embedding=None):
                self.news_embedding_dim = dim
                if category_embedding is not None:
                    self.category_embedding = category_embedding

            def __call__(self, x, *args, **kwargs):
                # return the first argument (already-encoded embedding) as-is
                return x

            def initialize(self):
                pass

        real_cat_emb = None
        if base_type == 'MINER':
            if hasattr(news_encoder, 'category_embedding'):
                real_cat_emb = news_encoder.category_embedding
            elif hasattr(news_encoder, 'base_encoder') and hasattr(news_encoder.base_encoder, 'category_embedding'):
                real_cat_emb = news_encoder.base_encoder.category_embedding

        identity_encoder = IdentityNewsEncoder(self.original_dim, real_cat_emb)

        if base_type == 'ATT':
            from userEncoders import ATT
            self.base_user_encoder = ATT(identity_encoder, config)
        elif base_type == 'MHSA':
            from userEncoders import MHSA
            self.base_user_encoder = MHSA(identity_encoder, config)
        elif base_type == 'CATT':
            from userEncoders import CATT
            self.base_user_encoder = CATT(identity_encoder, config)
        elif base_type == 'GRU':
            from userEncoders import GRU
            self.base_user_encoder = GRU(identity_encoder, config)
        elif base_type == 'SUE':
            from userEncoders import SUE
            self.base_user_encoder = SUE(identity_encoder, config)  
        elif base_type == 'LSTUR':
            from userEncoders import LSTUR
            self.base_user_encoder = LSTUR(identity_encoder, config)
        elif base_type == 'CROWN':
            from userEncoders import CROWN
            self.base_user_encoder = CROWN(identity_encoder, config)
        elif base_type == 'PENR':
            from userEncoders import PENR
            self.base_user_encoder = PENR(identity_encoder, config)
        elif base_type == 'MINER':
            from userEncoders import MINER
            self.base_user_encoder = MINER(identity_encoder, config)
        else:
            raise ValueError(f'Unknown base user encoder: {base_type}')

    def initialize(self):
        self.news_encoder.initialize()
        
        if self.use_I2:
            if self.attention_mode == 'topic-aware':
                nn.init.xavier_uniform_(self.W_Q_topic.weight)
                nn.init.xavier_uniform_(self.W_K_topic.weight)
            elif self.attention_mode == 'candidate-aware':
                if self.use_I1:
                    nn.init.xavier_uniform_(self.W_Q_f.weight)
                    nn.init.xavier_uniform_(self.W_K_f.weight)
                    nn.init.xavier_uniform_(self.W_Q_p.weight)
                    nn.init.xavier_uniform_(self.W_K_p.weight)
                else:
                    nn.init.xavier_uniform_(self.W_Q_h.weight)
                    nn.init.xavier_uniform_(self.W_K_h.weight)

            if self.use_gate:
                if self.use_I1:
                    nn.init.xavier_uniform_(self.W_g_f.weight)
                    nn.init.xavier_uniform_(self.W_g_p.weight)
                else:
                    nn.init.xavier_uniform_(self.W_g_single.weight)

        self.base_user_encoder.initialize()

    def _multihead_attention(self, Q_proj, K_proj, user_history_mask, N):
        """
        Multi-head Attention → α (batch_size, N, M)

        Args:
            Q_proj: (batch_size, N, num_heads * d_k) - projected query
            K_proj: (batch_size, M, num_heads * d_k) - projected key
            user_history_mask: (batch_size, M)
            N: number of candidate news

        Returns:
            α: (batch_size, N, M) - averaged attention weights across heads
        """
        batch_size = Q_proj.size(0)
        M = K_proj.size(1)

        Q = Q_proj.view(batch_size, N, self.num_heads, self.d_k)   # (B, N, H, d_k)
        K = K_proj.view(batch_size, M, self.num_heads, self.d_k)   # (B, M, H, d_k)

        Q_exp = Q.unsqueeze(2)  # (B, N, 1, H, d_k)
        K_exp = K.unsqueeze(1)  # (B, 1, M, H, d_k)

        e = (Q_exp * K_exp).sum(dim=-1) / (self.d_k ** 0.5)  # (B, N, M, H)

        mask_exp = user_history_mask.unsqueeze(1).unsqueeze(-1).expand(-1, N, -1, self.num_heads)
        e = e.masked_fill(mask_exp == 0, -1e9)

        α_multihead = F.softmax(e, dim=2)  # (B, N, M, H)
        α = α_multihead.mean(dim=-1)       # (B, N, M)
        return α

    def _topk_reweight(self, α, M):
        """
        Top-K Selection & Reweighting

        Args:
            α: (batch_size, N, M)
            M: history length

        Returns:
            α_final: (batch_size, N, M) - reweighted & renormalized
        """
        K_select = min(self.K, M)
        _, top_k_indices = torch.topk(α, k=K_select, dim=-1, largest=True)

        α_reweighted = α.clone()
        non_topk_mask = torch.ones_like(α, dtype=torch.bool)
        non_topk_mask.scatter_(dim=-1, index=top_k_indices, value=False)
        α_reweighted[non_topk_mask] *= self.epsilon

        α_sum = α_reweighted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        α_final = α_reweighted / α_sum
        return α_final

    def _gated_residual(self, α_final, val_vec, W_g, N):
        """
        Gated Residual Connection (optional)

        Args:
            α_final: (batch_size, N, M)
            val_vec: (batch_size, M, d_half) - value vectors (f_j or p_j)
            W_g: nn.Linear or None - gate projection (None if use_gate=False)
            N: number of candidate news

        Returns:
            r: (batch_size, N, M, d_half) - reweighted history
        """
        val_expanded = val_vec.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, M, d_half)

        # Branch depending on whether the gate is used
        if self.use_gate and W_g is not None:
            # Original gated residual: r = g_i ⊙ (α_j · p_j) + (1 - g_i) ⊙ p_j
            g_logits = W_g(val_expanded).squeeze(-1) + α_final
            g_i = torch.sigmoid(g_logits)  # (B, N, M)

            g_i_exp = g_i.unsqueeze(-1)
            α_exp = α_final.unsqueeze(-1)

            r = g_i_exp * (α_exp * val_expanded) + (1 - g_i_exp) * val_expanded
        else:
            # Direct attention weighting: r = α_j · p_j
            α_exp = α_final.unsqueeze(-1)  # (B, N, M, 1)
            r = α_exp * val_expanded       # (B, N, M, d_half)

        return r  # (B, N, M, d_half)

    def candidate_guided_selection_topic(self, t_c, t_j, f_j, p_j, user_history_mask):
        """
        Topic-aware Candidate-guided Selection
        Q=t_c (candidate topic), K=t_j (history topic) -> same-level attention
        Compute alpha only once and apply it to both f_j and p_j

        Args:
            t_c: (batch_size, N, d_topic) - candidate topic embedding
            t_j: (batch_size, M, d_topic) - history topic embedding
            f_j: (batch_size, M, d_half) - history content features (Value for f branch)
            p_j: (batch_size, M, d_half) - history popularity features (Value for p branch)
            user_history_mask: (batch_size, M)

        Returns:
            r_f: (batch_size, N, M, d_half)
            r_p: (batch_size, N, M, d_half)
        """
        N = t_c.size(1)
        M = t_j.size(1)

        # (1) compute alpha (shared by f/p — comparing topics)
        Q_proj = self.W_Q_topic(t_c)   # (B, N, H*d_k)
        K_proj = self.W_K_topic(t_j)   # (B, M, H*d_k)
        α = self._multihead_attention(Q_proj, K_proj, user_history_mask, N)  # (B, N, M)

        # (2) Top-K Reweighting
        α_final = self._topk_reweight(α, M)  # (B, N, M)

        # (3) Gated Residual (apply the same α_final to f_j and p_j separately)
        r_f = self._gated_residual(α_final, f_j, self.W_g_f, N)  # (B, N, M, d_half)
        r_p = self._gated_residual(α_final, p_j, self.W_g_p, N)  # (B, N, M, d_half)

        return r_f, r_p

    def candidate_guided_selection_candidate(self, f_c, p_c, f_j, p_j, user_history_mask):
        """
        Candidate-aware Candidate-guided Selection
        Q=f_c/p_c, K=f_j/p_j -> same-level attention (each branch independent)

        Args:
            f_c: (batch_size, N, d_half) - candidate content features
            p_c: (batch_size, N, d_half) - candidate popularity features
            f_j: (batch_size, M, d_half) - history content features
            p_j: (batch_size, M, d_half) - history popularity features
            user_history_mask: (batch_size, M)

        Returns:
            r_f: (batch_size, N, M, d_half)
            r_p: (batch_size, N, M, d_half)
        """
        N = f_c.size(1)
        M = f_j.size(1)

        # (1) f branch: Q=f_c, K=f_j
        Q_f = self.W_Q_f(f_c)  # (B, N, H*d_k)
        K_f = self.W_K_f(f_j)  # (B, M, H*d_k)
        α_f = self._multihead_attention(Q_f, K_f, user_history_mask, N)  # (B, N, M)
        α_f_final = self._topk_reweight(α_f, M)
        r_f = self._gated_residual(α_f_final, f_j, self.W_g_f, N)  # (B, N, M, d_half)

        # (2) p branch: Q=p_c, K=p_j
        Q_p = self.W_Q_p(p_c)  # (B, N, H*d_k)
        K_p = self.W_K_p(p_j)  # (B, M, H*d_k)
        α_p = self._multihead_attention(Q_p, K_p, user_history_mask, N)  # (B, N, M)
        α_p_final = self._topk_reweight(α_p, M)
        r_p = self._gated_residual(α_p_final, p_j, self.W_g_p, N)  # (B, N, M, d_half)

        return r_f, r_p

    def forward(self, user_title_text, user_title_mask, user_title_entity,
                user_content_text, user_content_mask, user_content_entity,
                user_category, user_subCategory, user_history_mask,
                user_history_graph, user_history_category_mask,
                user_history_category_indices, user_embedding,
                candidate_news_representation,
                candidate_category=None, candidate_subCategory=None):
        """
        Args:
            candidate_news_representation:
                - I1=False: (batch_size, N, original_dim) - h_c
                - I1=True: (batch_size, N, 2*original_dim) - [f_c ; p_c]
            candidate_category: (batch_size, N) - candidate news category indices
            candidate_subCategory: (batch_size, N) - candidate news subcategory indices
            user_history_mask: (batch_size, M)

        Returns:
            user_representation:
                - I1=False: (batch_size, N, original_dim) - u_h
                - I1=True, I3=False: (batch_size, N, original_dim) - f_u only
                - I1=True, I3=True: (batch_size, N, 2*original_dim) - [f_u ; p_u]
        """
        batch_size = user_title_text.size(0)
        N = candidate_news_representation.size(1)

        # User history news encoding
        history_repr = self.news_encoder(
            user_title_text, user_title_mask, user_title_entity,
            user_content_text, user_content_mask, user_content_entity,
            user_category, user_subCategory, user_embedding
        )  # I1=False: (batch, M, original_dim), I1=True: (batch, M, 2*original_dim)

        # ===== Branch into 4 paths depending on the I1/I2 combination =====

        if not self.use_I1 and not self.use_I2:
            raise ValueError("I1=False, I2=False case should not use POPCORN. Use base model directly.")

        elif not self.use_I1 and self.use_I2:
            # Path 2: I2 only (I1=False, I2=True)
            # h_j -> single-branch Top-K -> base_user_encoder -> u_h
            # I3 is impossible since I1=False (no p)
            return self._forward_I2_only(history_repr, user_history_mask, user_history_graph,
                                         user_history_category_mask, user_history_category_indices,
                                         user_embedding, candidate_news_representation,
                                         candidate_category, candidate_subCategory,
                                         user_category, user_subCategory, batch_size, N)

        elif self.use_I1 and not self.use_I2:
            # Path 3: I1 only (I1=True, I2=False)
            # f_j, p_j -> base_user_encoder (no Top-K) -> compute f_u, p_u
            # Final return: f_u only if I3=False, [f_u ; p_u] if I3=True
            base_dim_half = self.d_half
            f_j = history_repr[:, :, :base_dim_half]  # (batch, M, d_half)
            p_j = history_repr[:, :, base_dim_half:]  # (batch, M, d_half)

            return self._forward_I1_only(f_j, p_j, user_history_mask, user_history_graph,
                                         user_history_category_mask, user_history_category_indices,
                                         user_embedding, candidate_news_representation, batch_size, N)

        else:
            # Path 4: I1 + I2 (I1=True, I2=True)
            # f_j, p_j -> dual-branch Top-K -> base_user_encoder -> compute f_u, p_u
            # Final return: f_u only if I3=False, [f_u ; p_u] if I3=True
            base_dim_half = self.d_half
            f_j = history_repr[:, :, :base_dim_half]  # (batch, M, d_half)
            p_j = history_repr[:, :, base_dim_half:]  # (batch, M, d_half)

        # ===== Candidate-guided Top-k Attention =====
        if self.attention_mode == 'topic-aware':
            # t_c: candidate news topic, t_j: history news topic
            candidate_cat_emb = self.news_encoder.category_embedding(candidate_category)
            candidate_subcat_emb = self.news_encoder.subCategory_embedding(candidate_subCategory)
            t_c = torch.cat([candidate_cat_emb, candidate_subcat_emb], dim=-1)  # (B, N, d_topic)

            history_cat_emb = self.news_encoder.category_embedding(user_category)
            history_subcat_emb = self.news_encoder.subCategory_embedding(user_subCategory)
            t_j = torch.cat([history_cat_emb, history_subcat_emb], dim=-1)      # (B, M, d_topic)

            r_f, r_p = self.candidate_guided_selection_topic(
                t_c, t_j, f_j, p_j, user_history_mask
            )
        elif self.attention_mode == 'candidate-aware':
            # f_c, p_c: candidate news features
            f_c = candidate_news_representation[:, :, :base_dim_half]  # (B, N, d/2)
            p_c = candidate_news_representation[:, :, base_dim_half:]  # (B, N, d/2)

            r_f, r_p = self.candidate_guided_selection_candidate(
                f_c, p_c, f_j, p_j, user_history_mask
            )
        # r_f, r_p: (batch, N, M, d/2)

        # ===== Compute f_u (call Base user encoder) =====
        # flatten r_f: (batch*N, M, d/2)
        r_f_flat = r_f.reshape(batch_size * N, -1, base_dim_half)

        # expand mask: (batch*N, M)
        M = r_f_flat.size(1)
        mask_flat = user_history_mask.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1)

        # Expand extra arguments for SUE (only when not None)
        if user_history_graph is not None:
            G = user_history_graph.size(1)  # max_history_num + category_num (for GCN)
            graph_flat = user_history_graph.unsqueeze(1).expand(-1, N, -1, -1).reshape(batch_size * N, G, G)
        else:
            graph_flat = None
        cat_mask_flat = user_history_category_mask.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1) if user_history_category_mask is not None else None
        cat_indices_flat = user_history_category_indices.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1) if user_history_category_indices is not None else None

        # Expand user_embedding for LSTUR (batch_size, emb_dim) -> (batch_size * N, emb_dim)
        if user_embedding is not None:
            user_embedding_flat = user_embedding.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1)
        else:
            user_embedding_flat = None

        # Dummy candidate (needed for the base encoder to expand)
        dummy_candidate = torch.zeros(batch_size * N, 1, base_dim_half).to(r_f_flat.device)

        # Call Base user encoder forward() - compute f_u
        f_u_flat = self.base_user_encoder.forward(
            r_f_flat,  # in the user_title_text position (identity encoder returns it as-is)
            mask_flat,  # user_title_mask
            None, None, None, None,  # entity, content (unused)
            None, None,  # category, subCategory (unused)
            mask_flat,  # user_history_mask
            graph_flat, cat_mask_flat, cat_indices_flat,  # needed by SUE (ignored by other encoders)
            user_embedding_flat,  # user_embedding (needed by LSTUR etc.)
            dummy_candidate  # candidate_news_representation (treated as news_num=1)
        )  # (batch*N, 1, d/2), or (batch*N, 1, 5, d/2) for PENR

        # For multi-view encoders such as PENR, handle the view dimension (simple mean)
        if f_u_flat.dim() == 4:
            # [batch*N, 1, num_views, d/2] -> [batch*N, 1, d/2]
            f_u_flat = f_u_flat.mean(dim=2)

        # Reshape: (batch, N, d/2)
        f_u = f_u_flat.squeeze(1).reshape(batch_size, N, base_dim_half)

        # ===== Compute p_u (call Base user encoder) =====
        # flatten r_p: (batch*N, M, d/2)
        r_p_flat = r_p.reshape(batch_size * N, -1, base_dim_half)

        # Call Base user encoder forward() - compute p_u
        p_u_flat = self.base_user_encoder.forward(
            r_p_flat,  # in the user_title_text position (identity encoder returns it as-is)
            mask_flat,  # user_title_mask
            None, None, None, None,  # entity, content (unused)
            None, None,  # category, subCategory (unused)
            mask_flat,  # user_history_mask
            graph_flat, cat_mask_flat, cat_indices_flat,  # needed by SUE (ignored by other encoders)
            user_embedding_flat,  # user_embedding (needed by LSTUR etc.)
            dummy_candidate  # candidate_news_representation (treated as news_num=1)
        )  # (batch*N, 1, d/2), or (batch*N, 1, 5, d/2) for PENR

        # For multi-view encoders such as PENR, handle the view dimension (simple mean)
        if p_u_flat.dim() == 4:
            p_u_flat = p_u_flat.mean(dim=2)

        # Reshape: (batch, N, d/2)
        p_u = p_u_flat.squeeze(1).reshape(batch_size, N, base_dim_half)

        # ===== Concatenate [f_u ; p_u] =====
        use_I3 = getattr(self.config, 'use_I3', False)
        
        if use_I3:
            user_representation = torch.cat([f_u, p_u], dim=-1)  # (batch, N, base_dim)
        else:
            user_representation = f_u  # (batch, N, d_half)

        return user_representation


    def _forward_I1_only(self, f_j, p_j, user_history_mask, user_history_graph,
                         user_history_category_mask, user_history_category_indices,
                         user_embedding, candidate_news_representation, batch_size, N):
        """
        Path 3: +I1 only (I1=True, I2=False)
        f_j -> base_user_encoder -> f_u
        p_j -> base_user_encoder -> p_u (only when I3=True)

        Args:
            f_j: (batch, M, d_half) - content features
            p_j: (batch, M, d_half) - popularity features

        Returns:
            user_representation:
                - I3=False: (batch, N, d_half) - f_u only
                - I3=True: (batch, N, 2*d_half) - [f_u ; p_u]
        """
        # Call Base user encoder (compute f_u)
        dummy_candidate = torch.zeros(batch_size, 1, f_j.shape[-1]).to(f_j.device)

        f_u_base = self.base_user_encoder.forward(
            f_j,  # user_title_text position
            user_history_mask,  # user_title_mask
            None, None, None, None,
            None, None,
            user_history_mask,
            user_history_graph, user_history_category_mask, user_history_category_indices,
            user_embedding,
            dummy_candidate
        )  # (batch, d_half), (batch, 1, d_half), or (batch, 1, 5, d_half)

        # For multi-view encoders such as PENR, handle the view dimension (simple mean)
        if f_u_base.dim() == 4:
            f_u_base = f_u_base.mean(dim=2)

        if f_u_base.dim() == 3:
            f_u_base = f_u_base.squeeze(1)

        # Expand to (batch, N, d_half)
        f_u_expanded = f_u_base.unsqueeze(1).expand(-1, N, -1)  # (batch, N, d_half)

        # Check the I3 flag
        use_I3 = getattr(self.config, 'use_I3', False)

        if use_I3:
            # I3=True: compute p_u using p_j
            p_u_base = self.base_user_encoder.forward(
                p_j,  # user_title_text position
                user_history_mask,  # user_title_mask
                None, None, None, None,
                None, None,
                user_history_mask,
                user_history_graph, user_history_category_mask, user_history_category_indices,
                user_embedding,
                dummy_candidate
            )  # (batch, d_half), (batch, 1, d_half), or (batch, 1, 5, d_half)

            # For multi-view encoders such as PENR, handle the view dimension (simple mean)
            if p_u_base.dim() == 4:
                p_u_base = p_u_base.mean(dim=2)

            if p_u_base.dim() == 3:
                p_u_base = p_u_base.squeeze(1)

            # Expand to (batch, N, d_half)
            p_u_expanded = p_u_base.unsqueeze(1).expand(-1, N, -1)  # (batch, N, d_half)

            # Concatenate [f_u ; p_u]
            user_representation = torch.cat([f_u_expanded, p_u_expanded], dim=-1)  # (batch, N, 2*d_half)
        else:
            # I3=False: return f_u only
            user_representation = f_u_expanded  # (batch, N, d_half)

        return user_representation

    def _forward_I2_only(self, history_repr, user_history_mask, user_history_graph,
                         user_history_category_mask, user_history_category_indices,
                         user_embedding, candidate_news_representation,
                         candidate_category, candidate_subCategory,
                         user_category, user_subCategory, batch_size, N):
        """
        Path 2: +I2 only (I1=False, I2=True)
        h_j -> single-branch Top-K -> base_user_encoder -> u_h

        Args:
            history_repr: (batch, M, original_dim) - h_j
            candidate_news_representation: (batch, N, original_dim) - h_c

        Returns:
            user_representation: (batch, N, original_dim) - u_h
        """
        h_j = history_repr  # (batch, M, original_dim)
        h_c = candidate_news_representation  # (batch, N, original_dim)

        # Single-branch candidate-guided selection
        if self.attention_mode == 'topic-aware':
            # topic-based attention
            t_c = self._get_topic_embedding(candidate_category, candidate_subCategory)  # (B, N, d_topic)
            t_j = self._get_topic_embedding(user_category, user_subCategory)  # (B, M, d_topic)

            r_h = self.candidate_guided_selection_single_topic(t_c, t_j, h_j, user_history_mask)
        elif self.attention_mode == 'candidate-aware':
            # Feature-based attention
            r_h = self.candidate_guided_selection_single_candidate(h_c, h_j, user_history_mask)
        else:
            raise ValueError(f'Unknown attention mode: {self.attention_mode}')

        # r_h: (batch, N, M, original_dim)

        # Aggregate with the Base user encoder
        r_h_flat = r_h.reshape(batch_size * N, -1, self.original_dim)
        
        mask_flat = user_history_mask.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1)
        
        if user_history_graph is not None:
            G = user_history_graph.size(1)
            graph_flat = user_history_graph.unsqueeze(1).expand(-1, N, -1, -1).reshape(batch_size * N, G, G)
        else:
            graph_flat = None
            
        cat_mask_flat = user_history_category_mask.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1) if user_history_category_mask is not None else None
        cat_indices_flat = user_history_category_indices.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1) if user_history_category_indices is not None else None

        if user_embedding is not None:
            user_embedding_flat = user_embedding.unsqueeze(1).expand(-1, N, -1).reshape(batch_size * N, -1)
        else:
            user_embedding_flat = None

        dummy_candidate = torch.zeros(batch_size * N, 1, self.original_dim).to(r_h.device)

        u_h_flat = self.base_user_encoder.forward(
            r_h_flat, mask_flat,
            None, None, None, None,
            None, None,
            mask_flat,
            graph_flat, cat_mask_flat, cat_indices_flat,
            user_embedding_flat,
            dummy_candidate
        )

        # For multi-view encoders such as PENR, handle the view dimension (simple mean)
        if u_h_flat.dim() == 4:
            u_h_flat = u_h_flat.mean(dim=2)

        if u_h_flat.dim() == 3:
            u_h_flat = u_h_flat.squeeze(1)

        u_h = u_h_flat.reshape(batch_size, N, self.original_dim)  # (batch, N, original_dim)

        return u_h

    def _get_topic_embedding(self, category, subCategory):
        """Helper method to get topic embedding from category and subcategory"""
        cat_emb = self.news_encoder.category_embedding(category)
        subcat_emb = self.news_encoder.subCategory_embedding(subCategory)
        return torch.cat([cat_emb, subcat_emb], dim=-1)

    def candidate_guided_selection_single_topic(self, t_c, t_j, h_j, user_history_mask):
        """
        Single-branch Topic-aware Candidate-guided Selection (I1=False, I2=True)
        Q=t_c (candidate topic), K=t_j (history topic), V=h_j

        Args:
            t_c: (batch_size, N, d_topic) - candidate topic embedding
            t_j: (batch_size, M, d_topic) - history topic embedding
            h_j: (batch_size, M, original_dim) - history features (Value)
            user_history_mask: (batch_size, M)

        Returns:
            r_h: (batch_size, N, M, original_dim)
        """
        N = t_c.size(1)
        M = t_j.size(1)

        # (1) compute alpha (comparing topics)
        Q_proj = self.W_Q_topic(t_c)   # (B, N, H*d_k)
        K_proj = self.W_K_topic(t_j)   # (B, M, H*d_k)
        α = self._multihead_attention(Q_proj, K_proj, user_history_mask, N)  # (B, N, M)

        # (2) Top-K Reweighting
        α_final = self._topk_reweight(α, M)  # (B, N, M)

        # (3) Gated Residual
        r_h = self._gated_residual_single(α_final, h_j, self.W_g_single, N)  # (B, N, M, original_dim)

        return r_h

    def candidate_guided_selection_single_candidate(self, h_c, h_j, user_history_mask):
        """
        Single-branch Candidate-aware Candidate-guided Selection (I1=False, I2=True)
        Q=h_c, K=h_j, V=h_j

        Args:
            h_c: (batch_size, N, original_dim) - candidate features
            h_j: (batch_size, M, original_dim) - history features
            user_history_mask: (batch_size, M)

        Returns:
            r_h: (batch_size, N, M, original_dim)
        """
        N = h_c.size(1)
        M = h_j.size(1)

        # (1) compute alpha
        Q_proj = self.W_Q_h(h_c)  # (B, N, H*d_k)
        K_proj = self.W_K_h(h_j)  # (B, M, H*d_k)
        α = self._multihead_attention(Q_proj, K_proj, user_history_mask, N)  # (B, N, M)

        # (2) Top-K Reweighting
        α_final = self._topk_reweight(α, M)  # (B, N, M)

        # (3) Gated Residual
        r_h = self._gated_residual_single(α_final, h_j, self.W_g_single, N)  # (B, N, M, original_dim)

        return r_h

    def _gated_residual_single(self, α_final, val_vec, W_g, N):
        """
        Gated Residual Connection for single-branch (I1=False, I2=True)

        Args:
            α_final: (batch_size, N, M)
            val_vec: (batch_size, M, original_dim) - value vectors (h_j)
            W_g: nn.Linear or None - gate projection (None if use_gate=False)
            N: number of candidate news

        Returns:
            r: (batch_size, N, M, original_dim) - reweighted history
        """
        val_expanded = val_vec.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, M, original_dim)

        # Branch depending on whether the gate is used
        if self.use_gate and W_g is not None:
            # Original gated residual: r = g_i ⊙ (α_j · h_j) + (1 - g_i) ⊙ h_j
            g_logits = W_g(val_expanded).squeeze(-1) + α_final
            g_i = torch.sigmoid(g_logits)  # (B, N, M)

            g_i_exp = g_i.unsqueeze(-1)
            α_exp = α_final.unsqueeze(-1)

            r = g_i_exp * (α_exp * val_expanded) + (1 - g_i_exp) * val_expanded
        else:
            # Direct attention weighting: r = α_j · h_j
            α_exp = α_final.unsqueeze(-1)  # (B, N, M, 1)
            r = α_exp * val_expanded       # (B, N, M, original_dim)

        return r  # (B, N, M, original_dim)


class CROWN(UserEncoder):
    def __init__(self, news_encoder, config):
        super(CROWN, self).__init__(news_encoder, config)

        self.attention_dim = config.attention_dim
        self.graph_sage = GraphSAGE(in_channels = self.news_embedding_dim,
                                    hidden_channels = self.news_embedding_dim,
                                    num_layers = 1,
                                    out_channels = self.news_embedding_dim,
                                    dropout = config.dropout_rate)

        self.user_node_embedding = nn.Parameter(torch.zeros([config.batch_size * 5, self.news_embedding_dim]))

        self.K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=False)
        self.Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))
        self.affine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)


    def initialize(self):
        nn.init.zeros_(self.user_node_embedding)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)

    def create_bipartite_graph(self, user_history_mask, device):
        num_users, max_history_num = user_history_mask.size()
        row_indices = torch.arange(num_users).view(-1, 1).repeat(1, max_history_num).view(-1).to(device)
        col_indices = torch.arange(max_history_num).view(1, -1).repeat(num_users, 1).view(-1).to(device)
        edge_index = torch.stack([row_indices, col_indices], dim=0)
        return edge_index
        

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):

        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                  # [batch_size, max_history_num, news_embedding_dim]


        history_embedding = torch.cat([history_embedding, self.dropout_(self.user_node_embedding.expand(batch_size, -1, -1))], dim=1)      # [batch_size, max_history_num + 1, news_embedding_dim]

        # Create user-news bipartite graph
        edge_index = self.create_bipartite_graph(user_history_mask, history_embedding.device)
        # GNN convolution
        gcn_feature = self.graph_sage(history_embedding, edge_index)                            # [batch_size, max_history_num + num_users, news_embedding_dim]

        gcn_feature = gcn_feature[:, :self.max_history_num, :]                                  # [batch_size, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1, -1)                 # [batch_size, news_num, max_history_num, news_embedding_dim]
        
        # Attention
        K = self.K(gcn_feature).view([batch_news_num, self.max_history_num, self.attention_dim])            # [batch_size * news_num, max_history_num, attention_dim]
        Q = self.Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])             # [batch_size * news_num, attention_dim, 1]
        a = torch.bmm(K, Q).view([batch_news_num, self.max_history_num]) / self.attention_scalar            # [batch_size * news_num, max_history_num]
        alpha = F.softmax(a, dim=1)                                                                         # [batch_size * news_num, max_history_num]
        # bmm input: [batch_size * news_num, 1, max_history_num]
        # bmm mat2: [batch_size * news_num, max_history_num, news_embedding_dim]
        # bmm out: [batch_size * news_num, 1, news_embedding_dim]
        out = torch.bmm(alpha.unsqueeze(dim=1), gcn_feature.reshape([batch_news_num, self.max_history_num, self.news_embedding_dim]))       # [batch_size * news_num, 1, news_embedding_dim]
        out = out.squeeze(dim=1).view([batch_size, news_num, self.news_embedding_dim])                                                      # [batch_size, news_num, news_embedding_dim]
        
        user_representation = out # self.dropout(F.relu(self.affine(out), inplace=True) + out)                                                    # [batch_size, news_num, news_embedding_dim]

        return user_representation                                                          # [batch_size, news_num, news_embedding_dim]


