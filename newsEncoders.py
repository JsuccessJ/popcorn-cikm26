import math
import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention, MultiHeadSelfAttention, PositionalEncoding, PENR_AdditiveAttention

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for adversarial disentanglement.
    Forward: identity (x → x)
    Backward: gradient reversal (grad → -λ * grad)

    This lets the same popularity_predictor:
    - p_j path: normal gradient → CE↓ (predicts popularity well)
    - f_j path: reversed gradient → predictor gets stronger (CE↓), while the f_j encoder hides popularity (CE↑)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])
    
    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]

    def load_category_embeddings_from_glove(self, category_dict, frozen=True):
        """
        Initialize category embeddings with GloVe
        Args:
            category_dict: {category_name: index} dictionary
            frozen: if True, not trainable (paper setup); if False, trainable
        """
        from torchtext.vocab import GloVe, Vectors

        if hasattr(self, 'config') and self.config.dataset == 'eb-nerd':
            # Danish FastText for EB-NERD dataset
            print("Loading Danish FastText 300d for category embeddings...")
            print("Dataset: EB-NERD(Danish)")
            glove = Vectors(name='cc.da.300.vec', cache='../../glove_danish')
        else:
            # Glove for English datasets(MIND,Adressa,etc.)
            print("Loading Glove 840B 300d for category embeddings...")
            print("Dataset: MIND,Adressa,etc.")
            glove = GloVe(name='840B', dim=300, cache='../../glove')

        category_emb_dim = self.category_embedding.weight.size(1)  # weight.size(): [category_num, category_embedding_dim]
        print(f"Category embedding dimension: {category_emb_dim}")

        # Category name preprocessing function
        def preprocess_category(name):
            """Split compound words and return a list of words
            Examples:
                'foodanddrink' → ['food', 'drink']
                'middleeast' → ['middle', 'east']
                'northamerica' → ['north', 'america']
            """
            # Replace 'and' with a space
            name = name.replace('and', ' ')
            # Collapse duplicate spaces and lowercase
            words = name.lower().split()
            return words

        # Per-category initialization
        success_count = 0
        for category_name, idx in category_dict.items():
            words = preprocess_category(category_name)

            # Collect GloVe vectors
            vectors = []
            for word in words:
                if word in glove.stoi:
                    vectors.append(glove.vectors[glove.stoi[word]])

            if len(vectors) > 0:
                # Compute the mean vector
                avg_vector = torch.stack(vectors).mean(dim=0)  # [300]

                # Scale normalization for Danish FastText
                if hasattr(self, 'config') and self.config.dataset == 'eb-nerd':
                    scaling_factor = 5.77  # use a precomputed value
                    avg_vector = avg_vector * scaling_factor

                # Dimension adjustment
                if category_emb_dim == 300:
                    category_vector = avg_vector
                elif category_emb_dim < 300:
                    # Truncate
                    category_vector = avg_vector[:category_emb_dim]
                else:
                    # Zero padding
                    padding = torch.zeros(category_emb_dim - 300)
                    category_vector = torch.cat([avg_vector, padding])

                # Assign to the embedding
                self.category_embedding.weight.data[idx] = category_vector
                success_count += 1
                print(f"  ✓ '{category_name}' initialized from Glove")
            else:
                # Keep random init when the word is not in GloVe
                print(f"  ✗ '{category_name}' uses random initialization")

        # Set frozen status
        self.category_embedding.weight.requires_grad = not frozen  # when False, keep the GloVe vectors as-is

        status = "frozen" if frozen else "trainable"
        print(f"\nCategory embedding: {success_count}/{len(category_dict)} from Glove ({status})")


    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                           # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNE(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim
        # selective LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.title_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.title_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        self.content_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.content_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        # cross-attention
        self.title_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)
        self.content_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.title_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.title_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.content_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.title_cross_attention.initialize()
        self.content_cross_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                              # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                                        # [batch_size * news_num, max_content_length]
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                                         # [batch_size * news_num]
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                                     # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                              # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                                     # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                                        # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                                 # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])                 # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)                      # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)              # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. selective LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)                                                                      # [batch_size * news_num, hidden_dim * 2]
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)                                                                # [batch_size * news_num, hidden_dim * 2]
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                                      # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                                # [batch_size * news_num, max_content_length, hidden_dim * 2]
        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))        # gate                          # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))  # gate                          # [batch_size * news_num, max_content_length, hidden_dim * 2]
        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)                                                             # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)                                                     # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4]) # [batch_size, news_num, hidden_dim * 4]
        # 5. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                              # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNN(NewsEncoder):
    def __init__(self, config: Config):
        super(CNN, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.cnn_kernel_num])                        # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                    # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])                           # [batch_size, news_num, news_embedding_dim]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class KCNN(NewsEncoder):
    def __init__(self, config: Config):
        super(KCNN, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.entity_embedding_dim = config.entity_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.entity_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.entity_embedding_dim)
        self.context_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.context_embedding_dim)
        with open('entity_embedding-%s.pkl' % config.dataset, 'rb') as entity_embedding_f:
            self.entity_embedding.weight.data.copy_(pickle.load(entity_embedding_f))
        with open('context_embedding-%s.pkl' % config.dataset, 'rb') as context_embedding_f:
            self.context_embedding.weight.data.copy_(pickle.load(context_embedding_f))
        self.M_entity = nn.Linear(self.entity_embedding_dim, self.word_embedding_dim, bias=True)
        self.M_context = nn.Linear(self.context_embedding_dim, self.word_embedding_dim, bias=True)
        self.knowledge_cnn = Conv2D_Pool(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size, 3)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.M_entity.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_entity.bias)
        nn.init.xavier_uniform_(self.M_context.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_context.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word & entity & context embedding
        word_embedding = self.word_embedding(title_text).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                                  # [batch_size * news_num, max_title_length, word_embedding_dim]
        entity_embedding = self.entity_embedding(title_entity).view([batch_news_num, self.max_title_length, self.entity_embedding_dim])                          # [batch_size * news_num, max_title_length, entity_embedding_dim]
        context_embedding = self.context_embedding(title_entity).view([batch_news_num, self.max_title_length, self.context_embedding_dim])                       # [batch_size * news_num, max_title_length, context_embedding_dim]
        W = torch.stack([word_embedding, torch.tanh(self.M_entity(entity_embedding)), torch.tanh(self.M_context(context_embedding))], dim=3).permute(0, 2, 1, 3) # [batch_size * news_num, word_embedding_dim, max_title_length, 3]
        # 2. knowledge-aware CNN
        news_representation = self.knowledge_cnn(W).view([batch_size, news_num, self.cnn_kernel_num])                                                            # [batch_size, news_num, cnn_kernel_num]
        # 3. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                                    # [batch_size, news_num, news_embedding_dim]
        return news_representation


class HDC(NewsEncoder):
    def __init__(self, config: Config):
        super(HDC, self).__init__(config)
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.word_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.word_embedding_dim)
        self.HDC_sequence_length = config.max_title_length + 2
        self.HDC_filter_num = config.HDC_filter_num
        self.dilated_conv1 = nn.Conv1d(in_channels=config.word_embedding_dim, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2, dilation=1)
        self.dilated_conv2 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 1, dilation=2)
        self.dilated_conv3 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 2, dilation=3)
        self.layer_norm1 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm2 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm3 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.news_embedding_dim = None

    def initialize(self):
        super().initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. sequence embeddings
        word_embedding = self.word_embedding(title_text).permute(0, 1, 3, 2)                                                 # [batch_size, news_num, word_embedding_dim, title_length]
        category_embedding = self.category_embedding(category).unsqueeze(dim=3)                                              # [batch_size, news_num, word_embedding_dim, 1]
        subCategory_embedding = self.subCategory_embedding(subCategory).unsqueeze(dim=3)                                     # [batch_size, news_num, word_embedding_dim, 1]
        d0 = torch.cat([category_embedding, subCategory_embedding, word_embedding], dim=3)                                   # [batch_size, news_num, word_embedding_dim, HDC_sequence_length]
        d0 = d0.view([batch_news_num, self.word_embedding_dim, self.HDC_sequence_length])                                    # [batch_size * news_num, word_embedding_dim, HDC_sequence_length]
        # 2. hierarchical dilated convolution
        d1 = F.relu(self.layer_norm1(self.dilated_conv1(d0)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d2 = F.relu(self.layer_norm2(self.dilated_conv2(d1)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d3 = F.relu(self.layer_norm3(self.dilated_conv3(d2)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d0 = d0.view([batch_size, news_num, self.word_embedding_dim, self.HDC_sequence_length])                              # [batch_size, news_num, word_embedding_dim, HDC_sequence_length]
        dL = torch.stack([d1, d2, d3], dim=1).view([batch_size, news_num, 3, self.HDC_filter_num, self.HDC_sequence_length]) # [batch_size, news_num, 3, HDC_filter_num, HDC_sequence_length]
        return (d0, dL)


class NAML(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                  # [batch_size * news_num, max_title_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])                               # [batch_size, news_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)                  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                    # [batch_size, news_num, cnn_kernel_num]
        return news_representation


class PNE(NewsEncoder):
    def __init__(self, config: Config):
        super(PNE, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.personalized_embedding_dim = config.personalized_embedding_dim
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(config.cnn_kernel_num, config.personalized_embedding_dim, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        q_w = F.relu(self.dense(user_embedding), inplace=True).repeat([news_num, 1])                                                # [batch_size * news_num, personalized_embedding_dim]
        news_representation = self.personalizedAttention(c, q_w, mask).view([batch_size, news_num, self.cnn_kernel_num])            # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class DAE(NewsEncoder):
    def __init__(self, config: Config):
        super(DAE, self).__init__(config)
        self.Alpha = config.Alpha
        assert self.Alpha > 0, 'Reconstruction loss weight must be greater than 0'
        self.f1 = nn.Linear(config.word_embedding_dim, config.hidden_dim, bias=True)
        self.f2 = nn.Linear(config.hidden_dim, config.word_embedding_dim, bias=True)
        self.news_embedding_dim = config.hidden_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.f1.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f1.bias)
        nn.init.xavier_uniform_(self.f2.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f2.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        title_mask = title_mask.unsqueeze(dim=3)
        content_mask = content_mask.unsqueeze(dim=3)
        word_embedding = torch.sigmoid(((self.word_embedding(title_text) * title_mask).sum(dim=2) + (self.word_embedding(content_text) * content_mask).sum(dim=2)) \
                         / (title_mask.sum(dim=2, keepdim=False) + content_mask.sum(dim=2, keepdim=False)))           # [batch_size, news_num, word_embedding_dim]
        corrupted_word_embedding = self.dropout_(word_embedding)                                                      # [batch_size, news_num, word_embedding_dim]
        news_representation = torch.sigmoid(self.f1(corrupted_word_embedding))                                        # [batch_size, news_num, news_embedding_dim]
        denoised_word_embedding = torch.sigmoid(self.f2(news_representation))                                         # [batch_size, news_num, word_embedding_dim]
        self.auxiliary_loss = torch.norm(word_embedding - denoised_word_embedding, dim=2, keepdim=False) * self.Alpha # [batch_size, news_num]
        # feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                         # [batch_size, news_num, news_embedding_dim]
        return news_representation


class Inception(NewsEncoder):
    def __init__(self, config: Config):
        super(Inception, self).__init__(config)
        assert config.word_embedding_dim == config.category_embedding_dim and config.word_embedding_dim == config.subCategory_embedding_dim, 'embedding dimension must be the same in the Inception module'
        self.fc1_1 = nn.Linear(config.word_embedding_dim*4, config.hidden_dim, bias=True)
        self.fc1_2 = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.fc1_3 = nn.Linear(config.hidden_dim, config.word_embedding_dim, bias=True)
        self.fc2 = nn.Linear(config.word_embedding_dim*4, config.word_embedding_dim, bias=True)
        self.linear_transform = nn.Linear(config.word_embedding_dim*3, config.word_embedding_dim, bias=True)
        self.news_embedding_dim = config.word_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.fc1_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1_1.bias)
        nn.init.xavier_uniform_(self.fc1_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1_2.bias)
        nn.init.xavier_uniform_(self.fc1_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1_3.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.linear_transform.weight)
        nn.init.zeros_(self.linear_transform.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        title_mask[:, :, 0] = 1   # To avoid zero-length title
        content_mask[:, :, 0] = 1 # To avoid zero-length content
        title_embedding = (self.word_embedding(title_text) * title_mask.unsqueeze(dim=3)).sum(dim=2) / title_mask.sum(dim=2, keepdim=True)         # [batch_size, news_num, word_embedding_dim]
        content_embedding = (self.word_embedding(content_text) * content_mask.unsqueeze(dim=3)).sum(dim=2) / content_mask.sum(dim=2, keepdim=True) # [batch_size, news_num, word_embedding_dim]
        category_embedding = self.category_embedding(category)                                                                                     # [batch_size, news_num, category_embedding_dim]
        subCategory_embedding = self.subCategory_embedding(subCategory)                                                                            # [batch_size, news_num, subCategory_embedding_dim]
        embeddings = torch.cat([title_embedding, content_embedding, category_embedding, subCategory_embedding], dim=2)                             # [batch_size, news_num, embedding_dim * 4]
        subnetwork1 = F.relu(self.fc1_3(F.relu(self.fc1_2(F.relu(self.fc1_1(embeddings), inplace=True)), inplace=True)), inplace=True)             # [batch_size, news_num, embedding_dim]
        subnetwork2 = F.relu(self.fc2(embeddings), inplace=True)                                                                                   # [batch_size, news_num, embedding_dim]
        subnetwork3 = title_embedding + content_embedding + category_embedding + subCategory_embedding                                             # [batch_size, news_num, embedding_dim]
        news_representation = self.linear_transform(torch.cat([subnetwork1, subnetwork2, subnetwork3], dim=2))                                     # [batch_size, news_num, embedding_dim]
        return news_representation


class PLMNewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(PLMNewsEncoder, self).__init__()
        self.config = config
        self.plm_hidden_dim = 768 # Bert-Base hidden size

        # 1. Generic PLM loading (supports DistilBERT variants; removed the forced 8-layer cap)
        from transformers import AutoConfig, AutoModel
        plm_config = AutoConfig.from_pretrained(
            config.plm_model_name,
            output_hidden_states=True # set to track embeddings from all layers
        )
        self.plm = AutoModel.from_pretrained(
            config.plm_model_name,
            config=plm_config,
            cache_dir='plm_cache/'
        )
        self.plm_hidden_dim = self.plm.config.hidden_size



        # 2. PLM Layer Freezing (train only the last N layers)
        self._freeze_plm_layers(config.plm_frozen_layers)

        self.multihead_attention = MultiHeadSelfAttention(
            d_model=self.plm_hidden_dim, # 768
            num_heads=config.head_num, # 20
            head_dim=config.head_dim, # 20
            dropout=config.dropout_rate
        )
        
        # 3. Pooling -> attention
        self.pooling_method = config.plm_pooling
        if self.pooling_method == 'attention':
            self.attention = Attention(
                feature_dim=config.head_num * config.head_dim,  # 400
                attention_dim=config.attention_dim)
        
        # 4. Dropout configuration
        self.dropout =  nn.Dropout(p=config.dropout_rate, inplace=False)

        # 5. Auxiliary loss
        self.auxiliary_loss = None

    def _freeze_plm_layers(self, frozen_layers):
        if frozen_layers <= 0:
            print("No layers frozen. Learning all layers!")
            return

        # Fetch the right layer list for the model type (fully compatible with BERT vs DistilBERT)
        if hasattr(self.plm, 'encoder'):
            layers = self.plm.encoder.layer
        elif hasattr(self.plm, 'transformer'): # DistilBERT naming
            layers = self.plm.transformer.layer
        else:
            print("Warning: Could not find layer structure to freeze.")
            return
            
        total_layers = len(layers)

        # 1. Embedding layer freezing
        for param in self.plm.embeddings.parameters():
            param.requires_grad = False

        # 2. Sequentially freeze the specified number of layers
        actual_frozen = min(frozen_layers, total_layers)
        for layer_idx in range(actual_frozen):
            for param in layers[layer_idx].parameters():
                param.requires_grad = False

        print(f'Frozen {actual_frozen} layers out of {total_layers} in PLM')
        print(f'Trainable layers: {list(range(actual_frozen, total_layers))}')

    

    def _pool_hidden_states(self, hidden_states, attention_mask):
        """
        Pool PLM hidden states into a news embedding

        Args:
            hidden_states: [batch_news_num, seq_len, hidden_dim]
            attention_mask: [batch_news_num, seq_len]

        Returns:
            news_embedding: [batch_news_num, hidden_dim]
        """

        # hidden_states is already extracted and passed in by the subclass
        # PLMMiner: plm_output.last_hidden_state
        # [B*N, seq_len, 768]

        # Multi-head self-attention
        mhsa_output = self.multihead_attention(hidden_states, hidden_states, hidden_states, mask=attention_mask)  # [B*N, seq_len, 400]

    # [B*N, seq_len, 400]
        if self.pooling_method == 'cls':
            # Use the [CLS] token representation, hidden_states: [batch_news_num, seq_len, hidden_dim] -> [B*N, hidden_dim]
            news_embedding = mhsa_output[:, 0, :]  # use mhsa_output

        elif self.pooling_method == 'average':
            # Mean pooling that accounts for the attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(mhsa_output.size()).float()
            sum_hidden = torch.sum(mhsa_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            news_embedding = sum_hidden / sum_mask

        # Attention pooling
        elif self.pooling_method == 'attention':
            news_embedding = self.attention(mhsa_output, mask=attention_mask)
        # [B*N, 400]
        
        else:
            raise ValueError(f'Unknown pooling method: {self.pooling_method}')

        return news_embedding

    def load_category_embeddings_from_glove(self, category_dict, frozen=True):
        """
        Initialize category embeddings with GloVe
        Args:
            category_dict: {category_name: index} dictionary
            frozen: if True, not trainable (paper setup); if False, trainable
        """
        from torchtext.vocab import GloVe, Vectors

        if hasattr(self, 'config') and self.config.dataset == 'eb-nerd':
            # Danish FastText for EB-NERD dataset
            print("Loading Danish FastText 300d for category embeddings...")
            print("Dataset: EB-NERD(Danish)")
            glove = Vectors(name='cc.da.300.vec', cache='../../glove_danish')
        else:
            # Glove for English datasets(MIND,Adressa,etc.)
            print("Loading Glove 840B 300d for category embeddings...")
            print("Dataset: MIND,Adressa,etc.")
            glove = GloVe(name='840B', dim=300, cache='../../glove')

        category_emb_dim = self.category_embedding.weight.size(1)
        print(f"Category embedding dimension: {category_emb_dim}")

        # Category name preprocessing function
        def preprocess_category(name):
            # Replace 'and' with a space
            name = name.replace('and', ' ')
            # Collapse duplicate spaces and lowercase
            words = name.lower().split()
            return words

        # Per-category initialization
        success_count = 0
        for category_name, idx in category_dict.items():
            words = preprocess_category(category_name)

            # Collect GloVe vectors
            vectors = []
            for word in words:
                if word in glove.stoi:
                    vectors.append(glove.vectors[glove.stoi[word]])

            if len(vectors) > 0:
                # Compute the mean vector
                avg_vector = torch.stack(vectors).mean(dim=0)  # [300]

                # Scale normalization for Danish FastText
                if hasattr(self, 'config') and self.config.dataset == 'eb-nerd':
                    scaling_factor = 5.77  # use a precomputed value
                    avg_vector = avg_vector * scaling_factor

                # Dimension adjustment
                if category_emb_dim == 300:
                    category_vector = avg_vector
                elif category_emb_dim < 300:
                    category_vector = avg_vector[:category_emb_dim]
                else:
                    padding = torch.zeros(category_emb_dim - 300)
                    category_vector = torch.cat([avg_vector, padding])

                # Assign to the embedding
                self.category_embedding.weight.data[idx] = category_vector
                success_count += 1
                print(f"  ✓ '{category_name}' initialized from Glove")
            else:
                print(f"  ✗ '{category_name}' uses random initialization")

        # Set frozen status
        self.category_embedding.weight.requires_grad = not frozen

        status = "frozen" if frozen else "trainable"
        print(f"\nCategory embedding: {success_count}/{len(category_dict)} from Glove ({status})")

    def initialize(self):
        """Parameter initialization"""
        if self.pooling_method == 'attention':
            self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise NotImplementedError('Subclass must implement forward()')


class PLMMiner(PLMNewsEncoder):
    """
    PLM-empowered encoder for MINER user encoder
    - PLM for title encoding
    - Category/SubCategory fusion
    - Automatic GloVe initialization for category embeddings (for category-aware attention)
    """

    def __init__(self, config: Config, category_dict: dict):
        super(PLMMiner, self).__init__(config)

        # Use parent's MHSA (768 → 400, heads=20×20)
        # No need to override - parent class already has it

        # Dimension reduction: 400 -> 250
        self.news_compression_dim = 250
        self.reduce_dim = nn.Linear(
            config.head_num * config.head_dim,  # 400
            self.news_compression_dim           # 250
        )

        # Category embeddings (for MINER's category-aware attention)
        self.category_embedding = nn.Embedding(
            num_embeddings=config.category_num,
            embedding_dim=config.category_embedding_dim
        )

        # News embedding dimension (250, no category concat needed for MINER)
        self.news_embedding_dim = self.news_compression_dim  # 250

        # Store category_dict for GloVe initialization
        self.category_dict = category_dict
        self.use_category_glove = config.use_category_glove

    def initialize(self):
        super().initialize()

        # Initialize dimension reduction layer
        nn.init.xavier_uniform_(self.reduce_dim.weight)
        nn.init.zeros_(self.reduce_dim.bias)

        # Default: random uniform initialization for category embeddings
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

        # GloVe initialization for category embeddings (if enabled)
        if self.use_category_glove:
            print("\n" + "="*60)
            print("PLMMiner: Initializing category embeddings with GloVe 840B 300d")
            print("="*60)
            self.load_category_embeddings_from_glove(
                category_dict=self.category_dict,
                frozen=True  # category embeddings must not be trained
            )
            print("="*60 + "\n")

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        """
        Args:
            title_text: [batch_size, news_num, max_title_length] - PLM token IDs
            title_mask: [batch_size, news_num, max_title_length] - PLM attention mask
            category: [batch_size, news_num]
            subCategory: [batch_size, news_num]

        Returns:
            news_representation: [batch_size, news_num, news_embedding_dim]
        """
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        max_title_length = title_text.size(2)
        batch_news_num = batch_size * news_num

        # 1. Reshape for PLM
        title_text = title_text.view([batch_news_num, max_title_length])
        title_mask = title_mask.view([batch_news_num, max_title_length])

        # 2. PLM encoding
        plm_output = self.plm(
            input_ids=title_text, # [B*N, seq_len]
            attention_mask=title_mask, # [B*N, seq_len]
            return_dict=True,
            output_hidden_states=True  # output all layers
        )
        # Use the last transformer hidden layer
        hidden_states = plm_output.hidden_states[-1]  # [B*N, seq_len, 768]

        # 3. Pooling (use parent's implementation: 768 → MHSA → 400)
        news_repr = self._pool_hidden_states(hidden_states, title_mask)  # [B*N, 400]

        # 4. Dimension reduction (400 → 250)
        news_repr = self.reduce_dim(news_repr)  # [B*N, 250]
        news_repr = self.dropout(news_repr)
        news_repr = news_repr.view([batch_size, news_num, self.news_compression_dim])  # [B, N, 250]

        # No category concatenation - MINER uses category as attention bias
        return news_repr  # [B, N, 250]


class PENR(NewsEncoder):
    """
    PENR News Encoder (CIKM'21)
    Three parallel encoders: Title, Abstract, Category
    Final representation via view-level attention pooling
    """
    def __init__(self, config: Config):
        super(PENR, self).__init__(config)
        # Override dropout to avoid in-place operation issues with multi-view architecture
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)

        self.max_title_length = config.max_title_length
        self.max_abstract_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim  # 300
        self.num_heads = getattr(config, 'penr_num_attention_heads', 6)
        self.head_dim = self.word_embedding_dim // self.num_heads  # 50
        self.query_dim = getattr(config, 'penr_attention_query_dim', 200)
        self.news_embedding_dim = self.word_embedding_dim  # 300 (no category fusion)

        # Title Encoder Components
        self.title_pos_encoding = PositionalEncoding(d_model=self.word_embedding_dim, dropout=config.dropout_rate, max_len=self.max_title_length)
        self.title_mhsa = MultiHeadSelfAttention(
            d_model=self.word_embedding_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.dropout_rate
        )
        self.title_ffn_w1 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
        self.title_ffn_w2 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
        self.title_layer_norm1 = nn.LayerNorm(self.word_embedding_dim)
        self.title_layer_norm2 = nn.LayerNorm(self.word_embedding_dim)
        self.title_attention = PENR_AdditiveAttention(self.word_embedding_dim, self.query_dim)

        # Abstract Encoder Components (separate parameters)
        self.abstract_pos_encoding = PositionalEncoding(d_model=self.word_embedding_dim, dropout=config.dropout_rate, max_len=self.max_abstract_length)
        self.abstract_mhsa = MultiHeadSelfAttention(
            d_model=self.word_embedding_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.dropout_rate
        )
        self.abstract_ffn_w1 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
        self.abstract_ffn_w2 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
        self.abstract_layer_norm1 = nn.LayerNorm(self.word_embedding_dim)
        self.abstract_layer_norm2 = nn.LayerNorm(self.word_embedding_dim)
        self.abstract_attention = PENR_AdditiveAttention(self.word_embedding_dim, self.query_dim)

        # Category Encoder Components
        self.category_dense = nn.Linear(config.category_embedding_dim, self.word_embedding_dim, bias=True)

        # View-Level Attention
        self.view_attention = PENR_AdditiveAttention(self.word_embedding_dim, self.query_dim)

    def initialize(self):
        super().initialize()
        # Title encoder
        nn.init.xavier_uniform_(self.title_ffn_w1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.title_ffn_w1.bias)
        nn.init.xavier_uniform_(self.title_ffn_w2.weight)
        nn.init.zeros_(self.title_ffn_w2.bias)
        self.title_attention.initialize()

        # Abstract encoder
        nn.init.xavier_uniform_(self.abstract_ffn_w1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.abstract_ffn_w1.bias)
        nn.init.xavier_uniform_(self.abstract_ffn_w2.weight)
        nn.init.zeros_(self.abstract_ffn_w2.bias)
        self.abstract_attention.initialize()

        # Category encoder
        nn.init.xavier_uniform_(self.category_dense.weight)
        nn.init.zeros_(self.category_dense.bias)

        # View attention
        self.view_attention.initialize()

    def _encode_text(self, text_emb, mask, pos_encoding, mhsa, ffn_w1, ffn_w2, layer_norm1, layer_norm2, attention):
        """
        Shared encoding logic for title and abstract
        Input: text_emb [batch_size, seq_len, 300]
        Output: representation [batch_size, 300]
        """
        # Step 1: Add positional encoding
        x = pos_encoding(text_emb)  # [batch_size, seq_len, 300]

        # Step 2: Multi-Head Self-Attention with residual
        attn_out = mhsa(x, x, x, mask)  # [batch_size, seq_len, 300]
        x = layer_norm1(x + attn_out)  # [batch_size, seq_len, 300]

        # Step 3: Feed-Forward Network with residual (Equation 7)
        ffn_out = F.relu(ffn_w1(x))  # [batch_size, seq_len, 300]
        ffn_out = ffn_w2(ffn_out)  # [batch_size, seq_len, 300]
        m = layer_norm2(x + ffn_out)  # [batch_size, seq_len, 300]

        # Step 4: Word-Level Attention (Equations 8-9)
        r = attention(m, mask)  # [batch_size, 300]

        return r

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num

        # Reshape for processing
        title_mask_flat = title_mask.view([batch_news_num, self.max_title_length])  # [B*N, max_title_len]
        content_mask_flat = content_mask.view([batch_news_num, self.max_abstract_length])  # [B*N, max_abstract_len]

        # 1. Title Encoder
        title_emb = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])  # [B*N, max_title_len, 300]
        r_t = self._encode_text(
            title_emb, title_mask_flat,
            self.title_pos_encoding, self.title_mhsa,
            self.title_ffn_w1, self.title_ffn_w2,
            self.title_layer_norm1, self.title_layer_norm2,
            self.title_attention
        )  # [B*N, 300]

        # 2. Abstract Encoder
        abstract_emb = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_abstract_length, self.word_embedding_dim])  # [B*N, max_abstract_len, 300]
        r_a = self._encode_text(
            abstract_emb, content_mask_flat,
            self.abstract_pos_encoding, self.abstract_mhsa,
            self.abstract_ffn_w1, self.abstract_ffn_w2,
            self.abstract_layer_norm1, self.abstract_layer_norm2,
            self.abstract_attention
        )  # [B*N, 300]

        # 3. Category Encoder
        category_emb = self.category_embedding(category).view([batch_news_num, -1])  # [B*N, category_emb_dim]
        c = self.dropout(F.relu(self.category_dense(category_emb)))  # [B*N, 300]

        # 4. Multi-View Aggregation (Equation 10 + View-Level Attention)
        # Concatenate three views: r^t, r^a, c → R^900
        r_concat = torch.cat([r_t, r_a, c], dim=1)  # [B*N, 900]

        # Reshape to treat as 3 separate views for attention
        views = r_concat.view([batch_news_num, 3, self.word_embedding_dim])  # [B*N, 3, 300]

        # Apply view-level attention pooling
        r = self.view_attention(views)  # [B*N, 300]

        # Reshape to [batch_size, news_num, 300]
        news_representation = r.view([batch_size, news_num, self.news_embedding_dim])

        return news_representation


class POPCORN(NewsEncoder):
    """
    POPCORN News Encoder with Popularity Disentangler (Model-Agnostic)

    Architecture:
        Base News Encoder (selectable: MHSA/NAML/CNE/...)
        → h_j (original news embedding, d)
        → [MLP method - Recommended]
            ├─ h_projection: d → 2d
            ├─ 2-layer MLP decoder_f: 2d → d (f_j, popularity-free)
            └─ 2-layer MLP decoder_p: 2d → d (p_j, popularity-aware)
        → Topic Embedding (category + subcategory, d_topic=100)
        → Popularity Predictor with GRL ([f_j ; topic] or [p_j ; topic] → num_pop_classes)

    Disentangle Methods:
        - 'mlp' (default, recommended): h_projection + 2-layer MLP decoders
          * h: d → 2d (projection)
          * f_j, p_j: each output d dimensions
          * stronger expressiveness, independent decoders

        - 'gated' (legacy): Element-wise gate masking
          * f_j = h_j ⊙ σ(W_f · h_j), p_j = h_j ⊙ σ(W_p · h_j)
          * simple but limited expressiveness

    Inputs:
        - title_text: (batch_size, N, max_title_length)
        - category: (batch_size, N)
        - subCategory: (batch_size, N)

    Outputs:
        - news_representation: (batch_size, N, 2d) - [f_j ; p_j] concatenated
        - (Intermediate results for losses are stored in self.disentangle_outputs)
    """

    def __init__(self, config, category_dict=None):
        super(POPCORN, self).__init__(config)
        self.config = config

        # (1) Base News Encoder selection (Plug-in)
        base_encoder_name = getattr(config, 'popcorn_base_news_encoder', 'MHSA')

        if base_encoder_name == 'MHSA':
            self.base_encoder = MHSA(config)
        elif base_encoder_name == 'NAML':
            self.base_encoder = NAML(config)
        elif base_encoder_name == 'CNE':
            self.base_encoder = CNE(config)
        elif base_encoder_name == 'CNN':
            self.base_encoder = CNN(config)
        elif base_encoder_name == 'CROWN':
            self.base_encoder = CROWN(config)
        elif base_encoder_name == 'PENR':
            self.base_encoder = PENR(config)
        elif base_encoder_name == 'PLMMiner':
            assert category_dict is not None, 'POPCORN with PLMMiner requires category_dict'
            self.base_encoder = PLMMiner(config, category_dict)
        else:
            raise ValueError(f'Unknown base encoder: {base_encoder_name}')

        # Determine the final dimension based on the I1 flag
        original_dim = self.base_encoder.news_embedding_dim  # d
        self.original_dim = original_dim  # store for use in forward()

        if getattr(config, 'use_I1', False):
            # I1=True: enable Popularity Disentangling
            self.news_embedding_dim = 2 * original_dim  # [f_j ; p_j] = 2d (common)

            # (3) Topic Embedding (category/subCategory embeddings inherited from the parent class)
            d_topic = config.category_embedding_dim + config.subCategory_embedding_dim  # 100

            # GRL lambda: adversarial gradient reversal strength (1.0 = full reversal)
            self.grl_lambda = 1.0
            self.leaky_relu = nn.LeakyReLU(0.2)

            # (2) Popularity Disentangler: branch by disentangle_method
            disentangle_method = getattr(config, 'disentangle_method', 'mlp')
            self.disentangle_method = disentangle_method

            if disentangle_method == 'mlp':
                # ========== MLP method: h_projection + 2-layer MLP decoder ==========
                print(f'[POPCORN] Base encoder: {base_encoder_name}')
                print(f'[POPCORN] I1=True, Method=MLP: Original dim: {original_dim}, H projection dim: {2*original_dim}, Final Output dim: {self.news_embedding_dim}')

                # Project h to twice the dimension
                self.h_projection = nn.Linear(original_dim, 2 * original_dim)  # d → 2d
                base_dim = 2 * original_dim

                # 2-layer MLP decoders (f_j, p_j)
                self.decoder_f1 = nn.Linear(base_dim, base_dim)
                self.decoder_f2 = nn.Linear(2 * base_dim, base_dim // 2)  # base_dim//2 = original_dim
                self.decoder_p1 = nn.Linear(base_dim, base_dim)
                self.decoder_p2 = nn.Linear(2 * base_dim, base_dim // 2)

                # (4) Popularity Predictors
                # Input: [p_j ; topic] or [f_j ; topic] = (base_dim//2) + d_topic = original_dim + d_topic
                self.popularity_predictor = nn.Sequential(
                    nn.Linear(base_dim // 2 + d_topic, base_dim // 2 + d_topic),
                    nn.ReLU(),
                    nn.Linear(base_dim // 2 + d_topic, config.popcorn_num_pop_classes)
                )

            elif disentangle_method == 'gated':
                # ========== Gated method: Gate Masking ==========
                print(f'[POPCORN] Base encoder: {base_encoder_name}')
                print(f'[POPCORN] I1=True, Method=Gated: Original dim: {original_dim}, Final Output dim: {self.news_embedding_dim}')

                # Two Gate layers taking the original dim (d) and outputting a 0~1 mask (d)
                self.f_gate_layer = nn.Linear(original_dim, original_dim)
                self.p_gate_layer = nn.Linear(original_dim, original_dim)

                # (4) Popularity Predictors
                # With Gated Masking, f_j and p_j keep original_dim (same as the input)
                # Input: [p_j ; topic] or [f_j ; topic] = original_dim + d_topic
                self.popularity_predictor = nn.Sequential(
                    nn.Linear(original_dim + d_topic, original_dim + d_topic),
                    nn.ReLU(),
                    nn.Linear(original_dim + d_topic, config.popcorn_num_pop_classes)
                )
            else:
                raise ValueError(f'Unknown disentangle_method: {disentangle_method}. Choose from [mlp, gated]')

        else:
            # I1=False: return h_j as-is → d
            self.news_embedding_dim = original_dim
            print(f'[POPCORN] Base encoder: {base_encoder_name}')
            print(f'[POPCORN] I1=False: Output dim (unchanged): {self.news_embedding_dim}')

        # Storage for intermediate results used in loss computation
        self.disentangle_outputs = None
        self.history_disentangle_outputs = None

    def initialize(self):
        super().initialize()
        self.base_encoder.initialize()

        # Initialize the Disentangler and Popularity Predictor only when I1=True
        if getattr(self.config, 'use_I1', False):
            disentangle_method = getattr(self.config, 'disentangle_method', 'mlp')

            if disentangle_method == 'mlp':
                # ========== MLP method initialization ==========
                # Initialize h_projection
                nn.init.xavier_uniform_(self.h_projection.weight)
                nn.init.zeros_(self.h_projection.bias)

                # Initialize MLP decoders
                nn.init.xavier_uniform_(self.decoder_f1.weight)
                nn.init.zeros_(self.decoder_f1.bias)
                nn.init.xavier_uniform_(self.decoder_f2.weight)
                nn.init.zeros_(self.decoder_f2.bias)
                nn.init.xavier_uniform_(self.decoder_p1.weight)
                nn.init.zeros_(self.decoder_p1.bias)
                nn.init.xavier_uniform_(self.decoder_p2.weight)
                nn.init.zeros_(self.decoder_p2.bias)

            elif disentangle_method == 'gated':
                # ========== Gated method initialization ==========
                nn.init.xavier_uniform_(self.f_gate_layer.weight)
                nn.init.zeros_(self.f_gate_layer.bias)

                nn.init.xavier_uniform_(self.p_gate_layer.weight)
                nn.init.zeros_(self.p_gate_layer.bias)

            # Common: initialize the Popularity Predictor
            for layer in self.popularity_predictor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, title_text, title_mask, title_entity,
                content_text, content_mask, content_entity,
                category, subCategory, user_embedding):
        """
        Args:
            title_text: (batch_size, N, max_title_length)
            category: (batch_size, N)

        Returns:
            news_representation:
                - I1=False: (batch_size, N, original_dim) - h_j
                - I1=True: (batch_size, N, 2*original_dim) - [f_j ; p_j] concatenated

        Side Effect:
            Stores intermediate results for loss computation in self.disentangle_outputs (only when I1=True)
        """
        batch_size, N, _ = title_text.size()

        # (1) Base encoding
        h_j = self.base_encoder(title_text, title_mask, title_entity,
                                 content_text, content_mask, content_entity,
                                 category, subCategory, user_embedding)
        # (batch_size, N, original_dim=d)

        # Branch based on the I1 flag
        if not getattr(self.config, 'use_I1', False):
            # I1=False: bypass gating, return h_j as-is
            self.disentangle_outputs = None  # ignored during loss computation
            self.history_disentangle_outputs = None

            # Pass through the base encoder's auxiliary loss (e.g., CROWN)
            if hasattr(self.base_encoder, 'auxiliary_loss') and self.base_encoder.auxiliary_loss is not None:
                self.auxiliary_loss = self.base_encoder.auxiliary_loss
            else:
                self.auxiliary_loss = None

            return h_j  # (batch_size, N, original_dim)

        # I1=True: apply Popularity Disentangling
        disentangle_method = getattr(self.config, 'disentangle_method', 'mlp')

        # (2) Popularity Disentangler: branch by disentangle_method
        if disentangle_method == 'mlp':
            # ========== MLP method: h_projection + 2-layer MLP decoder ==========
            # (2-1) apply h_projection
            h_j = self.h_projection(h_j)  # (batch_size, N, 2*original_dim)
            h_j_flat = h_j.reshape(batch_size * N, -1)  # (batch_size * N, 2*original_dim)

            # (2-2) 2-layer MLP decoder
            d1_f = self.leaky_relu(self.decoder_f1(h_j_flat))  # (batch_size * N, 2*original_dim)
            f_j_flat = self.leaky_relu(self.decoder_f2(torch.cat([d1_f, h_j_flat], dim=-1)))  # (batch_size * N, original_dim)

            d1_p = self.leaky_relu(self.decoder_p1(h_j_flat))  # (batch_size * N, 2*original_dim)
            p_j_flat = self.leaky_relu(self.decoder_p2(torch.cat([d1_p, h_j_flat], dim=-1)))  # (batch_size * N, original_dim)

        elif disentangle_method == 'gated':
            # ========== Gated method: Gate Masking ==========
            # No h_projection; keep the original dimension
            h_j_flat = h_j.reshape(batch_size * N, -1)  # (batch_size * N, original_dim)

            # Generate a Gate mask with a 0~1 weight per feature
            gate_f = torch.sigmoid(self.f_gate_layer(h_j_flat))
            gate_p = torch.sigmoid(self.p_gate_layer(h_j_flat))

            # Filter by element-wise multiplying the original info (h_j_flat) with the mask
            f_j_flat = h_j_flat * gate_f  # (batch_size * N, original_dim)
            p_j_flat = h_j_flat * gate_p  # (batch_size * N, original_dim)

        # (3) Topic Embedding (common)
        category_flat = category.reshape(batch_size * N)  # (batch_size * N,)
        subCategory_flat = subCategory.reshape(batch_size * N)  # (batch_size * N,)
        category_emb_flat = self.category_embedding(category_flat)  # (batch_size * N, 50)
        subCategory_emb_flat = self.subCategory_embedding(subCategory_flat)  # (batch_size * N, 50)
        topic_emb_flat = torch.cat([category_emb_flat, subCategory_emb_flat], dim=-1)  # (batch_size * N, 100)

        # (4) Popularity Prediction (common, but the input dim differs by method)
        # MLP: f_j_flat, p_j_flat = original_dim
        # Gated: f_j_flat, p_j_flat = original_dim
        # So in both methods the input dim is original_dim + d_topic
        input_p = torch.cat([p_j_flat, topic_emb_flat], dim=-1)  # (batch_size * N, original_dim + 100)
        logits_p_flat = self.popularity_predictor(input_p)

        # Apply GRL: insert gradient reversal on the path from f_j to the predictor
        # Forward: f_j passes through unchanged → predictor outputs logits_f normally
        # Backward: predictor θ_pred receives CE↓ gradient (keeps a strong adversary)
        #           f_j encoder θ_f gets reversed by GRL → CE↑ (hide popularity)
        f_j_reversed = GradientReversalFunction.apply(f_j_flat, self.grl_lambda)
        input_f = torch.cat([f_j_reversed, topic_emb_flat], dim=-1)  # (batch_size * N, original_dim + 100)
        logits_f_flat = self.popularity_predictor(input_f)  # (batch_size * N, num_pop_classes)

        # (5) Reshape back to (batch_size, N, dim)
        f_j = f_j_flat.reshape(batch_size, N, -1)  # (batch_size, N, original_dim)
        p_j = p_j_flat.reshape(batch_size, N, -1)  # (batch_size, N, original_dim)
        logits_p = logits_p_flat.reshape(batch_size, N, -1)  # (batch_size, N, num_pop_classes)
        logits_f = logits_f_flat.reshape(batch_size, N, -1)  # (batch_size, N, num_pop_classes)

        # (6) Store intermediate results for loss computation (accessed from trainer.py)
        # Note: the h_j dimension differs by method
        # - MLP: h_j is (batch_size, N, 2*original_dim) - after h_projection
        # - Gated: h_j is (batch_size, N, original_dim) - without h_projection
        self.disentangle_outputs = {
            'f_j': f_j,           # (batch_size, N, original_dim) - for User Encoder and Click Predictor
            'p_j': p_j,           # (batch_size, N, original_dim) - for Click Predictor
            'logits_p': logits_p, # (batch_size, N, num_pop_classes) - for computing L_p
            'logits_f': logits_f, # (batch_size, N, num_pop_classes) - for computing L_a
            'h_j': h_j            # MLP: (batch_size, N, 2d), Gated: (batch_size, N, d) - for computing L_r
        }

        # Pass through the base encoder's auxiliary loss (e.g., CROWN)
        if hasattr(self.base_encoder, 'auxiliary_loss') and self.base_encoder.auxiliary_loss is not None:
            self.auxiliary_loss = self.base_encoder.auxiliary_loss
        else:
            self.auxiliary_loss = None

        # (7) Compatibility with the existing NNR interface: return a single tensor
        # [f_j ; p_j] concatenation → (batch_size, N, 2*original_dim)
        news_representation = torch.cat([f_j, p_j], dim=-1)

        return news_representation


class CategoryPredictor(nn.Module):
    def __init__(self, title_embedding, category_num):
        super(CategoryPredictor, self).__init__()
        self.fc = nn.Linear(title_embedding, category_num)

    def one_hot_encode(self, target, category_num):
        batch_size, _ = target.size()
        one_hot = torch.zeros(batch_size, category_num, device=target.device)
        target = target.long()
        one_hot.scatter_(1, target, 1)
        return one_hot
    # Input: title_intent_embedding             # [batch_size * news_num, intent_embedding_dim]
    # Output: category loss (auxiliary loss)    # [batch_size, news_num]
    def forward(self, title_intent_embedding, targets, category_num):
        category_logits = self.fc(title_intent_embedding)               # [batch_size * news_num, category_num]
        one_hot_targets = self.one_hot_encode(targets, category_num)    # [batch_size * news_num, category_num]
        category_loss = F.cross_entropy(category_logits, one_hot_targets)

        return category_loss


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class CROWN(NewsEncoder):
    def __init__(self, config: Config):
        super(CROWN, self).__init__(config)

        self.max_title_length = config.max_title_length
        self.max_body_length = config.max_abstract_length
        self.max_history_num = config.max_history_num
        self.category_embedding_dim = config.category_embedding_dim
        self.intent_embedding_dim = config.intent_embedding_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        self.category_num = config.category_num
        self.news_embedding_dim = config.intent_embedding_dim + config.category_embedding_dim + config.subCategory_embedding_dim

        # Transformer encoder
        self.title_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_title_length)
        self.body_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_abstract_length)
        title_encoder_layers = nn.TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.crown_feedforward_dim, config.dropout_rate, batch_first=True)
        self.title_transformer = nn.TransformerEncoder(title_encoder_layers, config.crown_num_layers)
        body_encoder_layers = nn.TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.crown_feedforward_dim, config.dropout_rate, batch_first=True)
        self.body_transformer = nn.TransformerEncoder(body_encoder_layers, config.crown_num_layers)

        # ISAB(Induced Set Attention Block) encoder
        self.ISAB = ISAB(dim_in = config.word_embedding_dim,
                         dim_out = config.word_embedding_dim,
                         num_heads = config.crown_isab_num_heads,
                         num_inds = config.crown_isab_num_inds,
                         ln = True)

        self.category_affine = nn.Linear(config.category_embedding_dim + config.subCategory_embedding_dim, config.category_embedding_dim)

        self.intent_num = config.intent_num     # hyperparameter k
        self.alpha = config.crown_alpha

        self.title_intent_attention = Attention(config.intent_embedding_dim, config.attention_dim)
        self.body_intent_attention = Attention(config.intent_embedding_dim, config.attention_dim)
        self.intent_layers = nn.ModuleList([nn.Linear(config.word_embedding_dim
                                                      + config.category_embedding_dim
                                                      , config.intent_embedding_dim, bias=True)
                                            for _ in range(self.intent_num)])

        self.category_predictor = CategoryPredictor(config.intent_embedding_dim, config.category_num)

    def initialize(self):
        super().initialize()
        self.title_intent_attention.initialize()
        self.body_intent_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        # Initialize each intent layer with different weights to learn different embedding for each intent
        for intent_layer in self.intent_layers:
            nn.init.xavier_uniform_(intent_layer.weight)
            nn.init.zeros_(intent_layer.bias)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

        

    # Apply k-FC layer for k-intent disentanglement
    def k_intent_disentangle(self, intent_num, news_embedding):
        k_intent_embeddings = []
        for i in range(intent_num):
            # Apply different linear transformations for each intent
            intent_embedding = F.relu(self.intent_layers[i](news_embedding), inplace=True)      # [batch_size * news_num, intent_embedding_dim]
            # Expand the dimension (axis 1)
            intent_embedding_exp = intent_embedding.unsqueeze(1)
            k_intent_embeddings.append(intent_embedding_exp)
        # Concatenate the k_intent_embeddings along the second axis (axis 1)
        k_intent_embeddings = torch.cat(k_intent_embeddings, dim=1)                             # [batch_size * news_num, intent_length, intent_embedding_dim]

        return k_intent_embeddings

    def similarity_compute(self, title, body):                              # [batch_size * news_num, intent_embedding_dim]
        cosine_similarity = F.cosine_similarity(title, body, dim=1)
        title_body_similarity = (cosine_similarity + 1) / 2.0
        return title_body_similarity

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num

        t_mask = title_mask.view([batch_news_num, self.max_title_length])                                   # [batch_size * news_num, max_title_length]
        b_mask = content_mask.view([batch_news_num, self.max_body_length])                                  # [batch_size * news_num, max_body_length]

        # Word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])          # [batch_size * news_num, max_title_length, word_embedding_dim]
        body_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_body_length, self.word_embedding_dim])          # [batch_size * news_num, max_content_length, word_embedding_dim]

        # Transformer encoding (for adressa)
        title_p = self.title_pos_encoder(title_w)                                                       # [batch_size * news_num, max_title_length, news_embedding_dim]
        title_t = self.title_transformer(title_p)                                                       # [batch_size * news_num, max_title_length, news_embedding_dim]
        title_embedding = title_t.mean(dim=1).view([batch_size * news_num, self.word_embedding_dim])    # [batch_size * news_num, news_embedding_dim]

        body_p = self.body_pos_encoder(body_w)                                                          # [batch_size * news_num, max_content_length, news_embedding_dim]
        body_t = self.body_transformer(body_p)                                                          # [batch_size * news_num, max_content_length, news_embedding_dim]
        body_embedding = body_t.mean(dim=1).view([batch_size * news_num, self.word_embedding_dim])      # [batch_size * news_num, news_embedding_dim]


        # Category-aware intent disentanglement
        category_representation = self.category_affine(torch.cat([self.category_embedding(category),
                                                                  self.subCategory_embedding(subCategory)],
                                                                  dim=2)).view([batch_news_num, self.category_embedding_dim])   # [batch_size * news_num, category_embedding_dim]
        category_aware_title_embedding = torch.cat([title_embedding, category_representation], dim=1)                            # [batch_size * news_num, news_embedding_dim + category_embedding_dim]
        category_aware_body_embedding = torch.cat([body_embedding, category_representation], dim=1)                             # [batch_size * news_num, news_embedding_dim + category_embedding_dim]

        k = self.intent_num
        title_k_intent_embeddings = self.k_intent_disentangle(k, category_aware_title_embedding)              # [batch_size * news_num, intent_length(k), intent_embedding_dim]
        body_k_intent_embeddings = self.k_intent_disentangle(k, category_aware_body_embedding)                # [batch_size * news_num, intent_length(k), intent_embedding_dim]

        # Intent-based Attention
        title_intent_embedding = self.title_intent_attention(title_k_intent_embeddings)              # [batch_size * news_num, intent_embedding_dim]  [batch_size * news_num, 1, intent_length(k)]
        body_intent_embedding = self.body_intent_attention(body_k_intent_embeddings)                  # [batch_size * news_num, intent_embedding_dim]  [batch_size * news_num, 1, intent_length(k)]

        # Category predictor
        target_category = category.view([batch_news_num, 1])
        category_loss = self.category_predictor(title_intent_embedding, target_category, self.category_num)

        self.auxiliary_loss = category_loss * self.alpha

        # Title-Body similarity computation
        title_body_similarity = self.similarity_compute(title_intent_embedding, body_intent_embedding).view([batch_size * news_num, 1])  # [batch_size * news_num, 1]
        news_representation = (title_intent_embedding + title_body_similarity * body_intent_embedding).view([batch_size, news_num, self.intent_embedding_dim])                          # [batch_size, news_num, title(body) intent_embedding_dim]     # average(weighted sum)

        news_representation = self.feature_fusion(news_representation, category, subCategory)                                   # [batch_size, news_num, intent_embedding_dim + cat + subcat]

        return news_representation

