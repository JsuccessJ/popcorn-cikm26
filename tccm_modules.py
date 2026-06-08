"""
TCCM (Time and Content-aware Causal Model) modules.

Paper: "TCCM: Time and Content-Aware Causal Model for Unbiased News Recommendation"
       (CIKM 2023, Chen et al.)
Re-implemented in PyTorch from the original TCCM reference.

This file implements TimeModule and PopularityModule in PyTorch.
They are instantiated only when `config.click_predictor == 'TCCM'`, so including
this file in the project does not affect any other model.
"""
import torch
import torch.nn as nn

from layers import MultiHeadAttention, Attention


class TimeModule(nn.Module):
    """
    Time Module (paper Section 3.2, reference `model.py:103, 143-149`).

    Input:
        elapsed_buckets: LongTensor [B, N], hours between current impression and news publish time
                         (already clamped ≥ 1 in the caller).
    Output:
        s_t: FloatTensor [B, N], recency score = (1 / t_prime)^λ

    Parameters:
        time_embedding(num_buckets, emb_dim) — learnable
        dense: Linear → Tanh → Linear → Linear → Sigmoid → scalar t_prime ∈ (0, 1)
    """

    def __init__(self, config):
        super().__init__()
        self.num_buckets = int(config.tccm_time_buckets)
        self.emb_dim = int(config.tccm_time_emb_dim)
        self.lam = float(config.tccm_lambda)
        self.time_embedding = nn.Embedding(self.num_buckets, self.emb_dim)
        # Keras reference: Dense(64,tanh) → Dense(64) → Dense(1, sigmoid)
        self.fc1 = nn.Linear(self.emb_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def initialize(self):
        nn.init.uniform_(self.time_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, elapsed_buckets: torch.Tensor) -> torch.Tensor:
        # Clamp to valid embedding range
        idx = elapsed_buckets.clamp(min=0, max=self.num_buckets - 1).long()
        emb = self.time_embedding(idx)                  # [B, N, emb_dim]
        h = torch.tanh(self.fc1(emb))                   # [B, N, 64]
        h = self.fc2(h)                                  # [B, N, 64]
        t_prime = torch.sigmoid(self.fc3(h)).squeeze(-1)  # [B, N]  ∈ (0, 1)
        # s_t = (1 / t_prime)^λ. Clamp t_prime to [0.1, 1.0] so s_t ∈ [1, 10^λ]
        # (with λ=2, s_t ∈ [1, 100]). Prevents gradient explosion in early training.
        s_t = (1.0 / t_prime.clamp(min=0.1, max=1.0)).pow(self.lam)  # [B, N]
        return s_t


class PopularityModule(nn.Module):
    """
    Popularity Module (paper Section 3.2, reference `model.py:102, 113-142`).

    Input (word+entity mode):
        word_ctr_ids:   LongTensor [B, N, L]  — discretized CTR bucket ids of title words
        entity_ctr_ids: LongTensor [B, N, E]  — discretized CTR bucket ids of news entities
    Input (word_only mode):
        word_ctr_ids only; entity_ctr_ids=None

    Output:
        s_p: FloatTensor [B, N]  ∈ (0, 1)

    Architecture (word+entity, reference):
        pop_embedding (shared for word and entity CTR ids)
        word_self  = MHSA(word, word, word)
        word_cross = MHCA(word, entity, entity)                  # Q=word
        word_vec   = AttLayer(word_self + word_cross)
        entity_self  = MHSA(entity, entity, entity)
        entity_cross = MHCA(entity, word, word)                  # Q=entity
        entity_vec   = AttLayer(entity_self + entity_cross)
        fused_vec    = AttLayer(stack([word_vec, entity_vec]))
        s_p = sigmoid(Dense(fused_vec))
    """

    def __init__(self, config, content_mode: str = 'word+entity'):
        super().__init__()
        assert content_mode in ('word+entity', 'word_only')
        self.mode = content_mode
        self.pop_buckets = int(config.tccm_pop_buckets)
        self.emb_dim = int(config.tccm_pop_emb_dim)
        self.num_heads = int(config.tccm_attention_heads)
        self.head_dim = int(config.tccm_attention_head_dim)
        self.d = self.num_heads * self.head_dim  # 400 (paper default 20*20)
        self.L = int(config.max_title_length)
        self.E = int(config.tccm_max_entities)
        self.attention_dim = int(config.attention_dim)  # reuse project's attention_dim (default 200)
        self.dropout = nn.Dropout(p=float(config.dropout_rate))

        # Shared popularity embedding (reference model.py:102)
        self.pop_embedding = nn.Embedding(self.pop_buckets, self.emb_dim)

        # Word-side attention
        self.word_self = MultiHeadAttention(self.num_heads, self.emb_dim,
                                            self.L, self.L, self.head_dim, self.head_dim)
        self.word_att = Attention(self.d, self.attention_dim)

        if content_mode == 'word+entity':
            self.entity_self = MultiHeadAttention(self.num_heads, self.emb_dim,
                                                  self.E, self.E, self.head_dim, self.head_dim)
            self.word_cross = MultiHeadAttention(self.num_heads, self.emb_dim,
                                                  self.L, self.E, self.head_dim, self.head_dim)
            self.entity_cross = MultiHeadAttention(self.num_heads, self.emb_dim,
                                                   self.E, self.L, self.head_dim, self.head_dim)
            self.entity_att = Attention(self.d, self.attention_dim)
            self.fuse_att = Attention(self.d, self.attention_dim)

        # Dense stack → 1d score (reference model.py:139-142)
        self.fc1 = nn.Linear(self.d, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def initialize(self):
        nn.init.uniform_(self.pop_embedding.weight, -0.1, 0.1)
        # Zero out the <PAD>=0 bucket so padded positions contribute nothing
        with torch.no_grad():
            self.pop_embedding.weight[0].zero_()
        self.word_self.initialize()
        self.word_att.initialize()
        if self.mode == 'word+entity':
            self.entity_self.initialize()
            self.word_cross.initialize()
            self.entity_cross.initialize()
            self.entity_att.initialize()
            self.fuse_att.initialize()
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, word_ctr_ids: torch.Tensor, entity_ctr_ids: torch.Tensor = None) -> torch.Tensor:
        """
        word_ctr_ids : [B, N, L]
        entity_ctr_ids : [B, N, E] or None
        returns s_p : [B, N] ∈ (0, 1)
        """
        B, N, L = word_ctr_ids.shape
        # Embed words: [B, N, L, emb_dim] → flatten batch*news to feed MHSA
        w_emb = self.pop_embedding(word_ctr_ids).view(B * N, L, self.emb_dim)
        w_emb = self.dropout(w_emb)

        # MHSA(word)
        w_self = self.word_self(w_emb, w_emb, w_emb)                # [B*N, L, d]

        if self.mode == 'word+entity' and entity_ctr_ids is not None:
            E = entity_ctr_ids.size(-1)
            e_emb = self.pop_embedding(entity_ctr_ids).view(B * N, E, self.emb_dim)
            e_emb = self.dropout(e_emb)
            # MHCA: word attends to entity
            w_cross = self.word_cross(w_emb, e_emb, e_emb)           # [B*N, L, d]
            # MHCA: entity attends to word
            e_cross = self.entity_cross(e_emb, w_emb, w_emb)         # [B*N, E, d]
            e_self = self.entity_self(e_emb, e_emb, e_emb)            # [B*N, E, d]
            # Add + Attention pooling (reference model.py:122-132)
            w_combined = self.dropout(w_self + w_cross)
            e_combined = self.dropout(e_self + e_cross)
            w_vec = self.word_att(w_combined)                         # [B*N, d]
            e_vec = self.entity_att(e_combined)                       # [B*N, d]
            # Stack → pooled news-level vector
            fused = torch.stack([w_vec, e_vec], dim=1)                # [B*N, 2, d]
            fused = self.dropout(fused)
            pooled = self.fuse_att(fused)                             # [B*N, d]
        else:
            # word_only mode: just MHSA(word) → AttLayer
            w_combined = self.dropout(w_self)
            pooled = self.word_att(w_combined)                        # [B*N, d]

        # Dense stack → sigmoid scalar
        h = torch.tanh(self.fc1(pooled))
        h = self.fc2(h)
        h = self.fc3(h)
        s_p = torch.sigmoid(self.fc4(h)).squeeze(-1)                   # [B*N]
        return s_p.view(B, N)
