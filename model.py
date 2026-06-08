from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import newsEncoders
import userEncoders
import variantEncoders


class Model(nn.Module):
    def __init__(self, config: Config, category_dict: dict = None, corpus=None):
        super(Model, self).__init__()
        self.config = config
        # TCCM click_predictor needs the corpus object (for publish_time / CTR tables).
        # Other click_predictors ignore this kwarg → full backward compatibility.
        self._tccm_corpus = corpus
        # POPCORN has base encoder
        if config.news_encoder == 'POPCORN' and config.user_encoder == 'POPCORN':
            base_news_encoder = getattr(config, 'popcorn_base_news_encoder', 'MHSA')
            base_user_encoder = getattr(config, 'popcorn_base_user_encoder', 'ATT')
            self.model_name = f'POPCORN-{base_news_encoder}-{base_user_encoder}'
        else:
            self.model_name = config.news_encoder + '-' + config.user_encoder

        # For main experiments of news encoding
        if config.news_encoder == 'PLMMiner':
            assert category_dict is not None, 'PLMMiner requires category_dict'
            self.news_encoder = newsEncoders.PLMMiner(config, category_dict)
        elif config.news_encoder == 'CNE':
            self.news_encoder = newsEncoders.CNE(config)
        elif config.news_encoder == 'CNN':
            self.news_encoder = newsEncoders.CNN(config)
        elif config.news_encoder == 'MHSA':
            self.news_encoder = newsEncoders.MHSA(config)
        elif config.news_encoder == 'KCNN':
            self.news_encoder = newsEncoders.KCNN(config)
        elif config.news_encoder == 'HDC':
            self.news_encoder = newsEncoders.HDC(config)
        elif config.news_encoder == 'NAML':
            self.news_encoder = newsEncoders.NAML(config)
        elif config.news_encoder == 'PNE':
            self.news_encoder = newsEncoders.PNE(config)
        elif config.news_encoder == 'DAE':
            self.news_encoder = newsEncoders.DAE(config)
        elif config.news_encoder == 'Inception':
            self.news_encoder = newsEncoders.Inception(config)
        elif config.news_encoder == 'PENR':
            self.news_encoder = newsEncoders.PENR(config)
        elif config.news_encoder == 'POPCORN':
            base_news_encoder = getattr(config, 'popcorn_base_news_encoder', 'MHSA')
            if base_news_encoder == 'PLMMiner':
                assert category_dict is not None, 'POPCORN with PLMMiner requires category_dict'
                self.news_encoder = newsEncoders.POPCORN(config, category_dict)
            else:
                self.news_encoder = newsEncoders.POPCORN(config)
        elif config.news_encoder == 'CROWN':
            self.news_encoder = newsEncoders.CROWN(config)
        # For ablations of news encoding _ not used
        elif config.news_encoder == 'NAML_Title':
            self.news_encoder = variantEncoders.NAML_Title(config)
        elif config.news_encoder == 'NAML_Content':
            self.news_encoder = variantEncoders.NAML_Content(config)
        elif config.news_encoder == 'CNE_Title':
            self.news_encoder = variantEncoders.CNE_Title(config)
        elif config.news_encoder == 'CNE_Content':
            self.news_encoder = variantEncoders.CNE_Content(config)
        elif config.news_encoder == 'CNE_wo_CS':
            self.news_encoder = variantEncoders.CNE_wo_CS(config)
        elif config.news_encoder == 'CNE_wo_CA':
            self.news_encoder = variantEncoders.CNE_wo_CA(config)
        else:
            raise Exception(config.news_encoder + 'is not implemented')

        # For main experiments of user encoding
        if config.user_encoder == 'SUE':
            self.user_encoder = userEncoders.SUE(self.news_encoder, config)
        elif config.user_encoder == 'LSTUR':
            self.user_encoder = userEncoders.LSTUR(self.news_encoder, config)
        elif config.user_encoder == 'MHSA':
            self.user_encoder = userEncoders.MHSA(self.news_encoder, config)
        elif config.user_encoder == 'ATT':
            self.user_encoder = userEncoders.ATT(self.news_encoder, config)
        elif config.user_encoder == 'CATT':
            self.user_encoder = userEncoders.CATT(self.news_encoder, config)
        elif config.user_encoder == 'FIM':
            self.user_encoder = userEncoders.FIM(self.news_encoder, config)
        elif config.user_encoder == 'PUE':
            self.user_encoder = userEncoders.PUE(self.news_encoder, config)
        elif config.user_encoder == 'GRU':
            self.user_encoder = userEncoders.GRU(self.news_encoder, config)
        elif config.user_encoder == 'OMAP':
            self.user_encoder = userEncoders.OMAP(self.news_encoder, config)
        elif config.user_encoder == 'MINER':
            self.user_encoder = userEncoders.MINER(self.news_encoder, config)
        elif config.user_encoder == 'PENR':
            self.user_encoder = userEncoders.PENR(self.news_encoder, config)
        elif config.user_encoder == 'POPCORN':
            self.user_encoder = userEncoders.POPCORN(self.news_encoder, config)
        elif config.user_encoder == 'CROWN':
            self.user_encoder = userEncoders.CROWN(self.news_encoder, config)
        # For ablations of user encoding _ not used
        elif config.user_encoder == 'SUE_wo_GCN':
            self.user_encoder = variantEncoders.SUE_wo_GCN(self.news_encoder, config)
        elif config.user_encoder == 'SUE_wo_HCA':
            self.user_encoder = variantEncoders.SUE_wo_HCA(self.news_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')

        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(p=config.dropout_rate)
        if config.user_encoder == 'LSTUR':
            self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=self.news_embedding_dim)
            self.use_user_embedding = True
        elif config.news_encoder == 'PNE' or config.user_encoder == 'PUE':
            self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=config.user_embedding_dim)
            self.use_user_embedding = True
        elif config.user_encoder == 'POPCORN' and getattr(config, 'popcorn_base_user_encoder', 'ATT') == 'LSTUR':
            # POPCORN user encoder with LSTUR base encoder also needs user_embedding
            # Use half dimension since LSTUR operates on f or p separately (not concatenated)
            if getattr(config, 'use_I1', True):
                # I1=True
                self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=self.news_embedding_dim // 2)
            else:
                # I1=False
                self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=self.news_embedding_dim)
            self.use_user_embedding = True
        else:
            self.use_user_embedding = False
        if config.news_encoder == 'HDC' or config.user_encoder == 'FIM':
            assert config.news_encoder == 'HDC' and config.user_encoder == 'FIM', 'HDC and FIM must be paired and can not be used alone'
            assert config.click_predictor == 'FIM', 'For the model FIM, the click predictor must be specially set as \'FIM\''
        
        # click_prdictor, when using popcorn, the click_predictor is POPCORN
        self.click_predictor = config.click_predictor
        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.news_embedding_dim * 2, out_features=self.news_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.news_embedding_dim // 2, out_features=1, bias=True)
        elif self.click_predictor == 'PENR':
            # PENR Click Predictor Components
            assert config.user_encoder == 'PENR', 'PENR click predictor requires PENR user encoder'
            self.num_views = getattr(config, 'penr_num_interest_views', 5)
            # Bilinear interaction (Equation 14)
            self.W_b = nn.Parameter(torch.zeros(self.news_embedding_dim, self.news_embedding_dim))
            self.b_bilinear = nn.Parameter(torch.zeros(1))
            # FFN for aggregating 5-view scores (Equation 15)
            self.penr_ffn = nn.Linear(self.num_views, 1, bias=True)
            # Popularity Predictor (Equation 16)
            self.popularity_predictor = nn.Sequential(
                nn.Linear(self.news_embedding_dim, self.news_embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(self.news_embedding_dim // 2, 1),
                nn.Sigmoid()
            )
            # Trainable parameters for final CTR (Equation 19)
            self.mu = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))  # μ: popularity influence weight
            self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # γ: popularity score scaling
        elif self.click_predictor == 'POPCORN':
            # POPCORN Contrastive Interest Matching (Model-Agnostic)
            assert config.user_encoder == 'POPCORN', 'POPCORN click predictor requires POPCORN user encoder'
            assert config.news_encoder == 'POPCORN', 'POPCORN click predictor requires POPCORN news encoder'

            # Model-agnostic: news_embedding_dim is determined dynamically by the base encoder
            # MHSA: 500, NAML: 400, CNE: 900
            self.popcorn_d_half = self.news_embedding_dim // 2  # f, p, u all have base_dim/2 dimensions

            # ===== POPCORN + TCCM add-on (opt-in) =====
            # If use_tccm_addon=True, register TCCM's TimeModule / PopularityModule / activity_gate
            # and CTR lookup buffers so the POPCORN forward can fuse s_t, s_p, g on top of its score.
            # When flag is off, nothing below runs → existing POPCORN runs are byte-identical.
            self.use_tccm_addon = bool(getattr(config, 'use_tccm_addon', False))
            if self.use_tccm_addon:
                from tccm_modules import TimeModule, PopularityModule
                assert corpus is not None, \
                    "POPCORN + use_tccm_addon requires mind_corpus: Model(config, category_dict, corpus=mind_corpus)"
                # Activity gate takes user_representation (dim = news_embedding_dim) → require I1+I3
                assert getattr(config, 'use_I1', False) and getattr(config, 'use_I3', False), \
                    "POPCORN + use_tccm_addon requires --use_I1 True --use_I3 True (so user_rep dim == news_embedding_dim)"
                self.tccm_time = TimeModule(config)
                _tccm_mode_addon = getattr(config, 'tccm_content_mode', 'word+entity')
                self.tccm_pop = PopularityModule(config, content_mode=_tccm_mode_addon)
                self.tccm_activity_gate = nn.Sequential(
                    nn.Linear(self.news_embedding_dim, 128), nn.Tanh(),
                    nn.Linear(128, 64),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
                self.register_buffer('tccm_news_publish_time', torch.from_numpy(corpus.news_publish_time).long())
                self.register_buffer('tccm_word_ctr_table', torch.from_numpy(corpus.word_ctr_table).long())
                # In word_only mode the entity buffers are not registered (forward also looks up entities only in the mode branch).
                if _tccm_mode_addon == 'word+entity':
                    self.register_buffer('tccm_news_entity_indices', torch.from_numpy(corpus.news_entity_indices).long())
                    self.register_buffer('tccm_entity_ctr_table', torch.from_numpy(corpus.entity_ctr_table).long())
                self.register_buffer('tccm_sp_train_mean', torch.tensor(0.5, dtype=torch.float32))
                self.tccm_ctr_window_hours = int(config.tccm_ctr_window_hours)
                self.tccm_real_publish_time = bool(getattr(corpus, 'tccm_real_publish_time', False))
                self.tccm_elapsed_div = 24 if self.tccm_real_publish_time else 1
                if self.tccm_real_publish_time and corpus.publish_time_valid is not None:
                    self.register_buffer('tccm_publish_time_valid',
                                         torch.from_numpy(corpus.publish_time_valid).bool())
        elif self.click_predictor == 'FIM':
            # compute the output size of 3D convolution and pooling
            def compute_convolution_pooling_output_size(input_size):
                conv1_size = input_size - config.conv3D_kernel_size_first + 1
                pool1_size = (conv1_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
                conv2_size = pool1_size - config.conv3D_kernel_size_second + 1
                pool2_size = (conv2_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
                return pool2_size
            feature_size = compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                           compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                           compute_convolution_pooling_output_size(config.max_history_num) * \
                           config.conv3D_filter_num_second
            self.fc = nn.Linear(in_features=feature_size, out_features=1, bias=True)
        elif self.click_predictor == 'TCCM':
            # TCCM (Time and Content-aware Causal Model, CIKM 2023)
            # Modules and buffers are created here ONLY when TCCM is selected.
            # No impact on other click_predictors' parameter initialization or RNG consumption.
            from tccm_modules import TimeModule, PopularityModule
            assert corpus is not None, \
                "TCCM click_predictor requires the mind_corpus to be passed: Model(config, category_dict, corpus=mind_corpus)"
            self.tccm_time = TimeModule(config)
            _tccm_mode = getattr(config, 'tccm_content_mode', 'word+entity')
            self.tccm_pop = PopularityModule(config, content_mode=_tccm_mode)
            # Per-user activity gate g(u) ∈ (0,1) — reference TCCM/model.py:240-246.
            # score = 2·g·s_m + 2·(1−g)·(s_p·s_t) at training, g·s_m + (1−g)·(s_p·s_t) at inference.
            self.tccm_activity_gate = nn.Sequential(
                nn.Linear(self.news_embedding_dim, 128), nn.Tanh(),
                nn.Linear(128, 64),
                nn.Linear(64, 1), nn.Sigmoid(),
            )
            # Lookup tables as non-trainable buffers (moved to device with model.to())
            self.register_buffer('tccm_news_publish_time', torch.from_numpy(corpus.news_publish_time).long())
            self.register_buffer('tccm_word_ctr_table', torch.from_numpy(corpus.word_ctr_table).long())
            # In word_only mode the entity buffers are not registered, saving GPU memory.
            # forward already looks up entities only in the self.tccm_pop.mode == 'word+entity' branch.
            if _tccm_mode == 'word+entity':
                self.register_buffer('tccm_news_entity_indices', torch.from_numpy(corpus.news_entity_indices).long())
                self.register_buffer('tccm_entity_ctr_table', torch.from_numpy(corpus.entity_ctr_table).long())
            # Intervention target: replaced with train-set mean after training (default 0.5)
            self.register_buffer('tccm_sp_train_mean', torch.tensor(0.5, dtype=torch.float32))
            self.tccm_ctr_window_hours = int(config.tccm_ctr_window_hours)
            self.tccm_real_publish_time = bool(getattr(corpus, 'tccm_real_publish_time', False))
            self.tccm_elapsed_div = 24 if self.tccm_real_publish_time else 1
            if self.tccm_real_publish_time and corpus.publish_time_valid is not None:
                self.register_buffer('tccm_publish_time_valid',
                                     torch.from_numpy(corpus.publish_time_valid).bool())

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])
        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)
        elif self.click_predictor == 'PENR':
            nn.init.xavier_uniform_(self.W_b)
            nn.init.zeros_(self.b_bilinear)
            nn.init.xavier_uniform_(self.penr_ffn.weight)
            nn.init.zeros_(self.penr_ffn.bias)
            for layer in self.popularity_predictor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        elif self.click_predictor == 'POPCORN':
            # Initialize TCCM add-on modules if enabled (no-op otherwise)
            if getattr(self, 'use_tccm_addon', False):
                self.tccm_time.initialize()
                self.tccm_pop.initialize()
                for layer in self.tccm_activity_gate:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        elif self.click_predictor == 'FIM':
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        elif self.click_predictor == 'TCCM':
            # Xavier init for TimeModule and PopularityModule
            self.tccm_time.initialize()
            self.tccm_pop.initialize()
            # Xavier init for activity gate (reference TCCM uses Keras default glorot_uniform)
            for layer in self.tccm_activity_gate:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, user_ID, user_category=None, user_subCategory=None, user_title_text=None, user_title_mask=None, user_title_entity=None, user_content_text=None, user_content_mask=None, user_content_entity=None, user_history_mask=None, user_history_graph=None, user_history_category_mask=None, user_history_category_indices=None, \
                      news_category=None, news_subCategory=None, news_title_text=None, news_title_mask=None, news_title_entity=None, news_content_text=None, news_content_mask=None, news_content_entity=None, \
                      news_indices_for_tccm=None, news_current_time=None):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # candidate news encoding                                                                                                        # [batch_size, news_embedding_dim]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding) # [batch_size, 1 + negative_sample_num, news_embedding_dim]

        # POPCORN: save the candidate news disentangle_outputs (before they are overwritten by the user_encoder)
        # because calling news_encoder again inside the user_encoder overwrites them with the history news outputs
        if self.click_predictor == 'POPCORN':
            self.candidate_disentangle_outputs = self.news_encoder.disentangle_outputs

        # MINER user encoder requires news_category for category-aware attention
        if self.user_encoder.__class__.__name__ == 'MINER':
            user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                    user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation, news_category)
        # POPCORN user encoder requires news_category & news_subCategory for topic-aware attention (t_c)
        # history news encoding
        elif self.user_encoder.__class__.__name__ == 'POPCORN':
            user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                    user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation, news_category, news_subCategory)
        else:
            user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                    user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)                           # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        if self.click_predictor == 'dot_product':
            # I1=True, I3=False: use only f to compute f_u·f_c
            # I1=False: use the full h
            if hasattr(self.config, 'use_I1') and self.config.use_I1:
                # Check if user/news encoders are POPCORN
                if (hasattr(self, 'news_encoder') and self.news_encoder.__class__.__name__ == 'POPCORN' and
                    hasattr(self, 'user_encoder') and self.user_encoder.__class__.__name__ == 'POPCORN'):
                    # I1=True: news/user representations may contain f and p
                    # Determine dimension based on news_embedding_dim
                    news_dim = news_representation.size(-1)
                    user_dim = user_representation.size(-1)

                    # If both dimensions are equal and news encoder has I1=True
                    if news_dim == self.news_embedding_dim and news_dim % 2 == 0:
                        # I1=True case: extract f only
                        d_half = news_dim // 2
                        f_c = news_representation[:, :, :d_half]  # (B, N, d_half)

                        # User representation dimension check
                        if user_dim == d_half:
                            # I3=False: user_representation is f_u only
                            f_u = user_representation  # (B, N, d_half)
                        elif user_dim == news_dim:
                            # I3=True: user_representation is [f_u ; p_u], extract f_u
                            f_u = user_representation[:, :, :d_half]  # (B, N, d_half)
                        else:
                            # Unexpected dimension, fallback to full dot product
                            logits = (user_representation * news_representation).sum(dim=2)
                            return logits

                        logits = (f_u * f_c).sum(dim=-1)  # (B, N)
                    else:
                        # Fallback: standard dot product
                        logits = (user_representation * news_representation).sum(dim=2)
                else:
                    # Not POPCORN encoders, standard dot product
                    logits = (user_representation * news_representation).sum(dim=2)
            else:
                # I1=False: standard dot product
                logits = (user_representation * news_representation).sum(dim=2)
        elif self.click_predictor == 'mlp':
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, news_representation], dim=2)), inplace=True))
            logits = self.out(context).squeeze(dim=2)
        elif self.click_predictor == 'PENR':
            # PENR Click Prediction with Popularity
            batch_size, news_num, num_views, emb_dim = user_representation.size()  # [B, N, 5, 300]

            # Equation 14: Bilinear interaction for each view
            # p_b,i = u_i^T W_b r_c + b
            news_rep_expanded = news_representation.unsqueeze(2).expand(-1, -1, num_views, -1)  # [B, N, 5, 300]

            # For each view: u_i^T @ W_b @ r_c
            # Reshape for batch matrix multiplication
            u_flat = user_representation.reshape(batch_size * news_num * num_views, emb_dim)  # [B*N*5, 300]
            r_flat = news_rep_expanded.reshape(batch_size * news_num * num_views, emb_dim)  # [B*N*5, 300]

            # u_i @ W_b
            u_W_b = torch.matmul(u_flat, self.W_b)  # [B*N*5, 300]
            # (u_i @ W_b) @ r_c
            p_b_flat = (u_W_b * r_flat).sum(dim=1) + self.b_bilinear  # [B*N*5]
            p_b = p_b_flat.reshape(batch_size, news_num, num_views)  # [B, N, 5]

            # Equation 15: Aggregate multi-view scores with FFN
            y_hat = torch.sigmoid(self.penr_ffn(p_b).squeeze(dim=2))  # [B, N]

            # Equation 16: Predict popularity for all news
            p_hat = self.popularity_predictor(news_representation).squeeze(dim=2)  # [B, N]

            # Store predicted popularity for computing popularity loss in trainer
            self.predicted_popularity = p_hat  # [B, N]

            # Equation 18: Calculate user's attention to popular news
            # a_u = (1/n) Σ p̂_h for news in browsing history
            history_repr = self.news_encoder(
                user_title_text, user_title_mask, user_title_entity,
                user_content_text, user_content_mask, user_content_entity,
                user_category, user_subCategory, user_embedding
            )  # [B, max_history_num, 300]

            p_hat_history = self.popularity_predictor(history_repr).squeeze(dim=2)  # [B, max_history_num]
            # Masked average (only count actual history items)
            p_hat_history_masked = p_hat_history * user_history_mask  # [B, max_history_num]
            history_count = user_history_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            a_u = p_hat_history_masked.sum(dim=1, keepdim=True) / history_count  # [B, 1]
            a_u = a_u.expand(-1, news_num)  # [B, N]

            # Equation 19: Final CTR = (1 - μa_u)ŷ + μa_u γp̂_c
            mu_clamped = torch.sigmoid(self.mu)  # Ensure μ ∈ [0,1]
            CTR = (1 - mu_clamped * a_u) * y_hat + mu_clamped * a_u * self.gamma * p_hat  # [B, N]

            # Return logits (before sigmoid for BCE loss)
            logits = torch.log(CTR / (1 - CTR + 1e-8))  # Inverse sigmoid

        elif self.click_predictor == 'POPCORN':
            # POPCORN Popularity-discounted Interest Matching
            use_I1 = getattr(self.config, 'use_I1', False)
            use_I3 = getattr(self.config, 'use_I3', False)

            # split news_representation (when I1 is enabled: [f_c ; p_c])
            if use_I1:
                d_half_c = news_representation.size(-1) // 2
                f_c = news_representation[:, :, :d_half_c]
                p_c = news_representation[:, :, d_half_c:]
            else:
                f_c = news_representation
                p_c = None

            # split user_representation (when I1 & I3 are enabled: [f_u ; p_u])
            if use_I1 and use_I3:
                d_half_u = user_representation.size(-1) // 2
                f_u = user_representation[:, :, :d_half_u]
                p_u = user_representation[:, :, d_half_u:]
            else:
                f_u = user_representation
                p_u = None

            # S_I: Interest Matching Score (f_u · f_c)
            # verify that the dimensions of f_u and f_c match (in the ablation study, with only I1 on they should match, e.g. 500 vs 500)
            S_I = (f_u * f_c).sum(dim=-1)  # (batch_size, N)

            # S_P: Popularity Matching Score & Discounting (only when both p_u and p_c exist)
            if p_u is not None and p_c is not None:
                # 1. Apply Cosine Similarity instead of dot-product (range: [-1, 1])
                S_P = F.cosine_similarity(p_u, p_c, dim=-1)  # (batch_size, N)

                # 2. Apply an additive penalty (prevents sign inversion of S_I and preserves ordering)
                # use the pop_penalty_weight parameter from config as beta
                beta = getattr(self.config, 'pop_penalty_weight', 5.0)

                penalty = beta * torch.sigmoid(self.config.popcorn_alpha * S_P)
                logits = S_I - penalty
            else:
                logits = S_I

            # ===== POPCORN + TCCM add-on =====
            # Fuse TCCM's time/content-popularity/activity-gate into the POPCORN score.
            # Runs ONLY when flag is on; existing POPCORN behavior untouched otherwise.
            if getattr(self, 'use_tccm_addon', False):
                assert news_indices_for_tccm is not None and news_current_time is not None, \
                    'POPCORN + use_tccm_addon requires news_indices_for_tccm and news_current_time'
                S_popcorn = logits   # preserve original POPCORN score as s_m

                # (B) Time score s_t — TCCM TimeModule over elapsed (hours by default, days when real-publish-time mode)
                publish_t = self.tccm_news_publish_time[news_indices_for_tccm]       # [B, N]
                cur_t = news_current_time.unsqueeze(1).expand_as(publish_t)          # [B, N]
                elapsed = ((cur_t - publish_t) // self.tccm_elapsed_div).clamp(min=0)
                s_t = self.tccm_time(elapsed)                                         # [B, N]
                if self.tccm_real_publish_time:
                    unknown_mask = ~self.tccm_publish_time_valid[news_indices_for_tccm]
                else:
                    unknown_mask = (publish_t < 0)
                s_t = torch.where(unknown_mask, torch.ones_like(s_t), s_t)

                # (C) Popularity score s_p — word (and optionally entity) CTR lookup
                # T-dim is identical for word/entity → use word_ctr_table as the reference (entity buffers are not registered in word_only).
                ctr_bucket = (news_current_time // self.tccm_ctr_window_hours).clamp(
                    min=0, max=self.tccm_word_ctr_table.size(0) - 1
                )                                                                     # [B]
                ctr_bucket_bn = ctr_bucket.unsqueeze(1).expand(-1, news_representation.size(1))  # [B, N]
                word_ctr_ids = self.tccm_word_ctr_table[
                    ctr_bucket_bn.unsqueeze(-1).expand(-1, -1, news_title_text.size(-1)),
                    news_title_text
                ]
                if self.tccm_pop.mode == 'word+entity':
                    cand_entity_idx = self.tccm_news_entity_indices[news_indices_for_tccm]       # [B, N, E]
                    entity_ctr_ids = self.tccm_entity_ctr_table[
                        ctr_bucket_bn.unsqueeze(-1).expand(-1, -1, cand_entity_idx.size(-1)),
                        cand_entity_idx
                    ]
                    s_p_raw = self.tccm_pop(word_ctr_ids, entity_ctr_ids)                        # [B, N]
                else:
                    s_p_raw = self.tccm_pop(word_ctr_ids, None)

                # (D) Causal intervention do(P) at inference only
                # _tccm_allow_intervention: True only when mode=='test' (defaults to True to preserve existing behavior).
                _allow_intv = getattr(self, '_tccm_allow_intervention', True)
                if (not self.training) and getattr(self.config, 'tccm_do_intervention', False) and _allow_intv:
                    s_p = self.tccm_sp_train_mean.expand_as(s_p_raw)
                else:
                    s_p = s_p_raw

                # (E) Fusion with per-user activity gate
                if user_representation.dim() == 3:
                    user_for_gate = user_representation[:, 0, :]
                else:
                    user_for_gate = user_representation
                g = self.tccm_activity_gate(user_for_gate)                             # [B, 1]
                s_bias = s_t                                                     # [B, N]

                logits = g * S_popcorn + (1.0 - g) * s_bias

        elif self.click_predictor == 'FIM':
            logits = self.fc(user_representation).squeeze(dim=2)
        elif self.click_predictor == 'TCCM':
            # === TCCM Fusion Scoring (CIKM 2023) ===
            # news_representation: [B, N, d_news], user_representation: [B, d_user] or [B, N, d_user]
            # news_indices_for_tccm: [B, N]  (int64) — corpus news indices for each candidate
            # news_current_time: [B]          (int64) — hour bucket of this behavior
            assert news_indices_for_tccm is not None and news_current_time is not None, \
                'TCCM click_predictor requires news_indices_for_tccm and news_current_time'

            # ---- (A) Matching score s_m ----
            # If POPCORN with I1+I3 → reuse I3 score (S_I − β·sigmoid(α·S_P)) as s_m.
            # Otherwise → standard dot product (user · news).
            if (self.config.news_encoder == 'POPCORN'
                    and getattr(self.config, 'use_I1', False)
                    and getattr(self.config, 'use_I3', False)
                    and news_representation.size(-1) % 2 == 0
                    and user_representation.dim() == 3
                    and user_representation.size(-1) == news_representation.size(-1)):
                d_half = news_representation.size(-1) // 2
                f_c = news_representation[:, :, :d_half]
                p_c = news_representation[:, :, d_half:]
                f_u = user_representation[:, :, :d_half]
                p_u = user_representation[:, :, d_half:]
                S_I = (f_u * f_c).sum(dim=-1)                              # [B, N]
                S_P = F.cosine_similarity(p_u, p_c, dim=-1)                # [B, N]
                beta = float(getattr(self.config, 'pop_penalty_weight', 2.0))
                alpha_popcorn = float(getattr(self.config, 'popcorn_alpha', 0.1))
                s_m = S_I - beta * torch.sigmoid(alpha_popcorn * S_P)
            else:
                # user_representation may be [B, d] or [B, N, d] — align to news
                if user_representation.dim() == 2:
                    user_representation_exp = user_representation.unsqueeze(1).expand(-1, news_representation.size(1), -1)
                else:
                    user_representation_exp = user_representation
                # Edge case: POPCORN with I1=True but I3=False gives user=[B,N,d] vs news=[B,N,2d].
                # Use the f_c half of news (first half) for dot product with f_u.
                if user_representation_exp.size(-1) != news_representation.size(-1):
                    d_u = user_representation_exp.size(-1)
                    news_for_dot = news_representation[..., :d_u]
                    s_m = (user_representation_exp * news_for_dot).sum(dim=-1)
                else:
                    s_m = (user_representation_exp * news_representation).sum(dim=-1)   # [B, N]

            # ---- (B) Time score s_t ----
            publish_t = self.tccm_news_publish_time[news_indices_for_tccm]   # [B, N] (int64, -1 for unknown in default mode)
            cur_t = news_current_time.unsqueeze(1).expand_as(publish_t)      # [B, N]
            elapsed = ((cur_t - publish_t) // self.tccm_elapsed_div).clamp(min=0)
            s_t = self.tccm_time(elapsed)                                     # [B, N]
            # Unknown publish_time → force s_t = 1 (truly neutral, no gradient into time_embedding)
            if self.tccm_real_publish_time:
                unknown_mask = ~self.tccm_publish_time_valid[news_indices_for_tccm]
            else:
                unknown_mask = (publish_t < 0)
            s_t = torch.where(unknown_mask, torch.ones_like(s_t), s_t)

            # ---- (C) Popularity score s_p ----
            # The T-dim size is identical for the word/entity CTR tables (num_ctr_buckets), so clamp based on word_ctr_table.
            # In word_only mode the entity_ctr_table buffer may not exist.
            ctr_bucket = (news_current_time // self.tccm_ctr_window_hours).clamp(
                min=0, max=self.tccm_word_ctr_table.size(0) - 1
            )  # [B]
            ctr_bucket_bn = ctr_bucket.unsqueeze(1).expand(-1, news_representation.size(1))  # [B, N]
            # word CTR ids: lookup word_ctr_table[t_bucket, title_text]  → [B, N, L]
            word_ctr_ids = self.tccm_word_ctr_table[ctr_bucket_bn.unsqueeze(-1).expand(-1, -1, news_title_text.size(-1)),
                                                     news_title_text]
            if self.tccm_pop.mode == 'word+entity':
                # entity indices for each candidate news: [B, N, E]
                cand_entity_idx = self.tccm_news_entity_indices[news_indices_for_tccm]       # [B, N, E]
                entity_ctr_ids = self.tccm_entity_ctr_table[
                    ctr_bucket_bn.unsqueeze(-1).expand(-1, -1, cand_entity_idx.size(-1)),
                    cand_entity_idx
                ]
                s_p_raw = self.tccm_pop(word_ctr_ids, entity_ctr_ids)                         # [B, N]
            else:
                s_p_raw = self.tccm_pop(word_ctr_ids, None)

            # ---- (D) Causal intervention at inference only ----
            # _tccm_allow_intervention: True only when mode=='test' in util_cached._tccm_compute_scores;
            # set to False during dev evaluation in training (mode=='dev') so the best epoch is chosen with raw s_p.
            # Defaults to True (= preserves existing behavior). During training, always raw s_p.
            _allow_intv = getattr(self, '_tccm_allow_intervention', True)
            if (not self.training) and getattr(self.config, 'tccm_do_intervention', False) and _allow_intv:
                s_p = self.tccm_sp_train_mean.expand_as(s_p_raw)
            else:
                s_p = s_p_raw

            # ---- (E) Fusion with per-user activity gate (reference TCCM/model.py:240-253, utils.py:139) ----
            # user_representation: [B, d] or [B, N, d]. Gate is per-user scalar → use first candidate slice.
            if user_representation.dim() == 3:
                user_for_gate = user_representation[:, 0, :]   # [B, d]
            else:
                user_for_gate = user_representation           # [B, d]
            g = self.tccm_activity_gate(user_for_gate)        # [B, 1] ∈ (0,1)
            s_bias = s_p * s_t                                 # [B, N]  popularity_fusion_score

            logits = g * s_m + (1.0 - g) * s_bias
        return logits
