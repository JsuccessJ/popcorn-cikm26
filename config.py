import os
import argparse
import time
import torch
import random
import numpy as np
import json
# prepare_MIND_{200k,small,large} are invoked dynamically via exec() in preliminary_setup()
from prepare_MIND_dataset import prepare_MIND_200k, prepare_MIND_small, prepare_MIND_large


class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='Neural news recommendation')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='CNE', choices=['CNE', 'CNN', 'MHSA', 'KCNN', 'HDC', 'NAML', 'PNE', 'DAE', 'Inception', 'PLMMiner', 'PENR', 'POPCORN', 'CROWN', 'NAML_Title', 'NAML_Content', 'CNE_Title', 'CNE_Content', 'CNE_wo_CS', 'CNE_wo_CA'], help='News encoder')
        parser.add_argument('--user_encoder', type=str, default='SUE', choices=['SUE', 'LSTUR', 'MHSA', 'ATT', 'CATT', 'FIM', 'PUE', 'GRU', 'OMAP', 'MINER', 'PENR', 'POPCORN', 'CROWN', 'SUE_wo_GCN', 'SUE_wo_HCA'], help='User encoder')
        parser.add_argument('--dev_model_path', type=str, default='', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Specific test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--config_file', type=str, default='', help='Config file path')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='small', choices=['200k', 'small', 'large', 'eb-nerd', 'adressa', 'adressa-small', 'adressa-lifetime'], help='Dataset type')
        parser.add_argument('--tokenizer', type=str, default='MIND', choices=['MIND', 'NLTK'], help='Sentence tokenizer')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        parser.add_argument('--max_abstract_length', type=int, default=128, help='Sentence truncate length for abstract')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50, help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=16, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4, help='Gradient clip norm (non-positive value for no clipping)')
        parser.add_argument('--world_size', type=int, default=1, help='World size of multi-process GPU training')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg', choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Validation criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=5, help='Epoch number of stop training after dev result does not improve')
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--entity_embedding_dim', type=int, default=100, choices=[100], help='Entity embedding dimension')
        parser.add_argument('--context_embedding_dim', type=int, default=100, choices=[100], help='Context embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'], help='CNN group')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')
        parser.add_argument('--attention_dim', type=int, default=200, help="Attention dimension")
        parser.add_argument('--head_num', type=int, default=20, help='Head number of multi-head self-attention')
        parser.add_argument('--head_dim', type=int, default=20, help='Head dimension of multi-head self-attention')
        parser.add_argument('--user_embedding_dim', type=int, default=50, help='User embedding dimension')
        parser.add_argument('--category_embedding_dim', type=int, default=50, help='Category embedding dimension')
        parser.add_argument('--subCategory_embedding_dim', type=int, default=50, help='SubCategory embedding dimension')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--no_self_connection', default=False, action='store_true', help='Whether the graph contains self-connection')
        parser.add_argument('--no_adjacent_normalization', default=False, action='store_true', help='Whether normalize the adjacent matrix')
        parser.add_argument('--gcn_normalization_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'], help='GCN normalization for adjacent matrix A (\"symmetric\" for D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}; \"asymmetric\" for D^{-\\frac{1}{2}}A)')
        parser.add_argument('--gcn_layer_num', type=int, default=4, help='Number of GCN layer')
        parser.add_argument('--no_gcn_residual', default=False, action='store_true', help='Whether apply residual connection to GCN')
        parser.add_argument('--gcn_layer_norm', default=False, action='store_true', help='Whether apply layer normalization to GCN')
        parser.add_argument('--hidden_dim', type=int, default=200, help='Encoder hidden dimension')
        parser.add_argument('--long_term_masking_probability', type=float, default=0.1, help='Probability of masking long-term representation for LSTUR')
        parser.add_argument('--click_predictor', type=str, default='dot_product', choices=['dot_product', 'mlp', 'sigmoid', 'FIM', 'PENR', 'POPCORN', 'TCCM'], help='Click predictor')
        # PLM-NR config
        parser.add_argument('--plm_type', type=str, default='bert',choices=['bert', 'roberta', 'unilm', 'none'], help='Pre-trained Language Model type')
        parser.add_argument('--plm_model_name', type=str, default='bert-base-uncased',help='Specific PLM model name from HuggingFace')
        parser.add_argument('--plm_frozen_layers', type=int, default=0, help='Number of PLM layers to freeze (0=fine-tune all)')
        parser.add_argument('--plm_lr', type=float, default=1e-5,help='Learning rate for PLM fine-tuning')
        parser.add_argument('--plm_pooling', type=str, default='attention',choices=['cls', 'average', 'attention'],help='Pooling method for PLM hidden states')
        parser.add_argument('--use_plm_news_encoder', action='store_true', help='Use PLM-based news encoder')
        # MINER-specific parameters
        parser.add_argument('--num_interest_vectors', type=int, default=32, help='Number of interest vectors K in MINER (default: 32)')
        parser.add_argument('--context_code_dim', type=int, default=200, help='Dimension of context codes in MINER (default: 200)')
        parser.add_argument('--disagreement_beta', type=float, default=0.8, help='Weight for disagreement regularization (default: 0.8)')
        parser.add_argument('--miner_aggregation', type=str, default='weighted', choices=['max', 'mean', 'weighted'], help='Score aggregation method in MINER (default: weighted)')
        # MINER Category-aware attention parameter
        parser.add_argument('--category_aware_lambda', type=float, default=0.5, help='Weight for category similarity in attention (default: 0.5)')
        parser.add_argument('--use_category_glove', action='store_true',help='Use Glove initialization for category embeddings')
        # PENR-specific parameters
        parser.add_argument('--penr_num_attention_heads', type=int, default=6, help='Number of attention heads in PENR MHSA (default: 6)')
        parser.add_argument('--penr_attention_query_dim', type=int, default=200, help='Attention query dimension in PENR (default: 200)')
        parser.add_argument('--penr_num_interest_views', type=int, default=5, help='Number of interest views in PENR user encoder (default: 5)')
        parser.add_argument('--penr_lambda_pop', type=float, default=0.01, help='Popularity loss weight in PENR (default: 0.01)')
        parser.add_argument('--penr_beta_aux', type=float, default=0.001, help='Discriminator loss weight in PENR (default: 0.001)')

         # CROWN-specific parameters
        parser.add_argument('--intent_num', type=int, default=3, help='Number of intents for CROWN (default: 3)')
        parser.add_argument('--intent_embedding_dim', type=int, default=400, help='Intent embedding dimension for CROWN (default: 400)')
        parser.add_argument('--crown_isab_num_heads', type=int, default=4, help='Number of ISAB heads for CROWN (default: 4)')
        parser.add_argument('--crown_isab_num_inds', type=int, default=4, help='Number of ISAB inducing points for CROWN (default: 4)')
        parser.add_argument('--crown_alpha', type=float, default=0.3, help='Weight for category prediction auxiliary loss in CROWN (default: 0.1)')
        parser.add_argument('--crown_num_layers', type=int, default=1, help='Number of transformer layers for CROWN (default: 1,2)')
        parser.add_argument('--crown_feedforward_dim', type=int, default=512, help='Feedforward dimension for CROWN transformer (default: 512)')

        #✅POPCORN-specific parameters✅
        # Base encoder selection
        parser.add_argument('--popcorn_base_news_encoder', type=str, default='MHSA',
                            choices=['MHSA', 'NAML', 'CNE', 'CNN', 'CROWN', 'PENR', 'PLMMiner'],
                            help='Base news encoder for POPCORN (default: MHSA)')
        parser.add_argument('--popcorn_base_user_encoder', type=str, default='ATT',
                            choices=['ATT', 'MHSA', 'CATT', 'GRU', 'SUE', 'LSTUR', 'CROWN', 'PENR', 'MINER'],
                            help='Base user encoder for POPCORN (default: ATT)')
        # Ablation flags
        parser.add_argument('--use_I1', default=False, action='store_true',
                            help='Enable I1: Popularity-decoupled News Modeling')
        parser.add_argument('--disentangle_method', type=str, default='mlp', choices=['mlp', 'gated'],
                            help='Disentangle method for I1 (default: mlp)')
        parser.add_argument('--use_I2', default=False, action='store_true',
                            help='Enable I2: Candidate-guided User Modeling (Top-K Attention)')
        parser.add_argument('--use_I3', default=False, action='store_true',
                            help='Enable I3: Popularity-discounted Interest Matching')
        # I1: Popularity Disentanglement
        parser.add_argument('--popcorn_num_pop_classes', type=int, default=50,
                            help='Number of popularity classes for classification (default: 50)')
        parser.add_argument('--popcorn_class_method', type=str, default='quantile_rank', choices=['quantile_rank', 'fixed'],
                            help='Popularity class assignment method: fixed or quantile_rank')
        parser.add_argument('--popcorn_pop_normalization', type=str, default='topic', choices=['topic', 'global'],
                            help='Popularity normalization scope: topic=normalize by topic max, global=rank across all news (no normalization)')
        parser.add_argument('--popcorn_lambda_pop', type=float, default=0.5,
                            help='Weight of L_pop in total loss (default: 0.5)')
        # I2: Candidate-guided User Modeling
        parser.add_argument('--popcorn_top_k', type=int, default=30,
                            help='Top-K history news selection in I2 (default: 30)')
        parser.add_argument('--popcorn_epsilon', type=float, default=0.01,
                            help='Reweighting factor for non-top-K news in I2 (default: 0.01)')
        parser.add_argument('--popcorn_attention_heads', type=int, default=20,
                            help='Number of attention heads in I2 candidate-guided selection (default: 20)')
        parser.add_argument('--popcorn_attention_mode', type=str, default='topic-aware', choices=['topic-aware', 'candidate-aware'],
                            help='I2 attention mode: topic-aware (Q=t_c, K=t_j) or candidate-aware (Q=f_c/p_c, K=f_j/p_j)')
        parser.add_argument('--popcorn_use_gate', default=True, action='store_true',
                            help='Use gated residual connection in I2 (default: True)')
        parser.add_argument('--no_popcorn_use_gate', dest='popcorn_use_gate', action='store_false',
                            help='Disable gated residual in I2')
        # I3: Popularity-discounted Interest Matching
        parser.add_argument('--popcorn_alpha', type=float, default=0.1,
                            help='Scaling factor alpha for S_P in I3 score = S_I - beta*sigmoid(alpha*S_P) (default: 0.1)')
        parser.add_argument('--pop_penalty_weight', type=float, default=2.0,
                            help='Penalty weight beta for I3 score = S_I - beta*sigmoid(alpha*S_P) (default: 2.0)')

        # ✅TCCM-specific parameters✅
        # Paper: "TCCM: Time and Content-Aware Causal Model for Unbiased News Recommendation" (CIKM 2023)
        # Activated when --click_predictor TCCM
        parser.add_argument('--tccm_time_emb_dim', type=int, default=200,
                            help='[TCCM] Time embedding dimension (default: 200, paper Section 4.1)')
        parser.add_argument('--tccm_pop_emb_dim', type=int, default=200,
                            help='[TCCM] Popularity (entity/word CTR) embedding dimension (default: 200)')
        parser.add_argument('--tccm_pop_buckets', type=int, default=200,
                            help='[TCCM] Number of CTR bucket ids (default: 200, reference NewsContent.py:316)')
        parser.add_argument('--tccm_time_buckets', type=int, default=2000,
                            help='[TCCM] Number of time bucket ids (default: 2000)')
        parser.add_argument('--tccm_lambda', type=float, default=0.5,
                            help='[TCCM] Recency power lambda: s_t=(1/t_prime)^lambda (default: 2.0)')
        parser.add_argument('--tccm_ctr_window_hours', type=int, default=24,
                            help='[TCCM] Hours per CTR time bucket (default: 24, reference day=1)')
        parser.add_argument('--tccm_max_entities', type=int, default=5,
                            help='[TCCM] Max number of entities per news (default: 5)')
        parser.add_argument('--tccm_attention_heads', type=int, default=20,
                            help='[TCCM] Number of attention heads in popularity module (default: 20)')
        parser.add_argument('--tccm_attention_head_dim', type=int, default=20,
                            help='[TCCM] Attention head dim in popularity module (default: 20)')
        parser.add_argument('--tccm_do_intervention', default=False, action='store_true',
                            help='[TCCM] Enable causal intervention do(P): replace s_p with train-set mean at inference only')
        parser.add_argument('--tccm_content_mode', type=str, default='word+entity', choices=['word+entity', 'word_only'],
                            help='[TCCM] Content source for popularity module: word+entity (MIND) or word_only (Adressa/EB-NeRD)')
        parser.add_argument('--tccm_use_real_publish_time', default=False, action='store_true',
                            help='[TCCM] Replace first-impression-based publish_time with real article published_time '
                                 '(EB-NERD only). Loads news_publish_time_real-eb-nerd.pkl and switches elapsed to '
                                 'day units (elapsed//24) so the 2000-bucket cap covers ~5.5 years instead of 83 days.')
        # ✅POPCORN + TCCM add-on✅
        # When click_predictor=POPCORN, optionally inject TCCM's s_t / s_p / activity_gate into the score.
        # Default False → existing POPCORN runs behave identically.
        parser.add_argument('--use_tccm_addon', default=False, action='store_true',
                            help='[POPCORN+TCCM] Fuse TCCM time/popularity/gate into POPCORN score (only takes effect with --click_predictor POPCORN)')

        # Parameters for base models not used in the current framework (DAE, NPA, HDC, FIM, OMAP/HiFiArk)
        # parser.add_argument('--Alpha', type=float, default=0.1, help='Reconstruction loss weight for DAE')
        # parser.add_argument('--personalized_embedding_dim', type=int, default=200, help='Personalized embedding dimension for NPA/PUE')
        # parser.add_argument('--HDC_window_size', type=int, default=3, help='Convolution window size of HDC for FIM')
        # parser.add_argument('--HDC_filter_num', type=int, default=150, help='Convolution filter num of HDC for FIM')
        # parser.add_argument('--conv3D_filter_num_first', type=int, default=32, help='3D matching convolution filter num of the first layer for FIM')
        # parser.add_argument('--conv3D_kernel_size_first', type=int, default=3, help='3D matching convolution kernel size of the first layer for FIM')
        # parser.add_argument('--conv3D_filter_num_second', type=int, default=16, help='3D matching convolution filter num of the second layer for FIM')
        # parser.add_argument('--conv3D_kernel_size_second', type=int, default=3, help='3D matching convolution kernel size of the second layer for FIM')
        # parser.add_argument('--maxpooling3D_size', type=int, default=3, help='3D matching pooling size for FIM')
        # parser.add_argument('--maxpooling3D_stride', type=int, default=3, help='3D matching pooling stride for FIM')
        # parser.add_argument('--OMAP_head_num', type=int, default=3, help='Head num of OMAP for Hi-Fi Ark')
        # parser.add_argument('--HiFi_Ark_regularizer_coefficient', type=float, default=0.1, help='Coefficient of regularization loss for Hi-Fi Ark')

        self.attribute_dict = dict(vars(parser.parse_args()))
        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        
        # Automatically enable use_plm_news_encoder when a PLM-based news encoder is selected
        is_plm_encoder = self.news_encoder in ['PLMMiner']
        is_popcorn_plm = (self.news_encoder == 'POPCORN' and getattr(self, 'popcorn_base_news_encoder', 'MHSA') in ['PLMMiner'])

        if is_plm_encoder or is_popcorn_plm or getattr(self, 'use_plm_news_encoder', False):
            self.use_plm_news_encoder = True
            self.attribute_dict['use_plm_news_encoder'] = True
        
        if self.dataset == 'eb-nerd':
            self.train_root = '../EB-NERD-Dataset/train'
            self.dev_root = '../EB-NERD-Dataset/dev'
            self.test_root = '../EB-NERD-Dataset/test'
        elif self.dataset == 'adressa':
            self.train_root = '../Adressa/train'
            self.dev_root = '../Adressa/dev'
            self.test_root = '../Adressa/test'
        elif self.dataset == 'adressa-small':
            self.train_root = '../Adressa-small/train'
            self.dev_root = '../Adressa-small/dev'
            self.test_root = '../Adressa-small/test'
        elif self.dataset == 'adressa-lifetime':
            self.train_root = '../Adressa-lifetime/train'
            self.dev_root = '../Adressa-lifetime/dev'
            self.test_root = '../Adressa-lifetime/test'
        else:
            self.train_root = '../MIND-%s/train' % self.dataset
            self.dev_root = '../MIND-%s/dev' % self.dataset
            self.test_root = '../MIND-%s/test' % self.dataset
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        self.attribute_dict['dropout_rate'] = self.dropout_rate
        self.attribute_dict['gcn_layer_num'] = self.gcn_layer_num
        self.attribute_dict['epoch'] = self.epoch
        self.attribute_dict['seed'] = self.seed
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file : ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    import sys
                    # Use a JSON config value only when that option was not explicitly passed on the CLI
                    for attribute in self.attribute_dict:
                        if attribute in configs:
                            # Apply the JSON value only if the option is absent from the command line (sys.argv)
                            if f'--{attribute}' not in sys.argv:
                                setattr(self, attribute, configs[attribute])
                                self.attribute_dict[attribute] = configs[attribute]
            else:
                raise Exception('Config file does not exist : ' + self.config_file)
        assert not (self.no_self_connection and not self.no_adjacent_normalization), 'Adjacent normalization of graph only can be set in case of self-connection'
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute in self.attribute_dict:
            print(attribute + ' : ' + str(getattr(self, attribute)))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        assert self.batch_size % self.world_size == 0, 'For multi-gpu training, batch size must be divisible by world size'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1024'


    def set_cuda(self):
        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # For reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)


    def preliminary_setup(self):
        dataset_files = [  # uses the _raw.tsv (formatted) files
            self.train_root + '/news_raw.tsv', self.train_root + '/behaviors_raw.tsv', self.train_root + '/entity_embedding.vec', self.train_root + '/context_embedding.vec', 
            self.dev_root + '/news_raw.tsv', self.dev_root + '/behaviors_raw.tsv', self.dev_root + '/entity_embedding.vec', self.dev_root + '/context_embedding.vec', 
            self.test_root + '/news_raw.tsv', self.test_root + '/behaviors_raw.tsv', self.test_root + '/entity_embedding.vec', self.test_root + '/context_embedding.vec'
        ]
        if not all(list(map(os.path.exists, dataset_files))):
            if self.dataset not in ['eb-nerd', 'adressa', 'adressa-small', 'adressa-lifetime']:  # EB-NERD and Adressa are already prepared
                exec('prepare_MIND_%s()' % self.dataset)
            else:
                raise FileNotFoundError(f'{self.dataset} dataset files not found. Please check the path: {self.train_root}')

        # Use a hierarchical folder structure for POPCORN models
        if self.news_encoder == 'POPCORN' and self.user_encoder == 'POPCORN':
            base_news_encoder = getattr(self, 'popcorn_base_news_encoder', 'MHSA')
            base_user_encoder = getattr(self, 'popcorn_base_user_encoder', 'ATT')
            base_model_name = f'{base_news_encoder}-{base_user_encoder}'
            model_name = f'POPCORN/{base_model_name}'
        else:
            model_name = self.news_encoder + '-' + self.user_encoder

        # POPCORN click_predictor + use_tccm_addon: a variant that fuses TCCM's time/popularity/gate
        # components into the POPCORN score. Stored separately so it does not mix with POPCORN-only results.
        if self.click_predictor == 'POPCORN' and getattr(self, 'use_tccm_addon', False):
            model_name = f'TCCM_addon/{model_name}'

        # When using the TCCM click_predictor, store results in a separate subdirectory,
        # fully isolated from other models' results
        if self.click_predictor == 'TCCM':
            model_name = f'TCCM/{model_name}'
        
        mkdirs = lambda x: os.makedirs(x, exist_ok=True) if not os.path.exists(x) else None
        self.config_dir = 'configs/' + self.dataset + '/' + model_name
        self.model_dir = 'models/' + self.dataset + '/' + model_name
        self.best_model_dir = 'best_model/' + self.dataset + '/' + model_name
        self.dev_res_dir = 'dev/res/' + self.dataset + '/' + model_name
        self.test_res_dir = 'test/res/' + self.dataset + '/' + model_name
        self.result_dir = 'results/' + self.dataset + '/' + model_name
        mkdirs(self.config_dir)
        mkdirs(self.model_dir)
        mkdirs(self.best_model_dir)
        mkdirs('dev/ref')
        mkdirs(self.dev_res_dir)
        mkdirs('test/ref')
        mkdirs(self.test_res_dir)
        mkdirs(self.result_dir)
        if not os.path.exists('dev/ref/truth-%s.txt' % self.dataset):
            with open(os.path.join(self.dev_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('dev/ref/truth-%s.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if self.dataset != 'large':
            if not os.path.exists('test/ref/truth-%s.txt' % self.dataset):
                with open(os.path.join(self.test_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as test_f:
                    with open('test/ref/truth-%s.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                        for test_ID, line in enumerate(test_f):
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        else:
            self.prediction_dir = 'prediction/large/' + model_name
            mkdirs(self.prediction_dir)


    def __init__(self):
        self.parse_argument()
        self.preliminary_setup()
        self.set_cuda()


if __name__ == '__main__':
    config = Config()
