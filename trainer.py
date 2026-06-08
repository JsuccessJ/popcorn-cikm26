import os
import signal
import shutil
import json
import time
from config import Config
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_Train_Dataset
from util import AvgMetric
from util_cached import compute_scores_with_cache
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
        self.model = model # model.py provides the forward() method
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        # TCCM uses BPR loss (paper Eq.10); other ranking models use softmax/InfoNCE-style loss.
        if config.click_predictor == 'TCCM':
            self.loss = self.bpr_loss
        elif config.click_predictor in ['dot_product', 'mlp', 'FIM', 'POPCORN']:
            self.loss = self.negative_log_softmax
        else:
            self.loss = self.negative_log_sigmoid

        # For PLM-based models, use separate learning rates
        if config.use_plm_news_encoder:
            # Separate PLM parameters from the remaining parameters
            plm_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'plm' in name:  # PLM parameters
                        plm_params.append(param)
                    else:  # other parameters (category embedding, attention, etc.)
                        other_params.append(param)

            # Set different learning rates
            self.optimizer = optim.Adam([
                {'params': plm_params, 'lr': config.plm_lr},      # 1e-5 (PLM)
                {'params': other_params, 'lr': config.lr}         # 1e-4 (others)
            ], weight_decay=config.weight_decay)

            print(f'[Single-GPU] PLM parameters: {len(plm_params)}, Other parameters: {len(other_params)}')
            print(f'[Single-GPU] PLM lr: {config.plm_lr}, Other lr: {config.lr}')
        else:
            # Original approach (non-PLM models)
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.lr,
                weight_decay=config.weight_decay
            )

        self._dataset = config.dataset
        self.mind_corpus = mind_corpus
        self.config = config  # Store config for PENR/POPCORN check
        # For PENR: return news indices for popularity loss
        return_news_indices = (config.click_predictor == 'PENR')
        # For POPCORN: return popularity labels for disentangling loss
        return_popularity_labels = (config.click_predictor == 'POPCORN')
        # For TCCM (or POPCORN + use_tccm_addon): return news indices + current time bucket
        return_tccm = (config.click_predictor == 'TCCM') or \
                      (config.click_predictor == 'POPCORN' and getattr(config, 'use_tccm_addon', False))

        self.train_dataset = MIND_Train_Dataset(
            mind_corpus,
            return_news_indices=return_news_indices,
            return_popularity_labels=return_popularity_labels,
            return_tccm=return_tccm,
        )

        # Compute total number of training steps
        total_steps = len(self.train_dataset) // config.batch_size * config.epoch
        warmup_steps = int(total_steps * 0.1)  # 10% warmup

        # Add LR scheduler (paper: 10% warmup + linear decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.run_index = run_index
        self.model_dir = config.model_dir + '/#' + str(self.run_index)
        self.best_model_dir = config.best_model_dir + '/#' + str(self.run_index)
        self.dev_res_dir = config.dev_res_dir + '/#' + str(self.run_index)
        self.result_dir = config.result_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
        with open(config.config_dir + '/#' + str(self.run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if self._dataset == 'large':
            self.prediction_dir = config.prediction_dir + '/#' + str(self.run_index)
            os.mkdir(self.prediction_dir)
        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc_results = []
        self.mrr_results = []
        self.ndcg5_results = []
        self.ndcg10_results = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg5 = 0
        self.best_dev_ndcg10 = 0
        self.best_dev_avg = AvgMetric(0, 0, 0, 0)
        self.epoch_not_increase = 0
        self.gradient_clip_norm = config.gradient_clip_norm
        self.train_loss_results = []  # for recording train loss
        self.val_loss_results = []  # for recording validation loss
        self.model.cuda()
        print('Running : ' + self.model.model_name + '\t#' + str(self.run_index))

    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def bpr_loss(self, logits):
        # BPR loss (Rendle+ 2009; TCCM paper Eq.10): -mean log sigmoid(s_pos - s_neg).
        # logits: [B, 1+K], index 0 is positive, 1..K are negatives.
        pos = logits[:, 0:1]
        neg = logits[:, 1:]
        return -F.logsigmoid(pos - neg).sum(dim=1).mean()

    def train(self):
        model = self.model
        # Resource tracking: per-epoch train/dev time and GPU peak memory
        self.epoch_train_times = []
        self.epoch_dev_times = []
        self.epoch_gpu_peak_alloc_mb = []
        self.epoch_gpu_peak_reserved_mb = []
        for e in tqdm(range(1, self.epoch + 1), desc='Epoch', dynamic_ncols=True, mininterval=1.0):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            epoch_train_start = time.perf_counter()
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.batch_size // 16,
                pin_memory=True
            )
            model.train()
            epoch_loss = 0
            
            # Show per-batch progress
            train_dataloader_with_progress = tqdm(train_dataloader, desc=f'Epoch {e}/{self.epoch}', leave=False, dynamic_ncols=True, mininterval=1.0)
            
            for batch_data in train_dataloader_with_progress:
                # Conditionally unpack based on click predictor mode (standard datasets)
                if self.config.click_predictor == 'PENR':
                    user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, sample_news_indices = batch_data
                elif self.config.click_predictor == 'POPCORN':
                    if getattr(self.config, 'use_tccm_addon', False):
                        user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                        news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, sample_candidate_popularity_labels, sample_history_popularity_labels, tccm_news_indices, tccm_current_time = batch_data
                    else:
                        user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                        news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, sample_candidate_popularity_labels, sample_history_popularity_labels = batch_data
                elif self.config.click_predictor == 'TCCM':
                    user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, tccm_news_indices, tccm_current_time = batch_data
                else:
                    user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity = batch_data
                    sample_candidate_popularity_labels = None
                    sample_history_popularity_labels = None

                # Move standard inputs to GPU
                user_ID = user_ID.cuda(non_blocking=True)                                                                                                                       # [batch_size]
                user_category = user_category.cuda(non_blocking=True)                                                                                                           # [batch_size, max_history_num]
                user_subCategory = user_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, max_history_num]
                user_title_text = user_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_mask = user_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_entity = user_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_title_length]
                user_content_text = user_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_mask = user_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_entity = user_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, max_history_num, max_content_length]
                user_history_mask = user_history_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num]
                user_history_graph = user_history_graph.cuda(non_blocking=True)                                                                                                 # [batch_size, max_history_num, max_history_num]
                user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)                                                                                 # [batch_size, category_num + 1]
                user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)                                                                           # [batch_size, max_history_num]
                news_category = news_category.cuda(non_blocking=True)                                                                                                           # [batch_size, 1 + negative_sample_num]
                news_subCategory = news_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, 1 + negative_sample_num]
                news_title_text = news_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_mask = news_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_entity = news_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_title_length]
                news_content_text = news_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_mask = news_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_entity = news_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, 1 + negative_sample_num, max_content_length]

                # Standard Forward Pass
                needs_tccm_forward = (self.config.click_predictor == 'TCCM') or \
                                     (self.config.click_predictor == 'POPCORN' and getattr(self.config, 'use_tccm_addon', False))
                if needs_tccm_forward:
                    tccm_news_indices = tccm_news_indices.cuda(non_blocking=True)     # [batch_size, 1+neg_num]
                    tccm_current_time = tccm_current_time.cuda(non_blocking=True)     # [batch_size]
                    logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                                   news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                                   news_indices_for_tccm=tccm_news_indices, news_current_time=tccm_current_time)
                else:
                    logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                                   news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) # [batch_size, 1 + negative_sample_num]

                # For batch size reference
                batch_size_dim = user_ID.size(0)

                # Recommendation loss (ℓ_rec)
                loss = self.loss(logits)

                # Auxiliary losses from encoders
                if model.news_encoder.auxiliary_loss is not None:
                    news_auxiliary_loss = model.news_encoder.auxiliary_loss.mean()
                    loss += news_auxiliary_loss

                # PENR multi-task loss (Equation 24): ℓ_total = ℓ_rec + λ_pop*ℓ_pop + β_aux*ℓ_aux
                if self.config.click_predictor == 'PENR':
                    # Popularity loss (ℓ_pop, Equation 20)
                    # Fetch ground truth popularity labels for sampled news
                    popularity_labels = torch.from_numpy(self.train_dataset.news_popularity[sample_news_indices.cpu().numpy()]).cuda()  # [batch_size, 1+K]
                    predicted_popularity = model.predicted_popularity  # [batch_size, 1+K]
                    loss_pop = F.mse_loss(predicted_popularity, popularity_labels)
                    loss += self.config.penr_lambda_pop * loss_pop  # λ_pop = 0.01

                    # Discriminator auxiliary loss with PENR weight (ℓ_aux, Equation 22)
                    if model.user_encoder.auxiliary_loss is not None:
                        user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                        loss += self.config.penr_beta_aux * user_encoder_auxiliary_loss  # β_aux = 0.001

                # POPCORN disentangling loss: L_total = L_click + λ_pop * (L_pop_candidate + L_pop_history)
                elif self.config.click_predictor == 'POPCORN':
                    # Compute L_pop only when I1=True
                    if getattr(self.config, 'use_I1', False):
                        # ========== (1) Candidate News Disentanglement Loss ==========
                        candidate_outputs = model.candidate_disentangle_outputs

                        # Check if disentangle_outputs exists (only present when I1=True)
                        if candidate_outputs is not None:
                            f_c = candidate_outputs['f_j']  # [batch_size, N, d/2]
                            p_c = candidate_outputs['p_j']  # [batch_size, N, d/2]
                            h_c = candidate_outputs['h_j']  # [batch_size, N, d]
                            logits_p_c = candidate_outputs['logits_p']  # [batch_size, N, num_classes]
                            logits_f_c = candidate_outputs['logits_f']  # [batch_size, N, num_classes]

                            # Move candidate popularity labels to GPU
                            y_c = sample_candidate_popularity_labels.cuda(non_blocking=True)  # [batch_size, N]

                            # L_r_candidate: Reconstruction loss
                            # Branch the reconstruction depending on disentangle_method
                            disentangle_method = getattr(self.config, 'disentangle_method', 'gated')
                            if disentangle_method == 'mlp':
                                # MLP variant: concatenation
                                # h_c has dimension 2d (it passed through h_projection)
                                reconstructed_c = torch.cat([f_c, p_c], dim=-1)  # [batch_size, N, 2d]
                            elif disentangle_method == 'gated':
                                # Gated variant: element-wise sum
                                # h_c has dimension d (it did not pass through h_projection)
                                reconstructed_c = f_c + p_c  # [batch_size, N, d]
                            L_r_candidate = 0.5 * F.mse_loss(reconstructed_c, h_c)  # MSE variant

                            # L_p_candidate: Popularity prediction loss
                            batch_size_train, N, num_classes = logits_p_c.shape
                            logits_p_c_flat = logits_p_c.reshape(-1, num_classes)
                            y_c_flat = y_c.reshape(-1).long()
                            L_p_candidate = F.cross_entropy(logits_p_c_flat, y_c_flat, ignore_index=-1)

                            # L_a_candidate: Adversarial loss (the GRL reverses the f_j encoder gradient)
                            logits_f_c_flat = logits_f_c.reshape(-1, num_classes)
                            L_a_candidate = F.cross_entropy(logits_f_c_flat, y_c_flat, ignore_index=-1)

                            L_pop_candidate = L_r_candidate + L_p_candidate + L_a_candidate

                            # ========== (2) History News Disentanglement Loss ==========
                            history_outputs = model.news_encoder.history_disentangle_outputs if hasattr(model.news_encoder, 'history_disentangle_outputs') else None

                            if history_outputs is not None:
                                f_h = history_outputs['f_j']  # [batch_size, max_history_num, d/2]
                                p_h = history_outputs['p_j']  # [batch_size, max_history_num, d/2]
                                h_h = history_outputs['h_j']  # [batch_size, max_history_num, d]
                                logits_p_h = history_outputs['logits_p']  # [batch_size, max_history_num, num_classes]
                                logits_f_h = history_outputs['logits_f']  # [batch_size, max_history_num, num_classes]

                                # Move history popularity labels to GPU
                                y_h = sample_history_popularity_labels.cuda(non_blocking=True)  # [batch_size, max_history_num]

                                # L_r_history: Reconstruction loss
                                # Branch the reconstruction depending on disentangle_method
                                disentangle_method = getattr(self.config, 'disentangle_method', 'gated')
                                if disentangle_method == 'mlp':
                                    # MLP variant: concatenation
                                    # h_h has dimension 2d (it passed through h_projection)
                                    reconstructed_h = torch.cat([f_h, p_h], dim=-1)  # [batch_size, max_history_num, 2d]
                                elif disentangle_method == 'gated':
                                    # Gated variant: element-wise sum
                                    # h_h has dimension d (it did not pass through h_projection)
                                    reconstructed_h = f_h + p_h  # [batch_size, max_history_num, d]
                                L_r_history = 0.5 * F.mse_loss(reconstructed_h, h_h)  # MSE variant

                                # L_p_history: Popularity prediction loss
                                batch_size_train, M, num_classes = logits_p_h.shape
                                logits_p_h_flat = logits_p_h.reshape(-1, num_classes)
                                y_h_flat = y_h.reshape(-1).long()
                                L_p_history = F.cross_entropy(logits_p_h_flat, y_h_flat, ignore_index=-1)

                                # L_a_history: Adversarial loss (the GRL reverses the f_j encoder gradient)
                                logits_f_h_flat = logits_f_h.reshape(-1, num_classes)
                                L_a_history = F.cross_entropy(logits_f_h_flat, y_h_flat, ignore_index=-1)
                                L_pop_history = L_r_history + L_p_history + L_a_history

                            # ========== (4) Total Loss ==========
                            # L_pop_candidate is always included regardless of whether history_outputs exists
                            L_pop = L_pop_candidate
                            if history_outputs is not None:
                                L_pop = L_pop + L_pop_history
                            loss += self.config.popcorn_lambda_pop * L_pop
                    # I1=False: L_pop is not computed (loss is L_click only)
                    
                    # Add base encoder's auxiliary loss (e.g., PENR discriminator loss) if present
                    if model.user_encoder.auxiliary_loss is not None:
                        loss += model.user_encoder.auxiliary_loss.mean()

                else:
                    # Non-PENR/POPCORN models: add auxiliary loss without special weighting
                    if model.user_encoder.auxiliary_loss is not None:
                        user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                        loss += user_encoder_auxiliary_loss
                epoch_loss += float(loss) * batch_size_dim
                
                # Display loss in real time
                train_dataloader_with_progress.set_postfix({'loss': f'{float(loss):.4f}', 'avg_loss': f'{epoch_loss / ((train_dataloader_with_progress.n + 1) * self.batch_size):.4f}'})
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                self.scheduler.step()  # LR scheduler step (10% warmup + linear decay)

            # Record train loss
            epoch_avg_loss = epoch_loss / len(self.train_dataset)
            self.train_loss_results.append(epoch_avg_loss)
            tqdm.write('Epoch %d : train done' % e)
            tqdm.write('loss =' + str(epoch_avg_loss))

            # Stop train timer / start dev timer
            torch.cuda.synchronize()
            epoch_train_end = time.perf_counter()
            epoch_train_time = epoch_train_end - epoch_train_start
            epoch_dev_start = time.perf_counter()

            # validation
            auc, mrr, ndcg5, ndcg10, val_loss = compute_scores_with_cache(
                model, self.mind_corpus, self.batch_size * 3 // 2, 'dev',
                self.dev_res_dir + '/' + model.model_name + '-' + str(e) + '.txt',
                self._dataset, compute_loss=True, loss_fn=self.loss,
                config=self.config
            )
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            self.val_loss_results.append(val_loss)
            tqdm.write('Epoch %d : dev done\nDev criterions' % e)
            tqdm.write('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}\nVal Loss = {:.6f}'.format(auc, mrr, ndcg5, ndcg10, val_loss))

            # Stop dev timer and capture per-epoch resource stats
            torch.cuda.synchronize()
            epoch_dev_end = time.perf_counter()
            epoch_dev_time = epoch_dev_end - epoch_dev_start
            gpu_peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            gpu_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            self.epoch_train_times.append(epoch_train_time)
            self.epoch_dev_times.append(epoch_dev_time)
            self.epoch_gpu_peak_alloc_mb.append(gpu_peak_alloc_mb)
            self.epoch_gpu_peak_reserved_mb.append(gpu_peak_reserved_mb)

            if self.dev_criterion == 'auc':
                if auc >= self.best_dev_auc:
                    self.best_dev_auc = auc
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'mrr':
                if mrr >= self.best_dev_mrr:
                    self.best_dev_mrr = mrr
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg5':
                if ndcg5 >= self.best_dev_ndcg5:
                    self.best_dev_ndcg5 = ndcg5
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg10':
                if ndcg10 >= self.best_dev_ndcg10:
                    self.best_dev_ndcg10 = ndcg10
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= self.best_dev_avg:
                    self.best_dev_avg = avg
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1

            tqdm.write('Best epoch : ' + str(self.best_dev_epoch))
            tqdm.write('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            torch.cuda.empty_cache()
            
            # Save the model (when the current epoch is the best)
            if self.epoch_not_increase == 0:
                torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))

            if self.epoch_not_increase == self.early_stopping_epoch:
                break


        with open('%s/%s-%s-dev_log.txt' % (self.dev_res_dir, model.model_name, self._dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(self.auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc_results[i], self.mrr_results[i], self.ndcg5_results[i], self.ndcg10_results[i]))

        # Save train & validation loss log
        with open('%s/%s-%s-loss_log.txt' % (self.dev_res_dir, model.model_name, self._dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tTrain_Loss\tVal_Loss\n')
            for i in range(len(self.train_loss_results)):
                f.write('%d\t%.6f\t%.6f\n' % (i + 1, self.train_loss_results[i], self.val_loss_results[i]))

        # Save resource log (training time + GPU memory)
        resource_log_path = os.path.abspath('%s/%s-%s-resource_log.txt' % (self.dev_res_dir, model.model_name, self._dataset))
        num_epochs_run = len(self.epoch_train_times)
        total_train_time = sum(self.epoch_train_times)
        total_dev_time = sum(self.epoch_dev_times)
        avg_epoch_time = (total_train_time + total_dev_time) / num_epochs_run if num_epochs_run > 0 else 0.0
        with open(resource_log_path, 'w', encoding='utf-8') as f:
            f.write('Epoch\tTrain_Time(s)\tDev_Time(s)\tCumulative(s)\tGPU_Peak_Alloc(MB)\tGPU_Peak_Reserved(MB)\n')
            cumul = 0.0
            for i in range(num_epochs_run):
                cumul += self.epoch_train_times[i] + self.epoch_dev_times[i]
                f.write('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (
                    i + 1,
                    self.epoch_train_times[i],
                    self.epoch_dev_times[i],
                    cumul,
                    self.epoch_gpu_peak_alloc_mb[i],
                    self.epoch_gpu_peak_reserved_mb[i],
                ))
            f.write('\n[Summary]\n')
            f.write('Total_Train_Time:\t%.2fs\n' % total_train_time)
            f.write('Total_Dev_Time:\t%.2fs\n' % total_dev_time)
            f.write('Total_Train+Dev:\t%.2fs\n' % (total_train_time + total_dev_time))
            f.write('Best_Epoch:\t%d\n' % self.best_dev_epoch)
            f.write('Avg_Epoch_Time:\t%.2fs\n' % avg_epoch_time)

        # Expose the path and cumulative times on config so main.py can append the test time
        self.config.total_train_time = total_train_time
        self.config.total_dev_time = total_dev_time
        self.config.resource_log_path = resource_log_path

        shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch), self.best_model_dir + '/' + model.model_name)

        # TCCM: compute sp_train_mean over the training set and persist into the best checkpoint
        if self.config.click_predictor == 'TCCM':
            self._tccm_update_sp_train_mean(model)
            # re-save the best checkpoint with updated buffer
            best_ckpt_path = self.best_model_dir + '/' + model.model_name
            torch.save({model.model_name: model.state_dict()}, best_ckpt_path)
            print(f'[TCCM] sp_train_mean updated and saved: {float(model.tccm_sp_train_mean):.4f}')

        print('Training : ' + model.model_name + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])

    def _tccm_update_sp_train_mean(self, model):
        """
        After training is complete, estimate s_p mean over the training set with the best model.
        This value is used when --tccm_do_intervention is enabled at inference.
        """
        # Load best model state to estimate sp mean on the truly selected model
        best_ckpt = self.best_model_dir + '/' + model.model_name
        if os.path.exists(best_ckpt):
            state = torch.load(best_ckpt, map_location='cpu')
            model.load_state_dict(state[model.model_name])
            model.cuda()

        model.eval()
        loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.batch_size // 16, pin_memory=True,
        )
        sp_sum = 0.0
        sp_count = 0
        with torch.no_grad():
            for batch_data in tqdm(loader, desc='[TCCM] Estimating sp_train_mean', mininterval=1.0):
                # MIND_Train_Dataset return order (TCCM path): base(21) + tccm(2) = 23-tuple total.
                # This method uses only 3 of them: news_title_text / tccm_news_indices / tccm_current_time.
                #   base_return[15] = news_title_text (candidates)
                #   batch_data[21..22] = (tccm_news_indices, tccm_current_time)
                news_title_text = batch_data[15].cuda(non_blocking=True)
                tccm_news_indices = batch_data[21].cuda(non_blocking=True)
                tccm_current_time = batch_data[22].cuda(non_blocking=True)
                # Compute s_p directly (word+entity popularity score)
                # The T dimension is identical for word/entity, so use word_ctr_table as reference (entity buffer is not registered in word_only).
                ctr_bucket = (tccm_current_time // model.tccm_ctr_window_hours).clamp(
                    min=0, max=model.tccm_word_ctr_table.size(0) - 1
                )
                ctr_bn = ctr_bucket.unsqueeze(1).expand(-1, news_title_text.size(1))
                word_ctr_ids = model.tccm_word_ctr_table[
                    ctr_bn.unsqueeze(-1).expand(-1, -1, news_title_text.size(-1)), news_title_text
                ]
                if model.tccm_pop.mode == 'word+entity':
                    cand_entity_idx = model.tccm_news_entity_indices[tccm_news_indices]
                    entity_ctr_ids = model.tccm_entity_ctr_table[
                        ctr_bn.unsqueeze(-1).expand(-1, -1, cand_entity_idx.size(-1)), cand_entity_idx
                    ]
                    s_p = model.tccm_pop(word_ctr_ids, entity_ctr_ids)
                else:
                    s_p = model.tccm_pop(word_ctr_ids, None)
                sp_sum += float(s_p.sum().item())
                sp_count += int(s_p.numel())
        mean_val = sp_sum / max(sp_count, 1)
        with torch.no_grad():
            model.tccm_sp_train_mean.fill_(float(mean_val))


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss

def distributed_train(rank, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
    # NOTE: This multi-GPU (DDP) path is NOT maintained/validated for POPCORN
    #       (it references self.config and selects the sigmoid loss for POPCORN).
    #       Train POPCORN models on a single GPU (--world_size=1).
    world_size = config.world_size
    model_name = model.model_name

    # NCCL initialization (timeout set effectively to infinity)
    import os
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # Non-blocking wait
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Async error handling

    import datetime
    # Dev evaluation time is unpredictable, so set the timeout very large (24 hours)
    timeout = datetime.timedelta(days=1)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timeout)
    config.device_id = rank
    config.set_cuda()
    model.cuda()
    loss_ = negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else negative_log_sigmoid
    epoch = config.epoch
    batch_size = config.batch_size // world_size
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # For PLM-based models, use separate learning rates
    if config.use_plm_news_encoder:
        # Separate PLM parameters from the remaining parameters
        plm_params = []
        other_params = []

        for name, param in model.module.named_parameters():
            if param.requires_grad:
                if 'plm' in name:  # PLM parameters
                    plm_params.append(param)
                else:  # other parameters (category embedding, attention, etc.)
                    other_params.append(param)

        # Set different learning rates
        optimizer = optim.Adam([
            {'params': plm_params, 'lr': config.plm_lr},      # 1e-5 (PLM)
            {'params': other_params, 'lr': config.lr}         # 1e-4 (others)
        ], weight_decay=config.weight_decay)

        if rank == 0:
            print(f'[Multi-GPU] PLM parameters: {len(plm_params)}, Other parameters: {len(other_params)}')
            print(f'[Multi-GPU] PLM lr: {config.plm_lr}, Other lr: {config.lr}')
    else:
        # Original approach (non-PLM models)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.module.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    # LR Scheduler (10% warmup + linear decay)
    from transformers import get_linear_schedule_with_warmup
    # For PENR: return news indices for popularity loss
    return_news_indices = (config.click_predictor == 'PENR')
    # For POPCORN: return popularity labels for disentangling loss
    return_popularity_labels = (config.click_predictor == 'POPCORN')

    train_dataset = MIND_Train_Dataset(mind_corpus, return_news_indices=return_news_indices, return_popularity_labels=return_popularity_labels)
    total_steps = len(train_dataset) // batch_size * epoch
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    gradient_clip_norm = config.gradient_clip_norm
    if rank == 0:
        model_dir = config.model_dir + '/#' + str(run_index)
        best_model_dir = config.best_model_dir + '/#' + str(run_index)
        dev_res_dir = config.dev_res_dir + '/#' + str(run_index)
        result_dir = config.result_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)
        if not os.path.exists(dev_res_dir):
            os.mkdir(dev_res_dir)
        with open(config.config_dir + '/#' + str(run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if config.dataset == 'large':
            prediction_dir = config.prediction_dir + '/#' + str(run_index)
            os.mkdir(prediction_dir)
        dev_criterion = config.dev_criterion
        early_stopping_epoch = config.early_stopping_epoch
        auc_results = []
        mrr_results = []
        ndcg5_results = []
        ndcg10_results = []
        best_dev_epoch = 0
        best_dev_auc = 0
        best_dev_mrr = 0
        best_dev_ndcg5 = 0
        best_dev_ndcg10 = 0
        best_dev_avg = AvgMetric(0, 0, 0, 0)
        epoch_not_increase = 0
        train_loss_results = []  # for recording train loss
        val_loss_results = []  # for recording validation loss
        print('Running : ' + model_name + '\t#' + str(run_index))

    for e in tqdm(range(1, epoch + 1), desc='Epoch', disable=(rank != 0)):
        train_dataset.negative_sampling(rank=rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_sampler.set_epoch(e)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=batch_size // 16, pin_memory=True, sampler=train_sampler)
        model.train()
        epoch_loss = 0
        
        # Show per-batch progress (rank 0 only)
        if rank == 0:
            train_dataloader_with_progress = tqdm(train_dataloader, desc=f'Epoch {e}/{epoch}', leave=False)
        else:
            train_dataloader_with_progress = train_dataloader
        
        for batch_data in train_dataloader_with_progress:
            # Conditionally unpack based on click predictor mode
            if config.click_predictor == 'PENR':
                user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, sample_news_indices = batch_data
            elif config.click_predictor == 'POPCORN':
                user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, sample_candidate_popularity_labels, sample_history_popularity_labels = batch_data
            else:
                user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity = batch_data
            user_ID = user_ID.cuda(non_blocking=True)                                                                                                                       # [batch_size]
            user_category = user_category.cuda(non_blocking=True)                                                                                                           # [batch_size, max_history_num]
            user_subCategory = user_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, max_history_num]
            user_title_text = user_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
            user_title_mask = user_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
            user_title_entity = user_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_title_length]
            user_content_text = user_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
            user_content_mask = user_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
            user_content_entity = user_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, max_history_num, max_content_length]
            user_history_mask = user_history_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num]
            user_history_graph = user_history_graph.cuda(non_blocking=True)                                                                                                 # [batch_size, max_history_num, max_history_num]
            user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)                                                                                 # [batch_size, category_num + 1]
            user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)                                                                           # [batch_size, max_history_num]
            news_category = news_category.cuda(non_blocking=True)                                                                                                           # [batch_size, 1 + negative_sample_num]
            news_subCategory = news_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, 1 + negative_sample_num]
            news_title_text = news_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
            news_title_mask = news_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
            news_title_entity = news_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_title_length]
            news_content_text = news_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
            news_content_mask = news_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
            news_content_entity = news_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, 1 + negative_sample_num, max_content_length]

            logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                           news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) # [batch_size, 1 + negative_sample_num]

            # Recommendation loss
            loss = loss_(logits)

            # Auxiliary losses from encoders
            if model.module.news_encoder.auxiliary_loss is not None:
                news_auxiliary_loss = model.module.news_encoder.auxiliary_loss.mean()
                loss += news_auxiliary_loss

            # PENR multi-task loss
            if config.click_predictor == 'PENR':
                # Popularity loss
                popularity_labels = torch.from_numpy(train_dataset.news_popularity[sample_news_indices.cpu().numpy()]).cuda()
                predicted_popularity = model.module.predicted_popularity
                loss_pop = F.mse_loss(predicted_popularity, popularity_labels)
                loss += config.penr_lambda_pop * loss_pop

                # Discriminator auxiliary loss
                if model.module.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.module.user_encoder.auxiliary_loss.mean()
                    loss += config.penr_beta_aux * user_encoder_auxiliary_loss

            # POPCORN disentangling loss
            elif config.click_predictor == 'POPCORN':
                # Compute L_pop only when I1=True
                if getattr(config, 'use_I1', False):
                    # ========== (1) Candidate News Disentanglement Loss ==========
                    candidate_outputs = model.module.candidate_disentangle_outputs

                    # Check if disentangle_outputs exists (only present when I1=True)
                    if candidate_outputs is not None:
                        f_c = candidate_outputs['f_j']  # [batch_size, N, d/2]
                        p_c = candidate_outputs['p_j']  # [batch_size, N, d/2]
                        h_c = candidate_outputs['h_j']  # [batch_size, N, d]
                        logits_p_c = candidate_outputs['logits_p']  # [batch_size, N, num_classes]
                        logits_f_c = candidate_outputs['logits_f']  # [batch_size, N, num_classes]

                        # Move candidate popularity labels to GPU
                        y_c = sample_candidate_popularity_labels.cuda(non_blocking=True)  # [batch_size, N]

                        # L_r_candidate: Reconstruction loss
                        # Branch the reconstruction depending on disentangle_method
                        disentangle_method = getattr(self.config, 'disentangle_method', 'gated')
                        if disentangle_method == 'mlp':
                            # MLP variant: concatenation
                            # h_c has dimension 2d (it passed through h_projection)
                            reconstructed_c = torch.cat([f_c, p_c], dim=-1)  # [batch_size, N, 2d]
                        elif disentangle_method == 'gated':
                            # Gated variant: element-wise sum
                            # h_c has dimension d (it did not pass through h_projection)
                            reconstructed_c = f_c + p_c  # [batch_size, N, d]
                        L_r_candidate = 0.5 * F.mse_loss(reconstructed_c, h_c)  # MSE variant

                        # L_p_candidate: Popularity prediction loss
                        batch_size_dist, N, num_classes = logits_p_c.shape
                        logits_p_c_flat = logits_p_c.reshape(-1, num_classes)
                        y_c_flat = y_c.reshape(-1).long()
                        L_p_candidate = F.cross_entropy(logits_p_c_flat, y_c_flat, ignore_index=-1)

                        # L_a_candidate: Adversarial loss (the GRL reverses the f_j encoder gradient)
                        logits_f_c_flat = logits_f_c.reshape(-1, num_classes)
                        L_a_candidate = F.cross_entropy(logits_f_c_flat, y_c_flat, ignore_index=-1)

                        L_pop_candidate = L_r_candidate + L_p_candidate + L_a_candidate

                        # ========== (2) History News Disentanglement Loss ==========
                        history_outputs = model.module.news_encoder.disentangle_outputs

                        if history_outputs is not None:
                            f_h = history_outputs['f_j']  # [batch_size, max_history_num, d/2]
                            p_h = history_outputs['p_j']  # [batch_size, max_history_num, d/2]
                            h_h = history_outputs['h_j']  # [batch_size, max_history_num, d]
                            logits_p_h = history_outputs['logits_p']  # [batch_size, max_history_num, num_classes]
                            logits_f_h = history_outputs['logits_f']  # [batch_size, max_history_num, num_classes]

                            # Move history popularity labels to GPU
                            y_h = sample_history_popularity_labels.cuda(non_blocking=True)  # [batch_size, max_history_num]

                            # L_r_history: Reconstruction loss
                            # Branch the reconstruction depending on disentangle_method
                            disentangle_method = getattr(self.config, 'disentangle_method', 'gated')
                            if disentangle_method == 'mlp':
                                # MLP variant: concatenation
                                # h_h has dimension 2d (it passed through h_projection)
                                reconstructed_h = torch.cat([f_h, p_h], dim=-1)  # [batch_size, max_history_num, 2d]
                            elif disentangle_method == 'gated':
                                # Gated variant: element-wise sum
                                # h_h has dimension d (it did not pass through h_projection)
                                reconstructed_h = f_h + p_h  # [batch_size, max_history_num, d]
                            L_r_history = 0.5 * F.mse_loss(reconstructed_h, h_h)  # MSE variant

                            # L_p_history: Popularity prediction loss
                            batch_size_dist, M, num_classes = logits_p_h.shape
                            logits_p_h_flat = logits_p_h.reshape(-1, num_classes)
                            y_h_flat = y_h.reshape(-1).long()
                            L_p_history = F.cross_entropy(logits_p_h_flat, y_h_flat, ignore_index=-1)

                            # L_a_history: Adversarial loss (the GRL reverses the f_j encoder gradient)
                            logits_f_h_flat = logits_f_h.reshape(-1, num_classes)
                            L_a_history = F.cross_entropy(logits_f_h_flat, y_h_flat, ignore_index=-1)

                            L_pop_history = L_r_history + L_p_history + L_a_history

                            # ========== (4) Total Loss ==========
                            L_pop = L_pop_candidate + L_pop_history
                            loss += config.popcorn_lambda_pop * L_pop  # λ_pop = 0.5
                # I1=False: L_pop is not computed (loss is L_click only)

                # Add base encoder's auxiliary loss (e.g., PENR discriminator loss) if present
                if model.module.user_encoder.auxiliary_loss is not None:
                    loss += model.module.user_encoder.auxiliary_loss.mean()

            else:
                # Non-PENR/POPCORN models
                if model.module.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.module.user_encoder.auxiliary_loss.mean()
                    loss += user_encoder_auxiliary_loss
            epoch_loss += float(loss) * user_ID.size(0)
            
            # Display loss in real time (rank 0 only)
            if rank == 0 and hasattr(train_dataloader_with_progress, 'set_postfix'):
                train_dataloader_with_progress.set_postfix({'loss': f'{float(loss):.4f}', 'avg_loss': f'{epoch_loss / ((train_dataloader_with_progress.n + 1) * batch_size):.4f}'})
            
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()  # LR scheduler step (10% warmup + linear decay)

        # Record train loss (rank 0 only)
        epoch_avg_loss = epoch_loss / len(train_dataset) * world_size
        if rank == 0:
            train_loss_results.append(epoch_avg_loss)

        print('rank %d : Epoch %d : train done' % (rank, e))
        print('rank %d : loss = %.6f' % (rank, epoch_avg_loss))

        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # dev (performed only on rank 0; rank 1 prepares for the next epoch)
        if rank == 0:
            # Use a larger dev batch size (for speed)
            dev_batch_size = batch_size * 4  # set to 4x with memory in mind

            auc, mrr, ndcg5, ndcg10, val_loss = compute_scores_with_cache(
                model.module, mind_corpus, dev_batch_size, 'dev',
                dev_res_dir + '/' + model_name + '-' + str(e) + '.txt',
                config.dataset, compute_loss=True, loss_fn=loss_,
                config=config
            )
            auc_results.append(auc)
            mrr_results.append(mrr)
            ndcg5_results.append(ndcg5)
            ndcg10_results.append(ndcg10)
            val_loss_results.append(val_loss)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}\nVal Loss = {:.6f}'.format(auc, mrr, ndcg5, ndcg10, val_loss))
            if dev_criterion == 'auc':
                if auc >= best_dev_auc:
                    best_dev_auc = auc
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'mrr':
                if mrr >= best_dev_mrr:
                    best_dev_mrr = mrr
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'ndcg5':
                if ndcg5 >= best_dev_ndcg5:
                    best_dev_ndcg5 = ndcg5
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'ndcg10':
                if ndcg10 >= best_dev_ndcg10:
                    best_dev_ndcg10 = ndcg10
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= best_dev_avg:
                    best_dev_avg = avg
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1

            print('Best epoch :', best_dev_epoch)
            if dev_criterion == 'auc':
                print('Best AUC : %.4f' % best_dev_auc)
            elif dev_criterion == 'mrr':
                print('Best MRR : %.4f' % best_dev_mrr)
            elif dev_criterion == 'ndcg5':
                print('Best nDCG@5 : %.4f' % best_dev_ndcg5)
            elif dev_criterion == 'ndcg10':
                print('Best nDCG@10 : %.4f' % best_dev_ndcg10)
            else:
                print('Best avg : ' + str(best_dev_avg))
            torch.cuda.empty_cache()
            if epoch_not_increase == 0:
                torch.save({model_name: model.module.state_dict()}, model_dir + '/' + model_name + '-' + str(best_dev_epoch))

            # Store the early-stopping decision in a tensor (existing patience + Epoch 2 boundary)

            is_early_stop = (epoch_not_increase > early_stopping_epoch)

            early_stop_signal = torch.tensor([1 if is_early_stop else 0],
                                            dtype=torch.int32, device='cuda')


        else:
            # Rank 1 creates an empty tensor
            early_stop_signal = torch.tensor([0], dtype=torch.int32, device='cuda')

        # Synchronize and check early stopping before starting the next epoch
        dist.barrier()
        dist.broadcast(early_stop_signal, src=0)

        if early_stop_signal[0] == 1:
            if rank == 0:
                print(f'Early stopping at epoch {e}')
            break

    if rank == 0:
        with open('%s/%s-%s-dev_log.txt' % (dev_res_dir, model_name, config.dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, auc_results[i], mrr_results[i], ndcg5_results[i], ndcg10_results[i]))

        # Save train & validation loss log
        with open('%s/%s-%s-loss_log.txt' % (dev_res_dir, model_name, config.dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tTrain_Loss\tVal_Loss\n')
            for i in range(len(train_loss_results)):
                f.write('%d\t%.6f\t%.6f\n' % (i + 1, train_loss_results[i], val_loss_results[i]))

        print('Training : ' + model_name + ' #' + str(run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % auc_results[best_dev_epoch - 1])
        print('MRR : %.4f' % mrr_results[best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % ndcg5_results[best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % ndcg10_results[best_dev_epoch - 1])
        shutil.copy(model_dir + '/' + model_name + '-' + str(best_dev_epoch), best_model_dir + '/' + model_name)
        os.kill(os.getpid(), signal.SIGKILL)
