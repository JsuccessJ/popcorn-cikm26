"""
Fast evaluation module leveraging News Encoder caching (fully cached version).

Bottleneck analysis:
  Previous version: only candidate repr cached -> history repr still runs PLM every batch (slow!)
  Current version:  both candidate + history cached -> PLM calls fully eliminated (fast!)

Supported models:
  - MINER, ATT, MHSA, GRU: fully bypass forward(), run only attention/aggregation
  - POPCORN: temporarily replace news_encoder with IdentityEncoder -> skip only the PLM,
             while POPCORN internal logic (Top-K Attention / Gated Residual, etc.) still runs
  - others: fallback (DataLoader-based, incurs PLM calls)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring
from tqdm import tqdm


def encode_all_news(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int):
    """
    Step 1: Pre-encode and cache all news.

    Role of this function:
    - Input: news raw features (title, mask, entity, category, etc.)
    - Process: run news_encoder (including the PLM) exactly once
    - Output: cache tensor of shape [num_news, hidden_dim]
    """
    num_news = len(mind_corpus.news_title_text)
    hidden_dim = model.news_embedding_dim
    news_representations = torch.zeros([num_news, hidden_dim], device='cuda')

    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for start_idx in tqdm(range(0, num_news, batch_size), desc='[1/2] Encoding all news'):
            end_idx = min(start_idx + batch_size, num_news)
            batch_indices = list(range(start_idx, end_idx))

            news_category = torch.tensor(mind_corpus.news_category[batch_indices], dtype=torch.long, device='cuda')
            news_subCategory = torch.tensor(mind_corpus.news_subCategory[batch_indices], dtype=torch.long, device='cuda')
            news_title_text = torch.from_numpy(mind_corpus.news_title_text[batch_indices]).long().cuda()
            news_title_mask = torch.from_numpy(mind_corpus.news_title_mask[batch_indices]).long().cuda()
            news_title_entity = torch.from_numpy(mind_corpus.news_title_entity[batch_indices]).long().cuda()
            news_content_text = torch.from_numpy(mind_corpus.news_abstract_text[batch_indices]).long().cuda()
            news_content_mask = torch.from_numpy(mind_corpus.news_abstract_mask[batch_indices]).long().cuda()
            news_content_entity = torch.from_numpy(mind_corpus.news_abstract_entity[batch_indices]).long().cuda()

            if model.use_user_embedding:
                batch_size_actual = len(batch_indices)
                user_embedding = torch.zeros(batch_size_actual, model.user_embedding.embedding_dim, device='cuda')
            else:
                user_embedding = None

            news_repr = model.news_encoder(
                news_title_text.unsqueeze(1),
                news_title_mask.unsqueeze(1),
                news_title_entity.unsqueeze(1),
                news_content_text.unsqueeze(1),
                news_content_mask.unsqueeze(1),
                news_content_entity.unsqueeze(1),
                news_category.unsqueeze(1),
                news_subCategory.unsqueeze(1),
                user_embedding
            )

            if news_repr.dim() == 3:
                news_repr = news_repr.squeeze(1)

            news_representations[start_idx:end_idx] = news_repr

    print(f'✓ News encoding completed: {news_representations.shape}')
    return news_representations


def _run_user_encoder_with_cache(model, user_encoder, history_repr, history_mask,
                                  user_category, candidate_repr, candidate_category,
                                  candidate_subCategory, user_embedding):
    """
    Run the user encoder with cached history representations.

    Bypasses the user encoder's news_encoder call and uses the already
    cached history_repr directly as input.

    Args:
        history_repr:       [B, max_history_num, D] - cached history news representations
        history_mask:       [B, max_history_num]    - history validity mask
        user_category:      [B, max_history_num]    - history news categories
        candidate_repr:     [B, 1, D]               - cached candidate representation
        candidate_category: [B, 1]                  - candidate category
    """
    encoder_name = user_encoder.__class__.__name__

    if encoder_name == 'MINER':
        # MINER can reuse poly_attention, which takes history_embedding directly
        # without going through news_encoder
        # -> bypass forward and run only poly_attention -> aggregation
        with torch.no_grad():
            news_num = candidate_repr.size(1)

            # Poly attention: feed history_repr directly
            interest_vectors = user_encoder.poly_attention(
                history_repr,
                history_mask,
                user_category,
                candidate_category
            )

            # Expand for candidate
            if candidate_category is not None:
                interest_vectors_exp = interest_vectors  # [B, N, K, D]
            else:
                interest_vectors_exp = interest_vectors.unsqueeze(1).expand(-1, news_num, -1, -1)

            # Aggregation
            if user_encoder.aggregation == 'weighted':
                W_e_h_c = F.gelu(user_encoder.W_e(candidate_repr))
                logits = torch.matmul(
                    W_e_h_c.unsqueeze(2),
                    interest_vectors_exp.transpose(2, 3)
                ).squeeze(2)
                alpha = F.softmax(logits, dim=2)
                user_repr = (alpha.unsqueeze(3) * interest_vectors_exp).sum(dim=2)
            elif user_encoder.aggregation == 'mean':
                user_repr = interest_vectors_exp.mean(dim=2)
            elif user_encoder.aggregation == 'max':
                user_repr = interest_vectors_exp.max(dim=2)[0]

        return user_repr

    elif encoder_name == 'ATT':
        # Attention aggregation
        with torch.no_grad():
            news_num = candidate_repr.size(1)
            user_repr = user_encoder.attention(history_repr).unsqueeze(1).expand(-1, news_num, -1)
        return user_repr

    elif encoder_name == 'MHSA':
        with torch.no_grad():
            news_num = candidate_repr.size(1)
            h = user_encoder.multiheadAttention(history_repr, history_repr, history_repr, history_mask)
            h = F.relu(F.dropout(user_encoder.affine(h), training=False, inplace=False), inplace=False)
            user_repr = user_encoder.attention(h).unsqueeze(1).repeat(1, news_num, 1)
        return user_repr

    elif encoder_name == 'GRU':
        with torch.no_grad():
            batch_size = history_repr.size(0)
            news_num = candidate_repr.size(1)
            user_history_num = history_mask.sum(dim=1, keepdim=False).long()
            sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)
            _, desorted_indices = torch.sort(sorted_indices, descending=False)
            nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)
            if nonzero_indices.size(0) == 0:
                return torch.zeros([batch_size, news_num, history_repr.size(-1)], device='cuda')
            from torch.nn.utils.rnn import pack_padded_sequence
            index = nonzero_indices[-1]
            if index + 1 == batch_size:
                sorted_history = history_repr.index_select(0, sorted_indices)
                packed = pack_padded_sequence(sorted_history, sorted_user_history_num.cpu(), batch_first=True)
                _, h = user_encoder.gru(packed)
                h = torch.tanh(user_encoder.dec(h.squeeze(0)))
                user_repr = h.index_select(0, desorted_indices)
            else:
                non_empty = sorted_indices[:index+1]
                sorted_history = history_repr.index_select(0, non_empty)
                packed = pack_padded_sequence(sorted_history, sorted_user_history_num[:index+1].cpu(), batch_first=True)
                _, h = user_encoder.gru(packed)
                h = torch.tanh(user_encoder.dec(h.squeeze(0)))
                user_repr = torch.cat([h, torch.zeros([batch_size - 1 - index, history_repr.size(-1)], device='cuda')],
                                      dim=0).index_select(0, desorted_indices)
            user_repr = user_repr.unsqueeze(1).expand(-1, news_num, -1)
        return user_repr

    elif encoder_name == 'PENR':
        # PENR: take history_repr directly and run only multi-view attention,
        # fully bypassing the news_encoder call
        with torch.no_grad():
            news_num = candidate_repr.size(1)
            batch_size = history_repr.size(0)
            max_history_num = history_repr.size(1)

            # history_mask expand [B, N, M]
            history_mask_exp = history_mask.unsqueeze(1).expand(-1, news_num, -1)  # [B, N, M]

            # Multi-view attention (reproduces PENR forward logic, with news_encoder call removed)
            view_interests = []
            for i in range(user_encoder.num_views):
                projected_single = torch.tanh(user_encoder.view_projections[i](history_repr))  # [B, M, query_dim]
                projected = projected_single.unsqueeze(1).expand(-1, news_num, -1, -1)         # [B, N, M, query_dim]
                scores = torch.matmul(projected, user_encoder.view_queries[i])                 # [B, N, M]
                scores = scores.masked_fill(history_mask_exp == 0, -1e9)
                alpha = F.softmax(scores, dim=2)                                               # [B, N, M]

                # [B*N, 1, M] @ [B*N, M, D] → [B, N, D]
                history_repr_exp = history_repr.unsqueeze(1).expand(-1, news_num, -1, -1)      # [B, N, M, D]
                u_i = torch.bmm(
                    alpha.reshape(batch_size * news_num, 1, -1),
                    history_repr_exp.reshape(batch_size * news_num, -1, history_repr.size(-1))
                ).reshape(batch_size, news_num, history_repr.size(-1))                         # [B, N, D]
                view_interests.append(u_i)

            # u: [B, N, N_a, D]
            user_repr = torch.stack(view_interests, dim=2)

            # store cached_history_embedding (used by the click predictor for popularity computation)
            user_encoder.cached_history_embedding = history_repr

        return user_repr  # [B, N, N_a, D]

    else:
        # unsupported encoder: return None -> caller handles fallback
        return None


class _CachedHistoryEncoder(nn.Module):
    """
    Temporary IdentityEncoder for the POPCORN fast path (subclasses nn.Module).

    Since a PyTorch nn.Module can only have nn.Module instances assigned as
    child modules, this must subclass nn.Module.

    In the POPCORN user encoder's forward():
        history_repr = self.news_encoder(user_title_text, ...)  # PLM call
    When this code runs, by passing the already cached history repr in place of
    user_title_text and having this encoder return it unchanged, the PLM can be
    fully bypassed.

    Because the original news_encoder's attributes (category_embedding,
    subCategory_embedding, etc.) must also be preserved, this is implemented by
    delegating attribute access to the original encoder.
    """
    def __init__(self, original_encoder: nn.Module):
        super().__init__()
        # Store _orig without registering it as an nn.Module (use object.__setattr__).
        # If we did self._orig = original_encoder, nn.Module.__setattr__ would
        # register it as a child module and duplicate the parameters.
        object.__setattr__(self, '_orig', original_encoder)
        object.__setattr__(self, 'news_embedding_dim', original_encoder.news_embedding_dim)

    def forward(self, user_title_text, *args, **kwargs):
        # the cache repr of shape [B, M, D] already arrives in place of user_title_text
        return user_title_text

    def __getattr__(self, name):
        # delegate original encoder attributes such as category_embedding, subCategory_embedding
        # nn.Module.__getattr__ first looks up its own _modules/_parameters/_buffers
        # and falls through to here if not found, so delegating to the original is safe
        try:
            orig = object.__getattribute__(self, '_orig')
            return getattr(orig, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def compute_scores_with_cache(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int,
                               mode: str, result_file: str, dataset: str,
                               compute_loss: bool = False, loss_fn=None,
                               config=None):
    """
    Fast evaluation leveraging cached news representations (history is also fully cached).

    Improvement progression:
      [v1] no caching:
           per batch -> news_encoder(history) x 50 + news_encoder(candidate) x 1 (all PLM)
      [v2] candidate caching:
           per batch -> news_encoder(history) x 50 (PLM, still slow) + lookup(candidate)
      [v3] full caching (current):
           encode_all_news() once -> per batch lookup(history) + lookup(candidate) (0 PLM calls)
    """
    assert mode in ['dev', 'test'], "mode must be 'dev' or 'test'"

    # TCCM (or POPCORN + use_tccm_addon): full forward() call (model owns publish_time/CTR buffers)
    if config is not None and (
        getattr(config, 'click_predictor', '') == 'TCCM'
        or (getattr(config, 'click_predictor', '') == 'POPCORN' and getattr(config, 'use_tccm_addon', False))
    ):
        mode_tag = 'TCCM' if getattr(config, 'click_predictor', '') == 'TCCM' else 'POPCORN+TCCM-addon'
        print(f'[{mode_tag}] evaluation mode (using model.forward())')
        return _tccm_compute_scores(
            model, mind_corpus, batch_size, mode, result_file, dataset, compute_loss, loss_fn
        )

    # Step 1: pre-encode all news (the PLM runs only here)
    news_cache = encode_all_news(model, mind_corpus, batch_size=batch_size * 4)

    # prepare news_category / news_subCategory as GPU tensors (for category-aware lookup)
    news_category_all = torch.tensor(mind_corpus.news_category, dtype=torch.long, device='cuda')  # [num_news]
    news_subCategory_all = torch.tensor(mind_corpus.news_subCategory, dtype=torch.long, device='cuda')

    # Step 2: load behaviors
    behaviors = mind_corpus.dev_behaviors if mode == 'dev' else mind_corpus.test_behaviors
    indices = mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices
    max_history_num = mind_corpus.max_history_num

    # check user encoder name
    encoder_name = model.user_encoder.__class__.__name__
    use_fast_path = encoder_name in ['MINER', 'ATT', 'MHSA', 'GRU', 'POPCORN', 'PENR']

    if not use_fast_path:
        print(f'[cache] {encoder_name} does not support fast path -> fallback (DataLoader-based)')
    else:
        print(f'[cache] {encoder_name} using fast path (bypassing news_encoder)')

    # Step 3: evaluation loop
    scores = torch.zeros([len(indices)], device='cuda')
    total_loss = 0.0
    total_samples = 0
    model.eval()
    torch.cuda.empty_cache()

    if use_fast_path and encoder_name != 'POPCORN':
        # Fast path (MINER/ATT/MHSA/GRU): process one item at a time, fully bypass news_encoder
        with torch.no_grad():
            for i in tqdm(range(len(behaviors)), desc=f'[2/2] {mode.upper()} evaluation', mininterval=1.0):
                behavior = behaviors[i]
                user_id_val     = behavior[0]
                history_ids     = behavior[1]
                history_mask_np = behavior[2]
                candidate_id    = behavior[3]

                hist_ids_t = torch.tensor(history_ids, dtype=torch.long, device='cuda')
                hist_repr  = news_cache[hist_ids_t].unsqueeze(0)                          # [1, M, D]
                hist_mask  = torch.tensor(history_mask_np, dtype=torch.float32, device='cuda').unsqueeze(0)
                cand_repr  = news_cache[candidate_id].unsqueeze(0).unsqueeze(0)           # [1, 1, D]
                cand_cat   = news_category_all[candidate_id].unsqueeze(0).unsqueeze(0)    # [1, 1]
                hist_cat   = news_category_all[hist_ids_t].unsqueeze(0)                   # [1, M]

                if model.use_user_embedding:
                    uid_t    = torch.tensor([user_id_val], dtype=torch.long, device='cuda')
                    user_emb = model.dropout(model.user_embedding(uid_t))
                else:
                    user_emb = None

                user_repr = _run_user_encoder_with_cache(
                    model, model.user_encoder,
                    hist_repr, hist_mask, hist_cat, cand_repr, cand_cat, None, user_emb
                )

                if model.click_predictor == 'dot_product':
                    if hasattr(model.config, 'use_I1') and model.config.use_I1:
                        d_half = cand_repr.size(-1) // 2
                        f_c = cand_repr[:, :, :d_half]
                        f_u = user_repr[:, :, :d_half] if user_repr.size(-1) != d_half else user_repr
                        logit = (f_u * f_c).sum(dim=-1)
                    else:
                        logit = (user_repr * cand_repr).sum(dim=-1)
                elif model.click_predictor == 'PENR':
                    # PENR click prediction: Bilinear + popularity
                    # user_repr: [1, 1, N_a, D], cand_repr: [1, 1, D]
                    num_views = user_repr.size(2)
                    emb_dim = user_repr.size(3)
                    news_rep_expanded = cand_repr.unsqueeze(2).expand(-1, -1, num_views, -1)  # [1, 1, N_a, D]
                    u_flat = user_repr.reshape(num_views, emb_dim)   # [N_a, D]
                    r_flat = news_rep_expanded.reshape(num_views, emb_dim)  # [N_a, D]
                    u_W_b = torch.matmul(u_flat, model.W_b)  # [N_a, D]
                    p_b_flat = (u_W_b * r_flat).sum(dim=1) + model.b_bilinear  # [N_a]
                    p_b = p_b_flat.unsqueeze(0).unsqueeze(0)  # [1, 1, N_a]
                    y_hat = torch.sigmoid(model.penr_ffn(p_b).squeeze(dim=2))  # [1, 1]
                    p_hat = model.popularity_predictor(cand_repr).squeeze(dim=2)  # [1, 1]
                    # history popularity (a_u)
                    hist_repr_pop = model.user_encoder.cached_history_embedding  # [1, M, D]
                    p_hat_hist = model.popularity_predictor(hist_repr_pop).squeeze(dim=2)  # [1, M]
                    hist_mask_1d = hist_mask  # [1, M]
                    p_hat_hist_masked = p_hat_hist * hist_mask_1d
                    hist_count = hist_mask_1d.sum(dim=1, keepdim=True).clamp(min=1)
                    a_u = (p_hat_hist_masked.sum(dim=1, keepdim=True) / hist_count).expand(-1, 1)  # [1, 1]
                    mu_clamped = torch.sigmoid(model.mu)
                    CTR = (1 - mu_clamped * a_u) * y_hat + mu_clamped * a_u * model.gamma * p_hat
                    logit = torch.log(CTR / (1 - CTR + 1e-8))  # [1, 1]                    
                else:
                    logit = (user_repr * cand_repr).sum(dim=-1)

                if compute_loss and loss_fn is not None:
                    batch_loss = loss_fn(logit)
                    total_loss += float(batch_loss)
                    total_samples += 1

                scores[i] = logit.squeeze()

    elif use_fast_path and encoder_name == 'POPCORN':
        # POPCORN Fast path: replace only the news_encoder with IdentityEncoder -> skip the PLM
        # POPCORN internal logic such as Top-K Attention and Gated Residual still runs
        #
        # Key trick:
        #   1. replace model.user_encoder.news_encoder with _CachedHistoryEncoder
        #   2. call user_encoder.forward(user_title_text=hist_repr, ...)
        #      -> internally runs self.news_encoder(hist_repr, ...)
        #      -> _CachedHistoryEncoder.__call__ returns hist_repr unchanged (no PLM)
        #   3. restore the original news_encoder after evaluation completes
        original_news_encoder = model.user_encoder.news_encoder
        cached_encoder = _CachedHistoryEncoder(original_news_encoder)
        model.user_encoder.news_encoder = cached_encoder
        print(f'[POPCORN cache] news_encoder replaced with IdentityEncoder')

        num_workers = min(8, os.cpu_count() // 2) if os.cpu_count() else 4
        dataloader = DataLoader(
            MIND_DevTest_Dataset(mind_corpus, mode),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        index = 0
        try:
            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc=f'[2/2] {mode.upper()} evaluation'):
                    (user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                     user_title_entity, user_content_text, user_content_mask, user_content_entity,
                     user_history_mask, user_history_graph, user_history_category_mask,
                     user_history_category_indices, news_category, news_subCategory,
                     news_title_text, news_title_mask, news_title_entity, news_content_text,
                     news_content_mask, news_content_entity) = batch_data

                    user_ID = user_ID.cuda(non_blocking=True)
                    user_category = user_category.cuda(non_blocking=True)
                    user_subCategory = user_subCategory.cuda(non_blocking=True)
                    user_history_mask = user_history_mask.cuda(non_blocking=True)
                    user_history_graph = user_history_graph.cuda(non_blocking=True)
                    user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
                    user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)

                    batch_size_actual = user_ID.size(0)

                    # Candidate: cache lookup
                    candidate_news_ids = []
                    for j in range(batch_size_actual):
                        behavior_idx = index + j
                        if behavior_idx < len(behaviors):
                            candidate_news_ids.append(behaviors[behavior_idx][3])
                    candidate_news_ids = torch.tensor(candidate_news_ids, dtype=torch.long, device='cuda')
                    candidate_repr = news_cache[candidate_news_ids].unsqueeze(1)  # [B, 1, D]

                    # History: cache lookup (inject the cache repr in place of user_title_text)
                    history_news_ids = []
                    for j in range(batch_size_actual):
                        behavior_idx = index + j
                        if behavior_idx < len(behaviors):
                            history_news_ids.append(behaviors[behavior_idx][1])  # hist_ids list
                    history_ids_t = torch.tensor(history_news_ids, dtype=torch.long, device='cuda')  # [B, M]
                    history_repr = news_cache[history_ids_t]  # [B, M, D]

                    news_category_input    = news_category.unsqueeze(1).cuda(non_blocking=True)
                    news_subCategory_input = news_subCategory.unsqueeze(1).cuda(non_blocking=True)

                    user_embedding = model.dropout(model.user_embedding(user_ID)) if model.use_user_embedding else None

                    # call POPCORN forward()
                    # inject history_repr in place of user_title_text -> _CachedHistoryEncoder returns it unchanged
                    user_repr = model.user_encoder(
                        user_title_text=history_repr,       # <- cache repr (bypasses PLM)
                        user_title_mask=user_history_mask,
                        user_title_entity=None,
                        user_content_text=None,
                        user_content_mask=None,
                        user_content_entity=None,
                        user_category=user_category,
                        user_subCategory=user_subCategory,
                        user_history_mask=user_history_mask,
                        user_history_graph=user_history_graph,
                        user_history_category_mask=user_history_category_mask,
                        user_history_category_indices=user_history_category_indices,
                        user_embedding=user_embedding,
                        candidate_news_representation=candidate_repr,
                        candidate_category=news_category_input,
                        candidate_subCategory=news_subCategory_input,
                    )

                    # Click prediction
                    if model.click_predictor == 'dot_product':
                        if hasattr(model.config, 'use_I1') and model.config.use_I1:
                            d_half = candidate_repr.size(-1) // 2
                            f_c = candidate_repr[:, :, :d_half]
                            f_u = user_repr[:, :, :d_half] if user_repr.size(-1) != d_half else user_repr
                            logits = (f_u * f_c).sum(dim=-1)
                        else:
                            logits = (user_repr * candidate_repr).sum(dim=-1)
                    elif model.click_predictor == 'POPCORN':
                        use_I1 = getattr(model.config, 'use_I1', False)
                        use_I3 = getattr(model.config, 'use_I3', False)
                        if use_I1:
                            d_half_c = candidate_repr.size(-1) // 2
                            f_c = candidate_repr[:, :, :d_half_c]
                            p_c = candidate_repr[:, :, d_half_c:]
                        else:
                            f_c = candidate_repr
                            p_c = None
                        if use_I1 and use_I3:
                            d_half_u = user_repr.size(-1) // 2
                            f_u = user_repr[:, :, :d_half_u]
                            p_u = user_repr[:, :, d_half_u:]
                        else:
                            f_u = user_repr
                            p_u = None
                        S_I = (f_u * f_c).sum(dim=-1)
                        if p_u is not None and p_c is not None:
                            S_P = F.cosine_similarity(p_u, p_c, dim=-1)
                            beta = getattr(model.config, 'pop_penalty_weight', 5.0)
                            penalty = beta * torch.sigmoid(model.config.popcorn_alpha * S_P)
                            logits = S_I - penalty
                        else:
                            logits = S_I
                    else:
                        logits = (user_repr * candidate_repr).sum(dim=-1)

                    if compute_loss and loss_fn is not None:
                        batch_loss = loss_fn(logits)
                        total_loss += float(batch_loss) * batch_size_actual
                        total_samples += batch_size_actual

                    scores[index: index + batch_size_actual] = logits.squeeze(1)
                    index += batch_size_actual
        finally:
            # always restore the original news_encoder
            model.user_encoder.news_encoder = original_news_encoder
            print(f'[POPCORN cache] original news_encoder restored')

    else:
        # Fallback: DataLoader-based (legacy approach, incurs PLM calls)
        num_workers = min(8, os.cpu_count() // 2) if os.cpu_count() else 4
        dataloader = DataLoader(
            MIND_DevTest_Dataset(mind_corpus, mode),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        index = 0

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f'[2/2] {mode.upper()} evaluation'):
                (user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                 user_title_entity, user_content_text, user_content_mask, user_content_entity,
                 user_history_mask, user_history_graph, user_history_category_mask,
                 user_history_category_indices, news_category, news_subCategory,
                 news_title_text, news_title_mask, news_title_entity, news_content_text,
                 news_content_mask, news_content_entity) = batch_data

                user_ID = user_ID.cuda(non_blocking=True)
                user_category = user_category.cuda(non_blocking=True)
                user_subCategory = user_subCategory.cuda(non_blocking=True)
                user_title_text = user_title_text.cuda(non_blocking=True)
                user_title_mask = user_title_mask.cuda(non_blocking=True)
                user_title_entity = user_title_entity.cuda(non_blocking=True)
                user_content_text = user_content_text.cuda(non_blocking=True)
                user_content_mask = user_content_mask.cuda(non_blocking=True)
                user_content_entity = user_content_entity.cuda(non_blocking=True)
                user_history_mask = user_history_mask.cuda(non_blocking=True)
                user_history_graph = user_history_graph.cuda(non_blocking=True)
                user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
                user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)

                batch_size_actual = user_ID.size(0)

                candidate_news_ids = []
                for j in range(batch_size_actual):
                    behavior_idx = index + j
                    if behavior_idx < len(behaviors):
                        candidate_news_ids.append(behaviors[behavior_idx][3])

                candidate_news_ids = torch.tensor(candidate_news_ids, dtype=torch.long, device='cuda')
                candidate_repr = news_cache[candidate_news_ids].unsqueeze(1)

                news_category_input = news_category.unsqueeze(1).cuda(non_blocking=True)
                news_subCategory_input = news_subCategory.unsqueeze(1).cuda(non_blocking=True)

                user_embedding = model.dropout(model.user_embedding(user_ID)) if model.use_user_embedding else None

                user_encoder_args = {
                    "user_title_text": user_title_text,
                    "user_title_mask": user_title_mask,
                    "user_title_entity": user_title_entity,
                    "user_content_text": user_content_text,
                    "user_content_mask": user_content_mask,
                    "user_content_entity": user_content_entity,
                    "user_category": user_category,
                    "user_subCategory": user_subCategory,
                    "user_history_mask": user_history_mask,
                    "user_history_graph": user_history_graph,
                    "user_history_category_mask": user_history_category_mask,
                    "user_history_category_indices": user_history_category_indices,
                    "user_embedding": user_embedding,
                    "candidate_news_representation": candidate_repr
                }

                if model.user_encoder.__class__.__name__ in ['MINER', 'POPCORN']:
                    user_encoder_args["candidate_category"] = news_category_input
                    if model.user_encoder.__class__.__name__ == 'POPCORN':
                        user_encoder_args["candidate_subCategory"] = news_subCategory_input

                user_repr = model.user_encoder(**user_encoder_args)

                if model.click_predictor == 'dot_product':
                    if hasattr(model.config, 'use_I1') and model.config.use_I1:
                        d_half = candidate_repr.size(-1) // 2
                        f_c = candidate_repr[:, :, :d_half]
                        f_u = user_repr[:, :, :d_half] if user_repr.size(-1) != d_half else user_repr
                        logits = (f_u * f_c).sum(dim=-1)
                    else:
                        logits = (user_repr * candidate_repr).sum(dim=-1)
                elif model.click_predictor == 'mlp':
                    context = model.dropout(F.relu(model.mlp(torch.cat([user_repr, candidate_repr], dim=2))))
                    logits = model.out(context).squeeze(dim=2)
                elif model.click_predictor == 'POPCORN':
                    use_I1 = getattr(model.config, 'use_I1', False)
                    use_I3 = getattr(model.config, 'use_I3', False)
                    if use_I1:
                        d_half_c = candidate_repr.size(-1) // 2
                        f_c = candidate_repr[:, :, :d_half_c]
                        p_c = candidate_repr[:, :, d_half_c:]
                    else:
                        f_c = candidate_repr
                        p_c = None
                    if use_I1 and use_I3:
                        d_half_u = user_repr.size(-1) // 2
                        f_u = user_repr[:, :, :d_half_u]
                        p_u = user_repr[:, :, d_half_u:]
                    else:
                        f_u = user_repr
                        p_u = None
                    S_I = (f_u * f_c).sum(dim=-1)
                    if p_u is not None and p_c is not None:
                        S_P = F.cosine_similarity(p_u, p_c, dim=-1)
                        beta = getattr(model.config, 'pop_penalty_weight', 5.0)
                        penalty = beta * torch.sigmoid(model.config.popcorn_alpha * S_P)
                        logits = S_I - penalty
                    else:
                        logits = S_I
                else:
                    logits = (user_repr * candidate_repr).sum(dim=-1)

                scores[index: index + batch_size_actual] = logits.squeeze(1)

                if compute_loss and loss_fn is not None:
                    batch_loss = loss_fn(logits)
                    total_loss += float(batch_loss) * batch_size_actual
                    total_samples += batch_size_actual

                index += batch_size_actual

    # save results and evaluate
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, idx in enumerate(indices):
        sub_scores[idx].append([scores[i], len(sub_scores[idx])])

    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))

    if dataset != 'large' or mode != 'test':
        with open(f'{mode}/ref/truth-{dataset}.txt', 'r', encoding='utf-8') as truth_f, \
             open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)

        if compute_loss:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            return auc, mrr, ndcg5, ndcg10, avg_loss
        else:
            return auc, mrr, ndcg5, ndcg10
    else:
        if compute_loss:
            return None, None, None, None, 0.0
        else:
            return None, None, None, None


def _tccm_compute_scores(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int,
                          mode: str, result_file: str, dataset: str,
                          compute_loss: bool, loss_fn):
    """
    TCCM-specific evaluation path.

    - Uses MIND_DevTest_Dataset(return_tccm=True) to additionally receive the candidate news index and current_time.
    - Calls model.forward(..., news_indices_for_tccm=..., news_current_time=...).
    - When `--tccm_do_intervention` is enabled, s_p is automatically replaced with the training-set average inside model.forward.
    """
    behaviors = mind_corpus.dev_behaviors if mode == 'dev' else mind_corpus.test_behaviors
    indices = mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices
    scores = torch.zeros([len(indices)], device='cuda')
    total_loss = 0.0
    total_samples = 0

    num_workers = min(8, os.cpu_count() // 2) if os.cpu_count() else 4
    dataloader = DataLoader(
        MIND_DevTest_Dataset(mind_corpus, mode, return_tccm=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    # Apply intervention only at final test time. During training, dev evaluation uses raw s_p
    # so that best-epoch selection reflects the model's actual ranking ability.
    # The flag defaults to True (= preserve legacy behavior); only this function sets it explicitly based on mode.
    _prev_allow_intv = getattr(model, '_tccm_allow_intervention', True)
    model._tccm_allow_intervention = (mode == 'test')
    torch.cuda.empty_cache()
    index = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f'[TCCM] {mode.upper()} evaluation', mininterval=1.0):
            (user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
             user_title_entity, user_content_text, user_content_mask, user_content_entity,
             user_history_mask, user_history_graph, user_history_category_mask,
             user_history_category_indices, news_category, news_subCategory,
             news_title_text, news_title_mask, news_title_entity, news_content_text,
             news_content_mask, news_content_entity,
             tccm_news_indices, tccm_current_time) = batch_data

            # Move to GPU (non-blocking)
            user_ID = user_ID.cuda(non_blocking=True)
            user_category = user_category.cuda(non_blocking=True)
            user_subCategory = user_subCategory.cuda(non_blocking=True)
            user_title_text = user_title_text.cuda(non_blocking=True)
            user_title_mask = user_title_mask.cuda(non_blocking=True)
            user_title_entity = user_title_entity.cuda(non_blocking=True)
            user_content_text = user_content_text.cuda(non_blocking=True)
            user_content_mask = user_content_mask.cuda(non_blocking=True)
            user_content_entity = user_content_entity.cuda(non_blocking=True)
            user_history_mask = user_history_mask.cuda(non_blocking=True)
            user_history_graph = user_history_graph.cuda(non_blocking=True)
            user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
            user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)
            # DevTest has a single candidate per sample → unsqueeze to [B, 1, ...]
            news_category = news_category.unsqueeze(1).cuda(non_blocking=True)
            news_subCategory = news_subCategory.unsqueeze(1).cuda(non_blocking=True)
            news_title_text = news_title_text.unsqueeze(1).cuda(non_blocking=True)
            news_title_mask = news_title_mask.unsqueeze(1).cuda(non_blocking=True)
            news_title_entity = news_title_entity.unsqueeze(1).cuda(non_blocking=True)
            news_content_text = news_content_text.unsqueeze(1).cuda(non_blocking=True)
            news_content_mask = news_content_mask.unsqueeze(1).cuda(non_blocking=True)
            news_content_entity = news_content_entity.unsqueeze(1).cuda(non_blocking=True)
            tccm_news_indices = tccm_news_indices.unsqueeze(1).cuda(non_blocking=True)     # [B, 1]
            tccm_current_time = tccm_current_time.cuda(non_blocking=True)                   # [B]

            logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                           user_title_entity, user_content_text, user_content_mask, user_content_entity,
                           user_history_mask, user_history_graph, user_history_category_mask,
                           user_history_category_indices,
                           news_category, news_subCategory, news_title_text, news_title_mask,
                           news_title_entity, news_content_text, news_content_mask, news_content_entity,
                           news_indices_for_tccm=tccm_news_indices,
                           news_current_time=tccm_current_time)      # [B, 1]
            batch_size_actual = user_ID.size(0)
            scores[index: index + batch_size_actual] = logits.squeeze(1)
            if compute_loss and loss_fn is not None:
                batch_loss = loss_fn(logits)
                total_loss += float(batch_loss) * batch_size_actual
                total_samples += batch_size_actual
            index += batch_size_actual

    # restore the flag after the evaluation loop ends (keep an externally set value if present, else default True)
    model._tccm_allow_intervention = _prev_allow_intv

    # save results and evaluate (duplicates the shared logic from the tail of compute_scores_with_cache)
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, idx in enumerate(indices):
        sub_scores[idx].append([scores[i], len(sub_scores[idx])])

    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))

    if dataset != 'large' or mode != 'test':
        with open(f'{mode}/ref/truth-{dataset}.txt', 'r', encoding='utf-8') as truth_f, \
             open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)
        if compute_loss:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            return auc, mrr, ndcg5, ndcg10, avg_loss
        return auc, mrr, ndcg5, ndcg10
    if compute_loss:
        return None, None, None, None, 0.0
    return None, None, None, None
