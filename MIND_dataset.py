from MIND_corpus import MIND_Corpus
import time
from config import Config
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, return_news_indices: bool = False, return_popularity_labels: bool = False, return_tccm: bool = False):
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.news_popularity = corpus.news_popularity  # PENR popularity labels
        self.news_popularity_class = corpus.news_popularity_class  # POPCORN popularity class labels
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)
        self.return_news_indices = return_news_indices  # PENR mode: return news indices for popularity loss
        self.return_popularity_labels = return_popularity_labels  # POPCORN mode: return popularity class labels
        # TCCM mode: also return news indices (for publish_time/CTR lookup) + behavior time bucket
        self.return_tccm = return_tccm
        if self.return_tccm:
            self.train_behavior_time_bucket = corpus.train_behavior_time_bucket

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4]
            news_num = len(negative_samples)
            if news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % news_num]
            else:
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    # user_ID                       : [1]
    # user_category                 : [max_history_num]
    # usre_subCategory              : [max_history_num]
    # user_title_text               : [max_history_num, max_title_length]
    # user_title_mask               : [max_history_num, max_title_length]
    # user_title_entity             : [max_history_num, max_title_length]
    # user_abstract_text            : [max_history_num, max_abstract_length]
    # user_abstract_mask            : [max_history_num, max_abstract_length]
    # user_abstract_entity          : [max_history_num, max_abstract_length]
    # user_history_mask             : [max_history_num]
    # user_history_graph            : [max_history_num, max_history_num]
    # user_history_category_mask    : [category_num + 1]
    # user_history_category_indices : [max_history_num]
    # news_category                 : [1 + negative_sample_num]
    # news_subCategory              : [1 + negative_sample_num]
    # news_title_text               : [1 + negative_sample_num, max_title_length]
    # news_title_mask               : [1 + negative_sample_num, max_title_length]
    # news_title_entity             : [1 + negative_sample_num, max_title_length]
    # news_abstract_text            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_mask            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_entity          : [1 + negative_sample_num, max_abstract_length]
    # sample_news_indices                  : [1 + negative_sample_num] - ONLY when return_news_indices=True (PENR)
    # sample_candidate_popularity_labels   : [1 + negative_sample_num] - ONLY when return_popularity_labels=True (POPCORN)
    # sample_history_popularity_labels     : [max_history_num] - ONLY when return_popularity_labels=True (POPCORN)
    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]

        base_return = (train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], train_behavior[2], self.user_history_graph[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
                       self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index])

        if self.return_news_indices:
            # PENR: Convert sample_index (list) to numpy array for proper tensor conversion
            return base_return + (np.array(sample_index, dtype=np.int64),)

        # POPCORN popularity labels (if requested)
        popularity_tuple = ()
        if self.return_popularity_labels:
            # POPCORN: Return popularity class labels for BOTH candidate AND history news
            candidate_pop_labels = self.news_popularity_class[sample_index]    # [1+negative_sample_num]
            history_pop_labels = self.news_popularity_class[history_index]     # [max_history_num]
            popularity_tuple = (candidate_pop_labels, history_pop_labels)

        # TCCM indices & time bucket (if requested)
        tccm_tuple = ()
        if self.return_tccm:
            candidate_news_indices = np.array(sample_index, dtype=np.int64)    # [1+negative_sample_num]
            news_current_time = np.int64(self.train_behavior_time_bucket[behavior_index])  # scalar
            tccm_tuple = (candidate_news_indices, news_current_time)

        # Order: base + popularity + tccm
        # - POPCORN only      : base + popularity
        # - TCCM only         : base + tccm
        # - POPCORN+TCCM addon: base + popularity + tccm
        return base_return + popularity_tuple + tccm_tuple

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, mode: str, return_tccm: bool = False):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)
        # TCCM mode: also return news indices (for publish_time/CTR lookup) + behavior time bucket
        self.return_tccm = return_tccm
        if self.return_tccm:
            self.behavior_time_bucket = corpus.dev_behavior_time_bucket if mode == 'dev' else corpus.test_behavior_time_bucket

    # user_ID                        : [1]
    # user_category                  : [max_history_num]
    # user_subCategory               : [max_history_num]
    # user_title_text                : [max_history_num, max_title_length]
    # user_title_mask                : [max_history_num, max_title_length]
    # user_title_entity              : [max_history_num, max_title_length]
    # user_abstract_text             : [max_history_num, max_abstract_length]
    # user_abstract_mask             : [max_history_num, max_abstract_length]
    # user_abstract_entity           : [max_history_num, max_abstract_length]
    # user_history_mask              : [max_history_num]
    # user_history_graph             : [max_history_num, max_history_num]
    # user_history_category_mask     : [category_num + 1]
    # user_history_category_indices  : [max_history_num]
    # candidate_news_category        : [1]
    # candidate_news_subCategory     : [1]
    # candidate_news_title_text      : [max_title_length]
    # candidate_news_title_mask      : [max_title_length]
    # candidate_news_title_entity    : [max_title_lenght]
    # candidate_news_abstract_text   : [max_abstract_length]
    # candidate_news_abstract_mask   : [max_abstract_length]
    # candidate_news_abstract_entity : [max_abstract_length]
    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]
        behavior_index = behavior[4]
        base_return = (behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], behavior[2], self.user_history_graph[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index])
        if self.return_tccm:
            # DevTest has a single candidate per sample → scalar index
            candidate_news_indices = np.int64(candidate_news_index)
            news_current_time = np.int64(self.behavior_time_bucket[behavior_index])
            return base_return + (candidate_news_indices, news_current_time)
        return base_return

    def __len__(self):
        return self.num
